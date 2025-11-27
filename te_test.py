#!/usr/bin/env python
import argparse
import time
import warnings
import logging
import os
import json
import random
import datetime

import torch

from kandinsky.utils import set_hf_token
from kandinsky import get_T2V_pipeline
from kandinsky.models.te_dit import get_dit as get_te_dit


# -------------------------------
# Utility helpers
# -------------------------------

def disable_warnings():
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    torch._logging.set_logs(
        dynamo=logging.ERROR,
        dynamic=logging.ERROR,
        aot=logging.ERROR,
        inductor=logging.ERROR,
        guards=False,
        recompiles=False,
    )


def set_seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def validate_args(args):
    # Minimal sanity + keep things compatible with the VAE/patching
    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"Width/height must be positive. Got ({args.width}, {args.height}).")
    if args.width % 64 != 0 or args.height % 64 != 0:
        raise ValueError(
            f"Width/height must be multiples of 64 for this model. "
            f"Got ({args.width}, {args.height})."
        )
    if args.video_duration < 0:
        raise ValueError(f"video_duration must be >= 0. Got {args.video_duration}.")


def mode_prefix_and_logfile(args):
    """
    Decide:
      - logical 'mode' label,
      - filename prefix,
      - JSONL logfile name.
    """
    if args.no_te:
        mode = "baseline_bf16"
        prefix = "bf16_"
        jsonl_name = "bf16_test.jsonl"
    else:
        if args.disable_fp8:
            mode = "te_bf16"
            prefix = "te_bf16_"
            jsonl_name = "te_bf16_test.jsonl"
        else:
            mode = "te_fp8"
            prefix = "te_fp8_"
            jsonl_name = "te_fp8_test.jsonl"
    return mode, prefix, jsonl_name


# -------------------------------
# Relax upstream resolution guard
# -------------------------------

def relax_resolution_check(pipe, height: int, width: int):
    """
    Upstream Kandinsky5T2VPipeline enforces a fixed set of (H, W) pairs via
    self.RESOLUTIONS[self.resolution].

    We *do not* modify kandinsky/t2v_pipeline.py.
    Instead, we patch the instance so that the requested (height, width)
    is considered valid for the current resolution.
    """
    if not hasattr(pipe, "RESOLUTIONS") or not hasattr(pipe, "resolution"):
        return

    res_table = getattr(pipe, "RESOLUTIONS", None)
    res_key = getattr(pipe, "resolution", None)
    if not isinstance(res_table, dict):
        return
    if res_key not in res_table:
        return

    hw = (height, width)
    cur_list = list(res_table.get(res_key, []))
    if hw not in cur_list:
        cur_list.append(hw)
        res_table[res_key] = cur_list
        pipe.RESOLUTIONS = res_table
        print(f"[te_test] Relaxed resolution guard: added {hw} to RESOLUTIONS[{res_key}].")


# -------------------------------
# CLI parsing
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TE-enhanced Kandinsky 5 T2V test harness"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file for the model (e.g. ./configs/te_fp8_k5_pro_t2v_10s_sft_hd.yaml)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Positive text prompt",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        help="Negative prompt for classifier-free guidance",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Video width in pixels (must be multiple of 64)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height in pixels (must be multiple of 64)",
    )
    parser.add_argument(
        "--video_duration",
        type=int,
        default=5,
        help="Duration of the video in whole seconds (0 => image)",
    )
    parser.add_argument(
        "--expand_prompt",
        type=int,
        default=0,
        help="Whether to use prompt expansion (1) or not (0).",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Number of sampling steps (defaults to config num_steps if None).",
    )
    parser.add_argument(
        "--guidance_weight",
        type=float,
        default=None,
        help="CFG guidance weight (defaults to config value if None).",
    )
    parser.add_argument(
        "--scheduler_scale",
        type=float,
        default=10.0,
        help="Scheduler scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (<= 2^32-1). If None, a 31-bit seed is sampled.",
    )

    # TE controls
    parser.add_argument(
        "--no_te",
        action="store_true",
        default=False,
        help="Disable Transformer Engine; use baseline bf16 DiT.",
    )
    parser.add_argument(
        "--disable_fp8",
        action="store_true",
        default=False,
        help="Use TE-backed DiT but *without* FP8 (te_bf16 mode).",
    )
    parser.add_argument(
        "--te_backend",
        type=str,
        default="te-fp8",
        choices=["te-fp8", "te_fp8"],
        help="TE DiT backend identifier.",
    )

    # Misc existing flags
    parser.add_argument(
        "--offload",
        action="store_true",
        default=False,
        help="Offload models to save memory or not.",
    )
    parser.add_argument(
        "--magcache",
        action="store_true",
        default=False,
        help="Use MagCache (for 50-step models only).",
    )
    parser.add_argument(
        "--qwen_quantization",
        action="store_true",
        default=False,
        help="Use quantized Qwen2.5-VL model (4-bit).",
    )
    parser.add_argument(
        "--attention_engine",
        type=str,
        default="auto",
        help="Full attention algorithm for <=5s generation.",
        choices=["flash_attention_2", "flash_attention_3", "sdpa", "sage", "auto"],
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HF token if needed for restricted models (e.g. FLUX.1-dev VAE).",
    )

    return parser.parse_args()


# -------------------------------
# Pipeline construction
# -------------------------------

def build_pipeline(args, device_map):
    # Base T2V pipeline (upstream)
    pipe = get_T2V_pipeline(
        device_map=device_map,
        conf_path=args.config,
        offload=args.offload,
        magcache=args.magcache,
        quantized_qwen=args.qwen_quantization,
        attention_engine=args.attention_engine,
    )

    if args.no_te:
        print("[te_test] Using baseline bf16 DiT (no TE).")
        return pipe

    # TE path: wrap the *existing, weight-loaded* DiT with our TE/FP8 wrapper
    print(
        f"[te_test] Swapping baseline DiT -> TE DiT "
        f"(backend={args.te_backend}, enable_fp8={not args.disable_fp8})..."
    )
    base_dit = pipe.dit.to(device_map["dit"])
    te_dit = get_te_dit(
        base_dit,
        backend=args.te_backend,
        enable_fp8=not args.disable_fp8,
    )
    pipe.dit = te_dit.to(device_map["dit"])
    return pipe


# -------------------------------
# JSONL logging
# -------------------------------

def append_jsonl_log(
    args,
    seed: int,
    mode: str,
    output_path: str,
    timestamp: str,
    jsonl_path: str,
):
    record = {
        "timestamp": timestamp,
        "config_path": os.path.abspath(args.config),
        "video_path": output_path,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": int(seed),
        "width": int(args.width),
        "height": int(args.height),
        "time_length": int(args.video_duration),
        "scheduler_scale": float(args.scheduler_scale),
        "num_steps": None if args.sample_steps is None else int(args.sample_steps),
        "guidance_weight": None if args.guidance_weight is None else float(args.guidance_weight),
        "mode": mode,  # "baseline_bf16", "te_bf16", "te_fp8"
        "te_backend": None if args.no_te else args.te_backend,
        "fp8_enabled": False if args.no_te else (not args.disable_fp8),
        "expand_prompts": bool(args.expand_prompt),
    }

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------------------
# Main
# -------------------------------

def main():
    disable_warnings()
    args = parse_args()
    validate_args(args)

    if args.hf_token:
        set_hf_token(args.hf_token)

    # Seed: explicit or 31-bit random
    if args.seed is None:
        seed = random.randrange(0, 2**31)
    else:
        seed = int(args.seed) & 0xFFFFFFFF
    set_seed_all(seed)

    mode, prefix, jsonl_name = mode_prefix_and_logfile(args)

    print("=== TE T2V Test ===")
    print(f"Config:         {args.config}")
    print(f"Resolution:     {args.width}x{args.height}")
    print(f"Duration:       {args.video_duration}s")
    print(f"Sample steps:   {args.sample_steps}")
    print(f"Mode:           {mode}")
    print(f"TE backend:     {args.te_backend if not args.no_te else 'none'}")
    print(f"FP8 enabled:    {False if args.no_te else (not args.disable_fp8)}")
    print(f"Seed:           {seed}")
    print()

    device_map = {"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"}
    pipe = build_pipeline(args, device_map)

    # Relax upstream size guard so we can use arbitrary multiples of 64
    relax_resolution_check(pipe, args.height, args.width)

    # Output naming: prefix + timestamp + seed + truncated prompt
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = args.prompt.replace(" ", "_")
    if len(safe_prompt) > 64:
        safe_prompt = safe_prompt[:64]

    out_dir = "k5pro_output"
    os.makedirs(out_dir, exist_ok=True)

    output_filename = f"{prefix}{timestamp}_seed{seed}_{safe_prompt}.mp4"
    output_path = os.path.join(out_dir, output_filename)

    print(f"[te_test] Output will be saved to: {output_path}")

    start_time = time.perf_counter()
    try:
        _ = pipe(
            args.prompt,
            time_length=args.video_duration,
            width=args.width,
            height=args.height,
            num_steps=args.sample_steps,
            guidance_weight=args.guidance_weight,
            scheduler_scale=args.scheduler_scale,
            negative_caption=args.negative_prompt,
            expand_prompts=bool(args.expand_prompt),
            save_path=output_path,
            seed=seed,
        )
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"TIME ELAPSED: {elapsed:.2f} seconds")

    jsonl_path = os.path.join(out_dir, jsonl_name)
    append_jsonl_log(
        args=args,
        seed=seed,
        mode=mode,
        output_path=output_path,
        timestamp=timestamp,
        jsonl_path=jsonl_path,
    )

    print(f"[te_test] Logged run to: {jsonl_path}")
    print(f"Generated file is saved to {output_path}")


if __name__ == "__main__":
    main()

