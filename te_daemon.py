#!/usr/bin/env python
import argparse
import datetime
import json
import logging
import os
import random
import textwrap
import time
import warnings
from typing import Any

import inspect

import torch
import torchvision  # imported but only used indirectly via the pipeline

from kandinsky.utils import set_hf_token
from kandinsky import get_T2V_pipeline
from kandinsky.models.te_dit import get_dit as get_te_dit


# -------------------------------
# Global resolution map (matches t2v_pipeline)
# -------------------------------

RESOLUTIONS = {
    512: [
        (512, 512),
        (512, 768),
        (768, 512),
    ],
    1024: [
        (1024, 1024),
        (640, 1408),
        (1408, 640),
        (768, 1280),
        (1280, 768),
        (896, 1152),
        (1152, 896),
    ],
}

NEGATIVE_PROMPT_DEFAULT = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
    "worst quality, low quality, ugly, deformed, walking backwards"
)


# -------------------------------
# Utilities
# -------------------------------

def disable_warnings():
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    if hasattr(torch, "_logging"):
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


def mode_prefix_and_logfile(args):
    """
    Match te_test naming so JSONLs line up:

      - baseline_bf16 -> bf16_test.jsonl
      - te_bf16       -> te_bf16_test.jsonl
      - te_fp8        -> te_fp8_test.jsonl
    """
    if args.no_te:
        mode_label = "baseline_bf16"
        prefix = "bf16_"
        jsonl_name = "bf16_test.jsonl"
    else:
        if args.disable_fp8:
            mode_label = "te_bf16"
            prefix = "te_bf16_"
            jsonl_name = "te_bf16_test.jsonl"
        else:
            mode_label = "te_fp8"
            prefix = "te_fp8_"
            jsonl_name = "te_fp8_test.jsonl"
    return mode_label, prefix, jsonl_name


def safe_prompt_slug(prompt: str, max_len: int = 64) -> str:
    slug = prompt.replace(" ", "_")
    if len(slug) > max_len:
        slug = slug[:max_len]
    return slug or "empty_prompt"


def append_jsonl(record: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_help():
    msg = """
Commands:
  help                        Show this help.
  show                        Show current state and generation settings.
  prompt <text>               Set the positive prompt.
  neg <text>                  Set the negative prompt.
  set <key> <value>           Set a generation setting:
                                width, height, video_duration,
                                steps, guidance_weight,
                                scheduler_scale,
                                negative_prompt,
                                expand_prompt (bool),
                                seed (int, -1 => random),
                                progress (bool)
                              Special: `set magcache <bool>` rebuilds the pipeline.
  gen [prompt text ...]       Generate with the given prompt (or current prompt if omitted).
                              Will ask for confirmation before running.
  run                         Same as 'gen' using the current prompt.
  exit / quit                 Terminate the daemon.

Examples:
  prompt a tiny glossy cube spinning on a black background
  set width 512
  set height 512
  set video_duration 5
  set steps 20
  set guidance_weight 5.0
  set scheduler_scale 10.0
  set seed -1
  gen a slow cinematic pan over a detailed fantasy landscape
"""
    print(textwrap.dedent(msg).strip())


# -------------------------------
# Pipeline construction (mirrors te_test.py)
# -------------------------------

def build_pipeline(args, device_map):
    """
    Build the base T2V pipeline and, unless --no_te is set, wrap the DiT
    with our TE/FP8 module. This is intentionally kept in lock-step with
    te_test.py so both paths see the exact same model wiring.
    """
    pipe = get_T2V_pipeline(
        device_map=device_map,
        conf_path=args.config,
        offload=args.offload,
        magcache=args.magcache,
        quantized_qwen=args.qwen_quantization,
        attention_engine=args.attention_engine,
    )

    if args.no_te:
        print("[te_daemon] Using baseline bf16 DiT (no TE).")
        return pipe

    print(
        f"[te_daemon] Swapping baseline DiT -> TE DiT "
        f"(backend={args.te_backend}, enable_fp8={not args.disable_fp8})..."
    )
    # Move the baseline DiT onto the target device *before* wrapping with TE.
    base_dit = pipe.dit.to(device_map["dit"])
    te_dit = get_te_dit(
        base_dit,
        backend=args.te_backend,
        enable_fp8=not args.disable_fp8,
    )
    pipe.dit = te_dit.to(device_map["dit"])
    return pipe


# -------------------------------
# Daemon
# -------------------------------

def main():
    disable_warnings()

    parser = argparse.ArgumentParser(
        description="Persistent TE-FP8 Kandinsky 5 T2V daemon"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config path (e.g. ./configs/te_fp8_k5_pro_t2v_10s_sft_hd.yaml)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HF token if needed for gated components.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        default=False,
        help="Offload models between CPU/GPU to save memory.",
    )
    parser.add_argument(
        "--magcache",
        action="store_true",
        default=False,
        help="Enable MagCache (for 50-step configs).",
    )
    parser.add_argument(
        "--qwen_quantization",
        action="store_true",
        default=False,
        help="Use quantized Qwen2.5-VL (4-bit).",
    )
    parser.add_argument(
        "--attention_engine",
        type=str,
        default="auto",
        choices=["flash_attention_2", "flash_attention_3", "sdpa", "sage", "auto"],
        help="Attention backend for <=5s visual attention.",
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
        help="Use TE-backed DiT but without FP8 (te_bf16).",
    )
    parser.add_argument(
        "--te_backend",
        type=str,
        default="te-fp8",
        choices=["te-fp8", "te_fp8"],
        help="TE DiT backend identifier.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="k5pro_output",
        help="Directory to store generated videos and logs.",
    )
    parser.add_argument(
        "--log_jsonl",
        type=str,
        default=None,
        help="Optional explicit JSONL log path. Default: <out_dir>/<mode_label>_test.jsonl",
    )

    args = parser.parse_args()

    if args.hf_token:
        set_hf_token(args.hf_token)

    # Device map: make sure "dit" exists so we never hit KeyError('dit')
    device_map = {
        "dit": "cuda:0",
        "vae": "cuda:0",
        "text_embedder": "cuda:0",
    }

    # Build base T2V pipeline using the same logic as te_test.py
    pipe = build_pipeline(args, device_map)

    mode_label, prefix, jsonl_name = mode_prefix_and_logfile(args)

    # --- Derive defaults from config ---

    conf = pipe.conf
    resolution = getattr(conf.metrics, "resolution", None)

    if isinstance(resolution, int) and resolution in RESOLUTIONS:
        # First supported (height, width) pair
        default_height, default_width = RESOLUTIONS[resolution][0]
    elif isinstance(resolution, int):
        # Fallback: square
        default_height = default_width = resolution
    else:
        # Final fallback
        default_height = default_width = 512

    cfg_default = float(getattr(conf.model, "guidance_weight", 5.0))

    # JSONL path (matching te_test naming)
    os.makedirs(args.out_dir, exist_ok=True)
    log_jsonl_path = args.log_jsonl or os.path.join(args.out_dir, jsonl_name)

    print()
    print("=== TE T2V Daemon ===")
    print(f"Config:         {args.config}")
    print(f"mode:           t2v")
    print(f"mode_label:     {mode_label}")
    print(f"TE enabled:     {not args.no_te}")
    print(f"TE backend:     {args.te_backend if not args.no_te else 'none'}")
    print(f"FP8 enabled:    {False if args.no_te else (not args.disable_fp8)}")
    print(f"Offload:        {args.offload}")
    print(f"MagCache:       {args.magcache}")
    print(f"Qwen quant:     {args.qwen_quantization}")
    print(f"Attention eng:  {args.attention_engine}")
    print(f"Default res:    {default_width}x{default_height} (from resolution={resolution})")
    print(f"Default CFG:    {cfg_default}")
    print(f"Output dir:     {args.out_dir}")
    print(f"JSONL log:      {log_jsonl_path}")
    print("=====================")
    print()
    print_help()
    print()

    # Daemon state
    state: dict[str, Any] = {
        "mode": "t2v",
        "mode_label": mode_label,            # "baseline_bf16", "te_bf16", "te_fp8"
        "te_mode": not args.no_te,           # True => TE path enabled
        "te_backend": args.te_backend if not args.no_te else "none",
        "fp8_enabled": False if args.no_te else (not args.disable_fp8),
        "out_dir": args.out_dir,
        "config_path": os.path.abspath(args.config),
        "jsonl_path": log_jsonl_path,
        "prompt": "test prompt",
        "magcache": args.magcache,
        "defaults": {
            "width": default_width,
            "height": default_height,
            "video_duration": 3,
            "steps": 20,
            "guidance_weight": cfg_default,
            "scheduler_scale": 10.0,
            "negative_prompt": NEGATIVE_PROMPT_DEFAULT,
            "expand_prompt": False,
            # seed >= 0 => fixed; seed < 0 => random per gen
            "seed": 34567,
            "progress": True,
        },
    }

    def print_state():
        print("\n[te_daemon] Current state:")
        print(f"  mode:           {state['mode']} (label={state['mode_label']})")
        print(f"  te_mode:        {state['te_mode']}")
        print(f"  te_backend:     {state['te_backend']}")
        print(f"  fp8_enabled:    {state['fp8_enabled']}")
        print(f"  magcache:       {state['magcache']}")
        print(f"  out_dir:        {state['out_dir']}")
        print(f"  config_path:    {state['config_path']}")
        print(f"  prompt:         {state['prompt']}")
        print("  generation settings:")
        d = state["defaults"]
        for key in [
            "width",
            "height",
            "video_duration",
            "steps",
            "guidance_weight",
            "scheduler_scale",
            "negative_prompt",
            "expand_prompt",
            "seed",
            "progress",
        ]:
            print(f"    {key}: {d[key]}")
        print(f"  jsonl_path:     {state['jsonl_path']}\n")

    # --- Helpers inside main ---

    def parse_bool(value_str: str) -> Any:
        v = value_str.lower()
        if v in ("true", "1", "yes", "on", "y"):
            return True
        if v in ("false", "0", "no", "off", "n"):
            return False
        return None

    # --- REPL loop ---

    while True:
        try:
            line = input("te-daemon> ")
        except EOFError:
            print()
            break

        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0]
        rest = parts[1:]

        if cmd in {"exit", "quit"}:
            break

        if cmd == "help":
            print_help()
            continue

        if cmd == "show":
            print_state()
            continue

        if cmd == "prompt":
            if not rest:
                print("[te_daemon] Usage: prompt <text>")
                continue
            state["prompt"] = line[len("prompt ") :].strip()
            print(f"[te_daemon] Prompt set to: {state['prompt']}")
            continue

        if cmd == "neg":
            if not rest:
                print("[te_daemon] Usage: neg <text>")
                continue
            state["defaults"]["negative_prompt"] = line[len("neg ") :].strip()
            print("[te_daemon] Negative prompt set.")
            continue

        if cmd == "set":
            if len(rest) < 2:
                print("[te_daemon] Usage: set <key> <value>")
                continue
            key = rest[0]
            value_str = " ".join(rest[1:])

            # Special-case: magcache toggles and rebuilds the pipeline
            if key == "magcache":
                new_val = parse_bool(value_str)
                if new_val is None:
                    print(
                        f"[te_daemon] Could not interpret '{value_str}' as bool for key 'magcache'."
                    )
                    continue
                if new_val == state["magcache"]:
                    print(f"[te_daemon] magcache already set to {new_val}")
                    continue

                print(
                    f"[te_daemon] Setting magcache to {new_val} and rebuilding pipeline "
                    "(models will be reloaded once)."
                )
                state["magcache"] = new_val
                args.magcache = new_val
                # Rebuild pipeline in-place
                del pipe
                torch.cuda.empty_cache()
                pipe = build_pipeline(args, device_map)
                continue

            d = state["defaults"]
            if key not in d:
                print(
                    "[te_daemon] Unknown key. Valid generation setting keys: "
                    "width, height, video_duration, steps, "
                    "guidance_weight, scheduler_scale, negative_prompt, "
                    "expand_prompt, seed, progress"
                )
                continue

            old_val = d[key]
            new_val: Any
            if isinstance(old_val, bool):
                new_val = parse_bool(value_str)
                if new_val is None:
                    print(
                        f"[te_daemon] Could not interpret '{value_str}' as bool for key '{key}'."
                    )
                    continue
            elif isinstance(old_val, int):
                try:
                    new_val = int(value_str)
                except ValueError:
                    print(
                        f"[te_daemon] Could not interpret '{value_str}' as int for key '{key}'."
                    )
                    continue
            elif isinstance(old_val, float):
                try:
                    new_val = float(value_str)
                except ValueError:
                    print(
                        f"[te_daemon] Could not interpret '{value_str}' as float for key '{key}'."
                    )
                    continue
            else:
                new_val = value_str

            d[key] = new_val
            print(f"[te_daemon] generation_settings.{key} set to {new_val}")
            continue

        if cmd in {"run", "gen"}:
            # Determine prompt
            if cmd == "gen" and rest:
                prompt = " ".join(rest)
                state["prompt"] = prompt
            else:
                prompt = state["prompt"]

            d = state["defaults"]
            width = int(d["width"])
            height = int(d["height"])
            video_duration = int(d["video_duration"])
            steps = int(d["steps"])
            guidance_weight = float(d["guidance_weight"])
            scheduler_scale = float(d["scheduler_scale"])
            negative_prompt = d["negative_prompt"]
            expand_prompts = bool(d["expand_prompt"])
            seed_setting = d["seed"]
            progress = bool(d["progress"])

            if width <= 0 or height <= 0:
                print("[te_daemon] Width/height must be positive.")
                continue
            if width % 64 != 0 or height % 64 != 0:
                print(
                    f"[te_daemon] Width/height must be multiples of 64. "
                    f"Got ({width}, {height})."
                )
                continue

            # Seed: seed < 0 => random per generation
            if isinstance(seed_setting, int) and seed_setting >= 0:
                seed = int(seed_setting) & 0xFFFFFFFF
            else:
                seed = random.randrange(0, 2**31)
            set_seed_all(seed)

            # Confirm before generating
            print("[te_daemon] Generation plan:")
            print(f"  prompt:          {prompt}")
            print(f"  resolution:      {width}x{height}")
            print(f"  duration:        {video_duration}s")
            print(f"  steps:           {steps}")
            print(f"  guidance_weight: {guidance_weight}")
            print(f"  scheduler_scale: {scheduler_scale}")
            print(f"  seed_setting:    {seed_setting}  -> actual seed: {seed}")
            print(f"  expand_prompt:   {expand_prompts}")
            print(f"  progress:        {progress}")
            print(f"  magcache:        {state['magcache']}")
            ans = input("[te_daemon] Proceed? Type 'Y' to generate: ").strip()
            if ans.upper() != "Y":
                print("[te_daemon] Generation cancelled.")
                continue

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = safe_prompt_slug(prompt)
            filename = f"{prefix}{timestamp}_seed{seed}_{slug}.mp4"
            output_path = os.path.join(state["out_dir"], filename)

            print(
                f"[te_daemon] Generating T2V ({width}x{height}, "
                f"{video_duration}s, steps={steps})"
            )
            print(
                f"[te_daemon] Mode label={state['mode_label']}, "
                f"TE enabled={state['te_mode']}, "
                f"TE backend={state['te_backend']}, "
                f"FP8={state['fp8_enabled']}"
            )
            print(f"[te_daemon] Output path: {output_path}")

            t0 = time.perf_counter()
            try:
                # Build kwargs and hook MagCache in a TE-friendly way if supported.
                call_kwargs = dict(
                    time_length=video_duration,
                    width=width,
                    height=height,
                    num_steps=steps,
                    guidance_weight=guidance_weight,
                    scheduler_scale=scheduler_scale,
                    negative_caption=negative_prompt,
                    expand_prompts=expand_prompts,
                    save_path=output_path,
                    seed=seed,
                    progress=progress,
                )

                # If the pipeline is the TE T2V pipeline, it will expose a
                # `use_magcache` kwarg that flows into generate_sample_te.
                try:
                    sig = inspect.signature(pipe.__call__)
                    if "use_magcache" in sig.parameters:
                        call_kwargs["use_magcache"] = bool(state["magcache"])
                except Exception:
                    # If introspection fails, silently skip; baseline pipeline
                    # simply won't see a use_magcache flag.
                    pass

                _ = pipe(
                    prompt,
                    **call_kwargs,
                )
            except Exception as e:
                t1 = time.perf_counter()
                print(
                    f"[te_daemon] ERROR during generation "
                    f"({t1 - t0:.2f}s): {e}"
                )
                continue
            t1 = time.perf_counter()
            time_total = t1 - t0

            # JSONL record (keep shape broadly compatible with te_test.py)
            record = {
                "timestamp": timestamp,
                "config_path": os.path.abspath(args.config),
                "video_path": output_path,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": int(seed),
                "width": int(width),
                "height": int(height),
                "time_length": int(video_duration),
                "scheduler_scale": float(scheduler_scale),
                "num_steps": int(steps),
                "guidance_weight": float(guidance_weight),
                "mode": state["mode_label"],                     # "baseline_bf16", "te_bf16", "te_fp8"
                "te_mode": bool(state["te_mode"]),               # True/False: TE path enabled
                "te_backend": None if args.no_te else args.te_backend,
                "fp8_enabled": False if args.no_te else (not args.disable_fp8),
                "expand_prompts": bool(expand_prompts),
                "magcache": bool(state["magcache"]),
                "time_total": float(time_total),                 # outer wall-clock
            }
            append_jsonl(record, state["jsonl_path"])

            print(f"[te_daemon] Done in {time_total:.2f}s.")
            continue

        print(f"[te_daemon] Unknown command: {cmd}. Type 'help' for usage.")

    print("[te_daemon] Exiting.")


if __name__ == "__main__":
    main()

