import argparse
import logging
import os
import random
import time
import warnings
from datetime import datetime

import torch

from kandinsky.utils import set_hf_token
from kandinsky import get_T2V_pipeline
from kandinsky.te_t2v_pipeline import Kandinsky5T2VTEPipeline
from kandinsky.models.te_dit import get_dit as get_te_dit


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video using Kandinsky 5 with TE-FP8 DiT"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/te_fp8_k5_pro_t2v_10s_sft_hd.yaml",
        help="Config file (TE-FP8 variant of Pro T2V 10s SFT HD)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A small spinning cube.",
        help="Positive prompt to generate video",
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
        default=128,
        help="Output width in pixels (must be a multiple of 128 for this model)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Output height in pixels (must be a multiple of 128 for this model)",
    )
    parser.add_argument(
        "--video_duration",
        type=int,
        default=1,
        help="Duration of the video in seconds (0 for a single image)",
    )
    parser.add_argument(
        "--expand_prompt",
        action="store_true",
        default=False,
        help="Enable prompt expansion (default: disabled)",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Number of sampling steps (default: use config value)",
    )
    parser.add_argument(
        "--guidance_weight",
        type=float,
        default=None,
        help="Guidance weight (default: use config value)",
    )
    parser.add_argument(
        "--scheduler_scale",
        type=float,
        default=5.0,
        help="Scheduler scale (default: 5.0)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Explicit output filename (otherwise auto-named under k5pro_output/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed (masked to 31 bits). If omitted, a random seed is used inside the pipeline.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        default=False,
        help="Offload models to CPU between stages to save memory",
    )
    parser.add_argument(
        "--magcache",
        action="store_true",
        default=False,
        help="Use MagCache (for 50-step models only)",
    )
    parser.add_argument(
        "--qwen_quantization",
        action="store_true",
        default=False,
        help="Use quantized Qwen2.5-VL (4-bit) for text encoding",
    )
    parser.add_argument(
        "--attention_engine",
        type=str,
        default="auto",
        choices=["flash_attention_2", "flash_attention_3", "sdpa", "sage", "auto"],
        help="Full attention algorithm for <=5 second generation",
    )
    parser.add_argument(
        "--te_backend",
        type=str,
        default="te-fp8",
        choices=["te-fp8"],
        help="TE backend to use (currently only te-fp8 is implemented)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (for restricted models / VAEs)",
    )
    return parser.parse_args()


def sanitize_seed(seed: int | None) -> int | None:
    """
    Make sure the seed fits comfortably within 32-bit limits.
    We mask to 31 bits for extra safety.
    """
    if seed is None:
        return None
    return seed & 0x7FFFFFFF


def make_default_output_path(prompt: str, time_length: int) -> str:
    os.makedirs("k5pro_output", exist_ok=True)
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")

    # Safe-ish prompt slug
    base = prompt.strip().replace(" ", "_")
    base = "".join(ch for ch in base if ch.isalnum() or ch in ("_", "-"))
    if len(base) > 48:
        base = base[:48]
    if not base:
        base = "sample"

    prefix = "te_fp8"
    ext = ".png" if time_length == 0 else ".mp4"
    filename = f"{prefix}_{ts}_{base}{ext}"
    return os.path.join("k5pro_output", filename)


def main():
    disable_warnings()
    args = parse_args()

    if args.hf_token:
        set_hf_token(args.hf_token)

    # Sanitize seed if provided
    user_seed = sanitize_seed(args.seed)

    # Device map – simple single-GPU default
    device_map = {"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"}

    # Build the standard T2V pipeline first to reuse all loading logic (config, text embedder, VAE, etc.)
    base_pipe = get_T2V_pipeline(
        device_map=device_map,
        conf_path=args.config,
        offload=args.offload,
        magcache=args.magcache,
        quantized_qwen=args.qwen_quantization,
        attention_engine=args.attention_engine,
    )

    # Build a TE-FP8 DiT from the same config and copy weights from the original DiT
    te_dit = get_te_dit(base_pipe.conf.model.dit_params, backend=args.te_backend)
    te_dit = te_dit.to(device_map["dit"])
    te_dit.load_state_dict(base_pipe.dit.state_dict(), strict=False)

    # Construct the TE pipeline shell around the TE DiT
    pipe = Kandinsky5T2VTEPipeline(
        device_map=device_map,
        dit=te_dit,
        text_embedder=base_pipe.text_embedder,
        vae=base_pipe.vae,
        local_dit_rank=base_pipe.local_dit_rank,
        world_size=base_pipe.world_size,
        conf=base_pipe.conf,
        offload=args.offload,
        device_mesh=getattr(base_pipe, "device_mesh", None),
    )

    # Free the original BF16 DiT + base pipeline object to reclaim memory
    del base_pipe.dit
    del base_pipe
    torch.cuda.empty_cache()

    # Output path
    time_length = int(args.video_duration)
    if args.output_filename is None:
        output_filename = make_default_output_path(args.prompt, time_length)
    else:
        output_filename = args.output_filename
    out_dir = os.path.dirname(output_filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Call TE pipeline
    start_time = time.perf_counter()
    result = pipe(
        args.prompt,
        time_length=time_length,
        width=args.width,
        height=args.height,
        num_steps=args.sample_steps,
        guidance_weight=args.guidance_weight,
        scheduler_scale=args.scheduler_scale,
        negative_caption=args.negative_prompt,
        expand_prompts=args.expand_prompt,
        save_path=output_filename,
        seed=user_seed,
    )
    elapsed = time.perf_counter() - start_time

    print(f"TIME ELAPSED: {elapsed:.2f} seconds")
    print(f"Generated file is saved to {output_filename}")

    # `result` is the tensor of decoded frames; we don't need it here, but
    # returning it makes this script usable as an importable module too.
    return result


if __name__ == "__main__":
    main()
