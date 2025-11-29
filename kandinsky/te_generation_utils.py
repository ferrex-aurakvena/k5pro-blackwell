# kandinsky/te_generation_utils.py
import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"

import time

import torch
from torch.distributed import all_gather

# Reuse the existing diffusion core from generation_utils
from .generation_utils import generate as _generate_core
from .te_magcache_utils import enable_magcache_for_dit


@torch.no_grad()
def generate_sample_te(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1.0,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    tp_mesh=None,
    use_magcache=False,
):
    """
    TE-aware T2V sampler that:
      - uses the same math as generation_utils.generate_sample
      - optionally enables MagCache on the underlying DiT
      - measures coarse timings:
          * text_encode_seconds
          * sampling_seconds  (DiT loop, including compile/autotune/FP8)
          * vae_decode_seconds
          * total_seconds
    """
    bs, duration, height, width, dim = shape

    # Latent noise
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(
        bs * duration,
        height,
        width,
        dim,
        device=device,
        generator=g,
        dtype=torch.bfloat16,
    )

    type_of_content = "image" if duration == 1 else "video"
    timings = {}

    t_total_start = time.perf_counter()

    # -------------------------
    # 1. TEXT ENCODING
    # -------------------------
    torch.cuda.synchronize()
    t_text_start = time.perf_counter()
    bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
        [caption],
        type_of_content=type_of_content,
    )
    bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
        [negative_caption],
        type_of_content=type_of_content,
    )
    torch.cuda.synchronize()
    timings["text_encode_seconds"] = time.perf_counter() - t_text_start

    if offload:
        text_embedder = text_embedder.to("cpu")

    # Move embeds to DiT device
    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    # Optionally enable MagCache on the underlying DiT
    if use_magcache and hasattr(conf, "magcache") and hasattr(conf.magcache, "mag_ratios"):
        enable_magcache_for_dit(
            dit,
            mag_ratios=conf.magcache.mag_ratios,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
        )

    if offload:
        dit.to(device, non_blocking=True)

    # -------------------------
    # 2. SAMPLING (DiT + TE-FP8)
    # -------------------------
    torch.cuda.synchronize()
    t_sampling_start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        latent_visual = _generate_core(
            dit,
            device,
            img,
            num_steps,
            bs_text_embed,
            bs_null_text_embed,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            scheduler_scale,
            None,
            conf,
            seed=seed,
            progress=progress,
            tp_mesh=tp_mesh,
            attention_mask=attention_mask,
            null_attention_mask=null_attention_mask,
        )
    torch.cuda.synchronize()
    timings["sampling_seconds"] = time.perf_counter() - t_sampling_start

    # Handle tensor-parallel layouts if present
    if tp_mesh:
        tensor_list = [
            torch.zeros_like(latent_visual, device=latent_visual.device)
            for _ in range(tp_mesh["tensor_parallel"].size())
        ]
        all_gather(
            tensor_list,
            latent_visual.contiguous(),
            group=tp_mesh.get_group(mesh_dim="tensor_parallel"),
        )
        latent_visual = torch.cat(tensor_list, dim=1)

    if offload:
        dit = dit.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    # -------------------------
    # 3. VAE DECODE
    # -------------------------
    if offload:
        vae = vae.to(vae_device, non_blocking=True)

    torch.cuda.synchronize()
    t_vae_start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        images = latent_visual.reshape(
            bs,
            -1,
            latent_visual.shape[-3],
            latent_visual.shape[-2],
            latent_visual.shape[-1],
        )
        images = images.to(device=vae_device)
        images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
        images = vae.decode(images).sample
        images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    torch.cuda.synchronize()
    timings["vae_decode_seconds"] = time.perf_counter() - t_vae_start

    if offload:
        vae = vae.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    timings["total_seconds"] = time.perf_counter() - t_total_start

    return images, timings


