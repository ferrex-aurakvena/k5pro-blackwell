"""
te_magcache_utils.py

Thin TE-friendly shim around the upstream MagCache utilities.

We keep `magcache_utils.set_magcache_params` untouched and call it in a
way that plays nicely with both baseline DiT and TE-wrapped DiT models.
"""

from __future__ import annotations

from .magcache_utils import set_magcache_params as _set_magcache_params


def enable_magcache_for_dit(
    dit,
    mag_ratios,
    num_steps: int,
    guidance_weight: float,
):
    """
    Apply MagCache to `dit` in a TE-safe way.

    - If `dit` is a TE wrapper (DiffusionTransformer3DTEFP8), we patch
      its underlying `.dit` module so that TE's fp8 context is preserved.
    - If `dit` is a plain DiffusionTransformer3D, we patch it directly.

    Args:
        dit:  DiT module or TE wrapper around it.
        mag_ratios: Sequence of MagCache ratios from the config.
        num_steps:  Number of diffusion steps actually used this run.
        guidance_weight: CFG guidance weight; MagCache needs to know
                         whether CFG is enabled (guidance_weight != 1.0).
    """
    if mag_ratios is None:
        print("[TE-MagCache] mag_ratios is None; skipping MagCache.")
        return dit

    try:
        num_steps = int(num_steps)
    except Exception:
        print("[TE-MagCache] Invalid num_steps for MagCache; skipping.")
        return dit

    if num_steps <= 0:
        print(f"[TE-MagCache] num_steps={num_steps} <= 0; skipping MagCache.")
        return dit

    try:
        gw = float(guidance_weight)
    except Exception:
        gw = 1.0

    # guidance_weight == 1.0 => no CFG => only one forward per step
    no_cfg = abs(gw - 1.0) < 1e-6

    # If this is a TE wrapper, unwrap to get the underlying DiT.
    base_dit = getattr(dit, "dit", dit)

    _set_magcache_params(base_dit, mag_ratios, num_steps, no_cfg=no_cfg)
    return dit

