import warnings

import torch
from torch import nn

from .nn import (
    MultiheadSelfAttentionEnc,
    MultiheadSelfAttentionDec,
    MultiheadCrossAttention,
    FeedForward,
)

# Try to import Transformer Engine
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling

    _TE_AVAILABLE = True
except Exception as e:
    te = None
    Format = None
    DelayedScaling = None
    _TE_AVAILABLE = False
    _TE_IMPORT_ERROR = e


def _convert_linear_to_te(linear: nn.Linear, params_dtype=torch.bfloat16):
    """
    Replace a torch.nn.Linear with a TE Linear, copying weights and bias.
    If TE is unavailable or this isn't a plain Linear, return the module unchanged.
    """
    if not _TE_AVAILABLE:
        return linear
    if isinstance(linear, te.Linear):
        return linear
    if not isinstance(linear, nn.Linear):
        return linear

    new = te.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        params_dtype=params_dtype,
    )
    with torch.no_grad():
        new.weight.copy_(linear.weight.to(params_dtype))
        if linear.bias is not None:
            new.bias.copy_(linear.bias.to(params_dtype))
    return new


def _patch_dit_for_te(base_dit: nn.Module, params_dtype=torch.bfloat16) -> nn.Module:
    """
    In-place convert heavy Linear layers in the existing DiT to TE Linear:
      - Q/K/V/out in all self/cross attentions
      - FFN in/out projections
    """
    if not _TE_AVAILABLE:
        return base_dit

    for module in base_dit.modules():
        if isinstance(
            module,
            (MultiheadSelfAttentionEnc, MultiheadSelfAttentionDec, MultiheadCrossAttention),
        ):
            module.to_query = _convert_linear_to_te(module.to_query, params_dtype)
            module.to_key = _convert_linear_to_te(module.to_key, params_dtype)
            module.to_value = _convert_linear_to_te(module.to_value, params_dtype)
            module.out_layer = _convert_linear_to_te(module.out_layer, params_dtype)
        elif isinstance(module, FeedForward):
            module.in_layer = _convert_linear_to_te(module.in_layer, params_dtype)
            module.out_layer = _convert_linear_to_te(module.out_layer, params_dtype)

    return base_dit


class DiffusionTransformer3DTEFP8(nn.Module):
    """
    Thin wrapper around an already-loaded baseline DiffusionTransformer3D that:

      - Optionally converts key Linear layers to Transformer Engine Linears
      - Optionally runs the forward pass under TE FP8 autocast

    We DO NOT re-load weights here: we assume the upstream pipeline has already
    constructed and loaded the baseline DiT and we just wrap/augment it.
    """

    def __init__(
        self,
        base_dit: nn.Module,
        backend: str = "te-fp8",
        enable_fp8: bool = True,
        params_dtype=torch.bfloat16,
    ):
        super().__init__()

        if backend not in ("te-fp8", "te_fp8"):
            raise ValueError(f"Unsupported TE backend: {backend}")

        self.backend = backend
        self.params_dtype = params_dtype
        self.te_available = _TE_AVAILABLE
        self.enable_fp8 = bool(enable_fp8 and _TE_AVAILABLE)

        if not _TE_AVAILABLE:
            warnings.warn(
                "Transformer Engine is not available; "
                "DiffusionTransformer3DTEFP8 will fall back to baseline DiT "
                "without TE acceleration.",
                RuntimeWarning,
            )
            self.dit = base_dit
            self.fp8_recipe = None
            return

        print(
            f"[TE-DiT] Initializing DiffusionTransformer3DTEFP8 "
            f"(backend={backend}, enable_fp8={self.enable_fp8}, "
            f"params_dtype={params_dtype})"
        )

        # Convert heavy Linear modules to TE Linear in-place on the baseline DiT
        base_dit = _patch_dit_for_te(base_dit, params_dtype=params_dtype)

        # Optionally build an FP8 recipe (E4M3, same as our TE FP8 sanity check)
        self.fp8_recipe = None
        if self.enable_fp8 and DelayedScaling is not None and Format is not None:
            try:
                self.fp8_recipe = DelayedScaling(fp8_format=Format.E4M3)
            except Exception:
                # If recipe construction fails, we'll just use default fp8_autocast settings.
                self.fp8_recipe = None

        self.dit = base_dit

    def forward(self, *args, **kwargs):
        if self.enable_fp8 and self.te_available and te is not None:
            ctx_kwargs = {}
            if self.fp8_recipe is not None:
                ctx_kwargs["fp8_recipe"] = self.fp8_recipe
            with te.fp8_autocast(enabled=True, **ctx_kwargs):
                return self.dit(*args, **kwargs)
        else:
            return self.dit(*args, **kwargs)

    def __getattr__(self, name: str):
        """
        Delegate unknown attribute access to the underlying DiT so that
        code like `model.visual_cond` or `model.instruct_type` still works.
        """
        if name in {"backend", "enable_fp8", "params_dtype", "te_available", "dit", "fp8_recipe"}:
            return super().__getattr__(name)
        return getattr(self.dit, name)


def get_dit(
    base_dit: nn.Module,
    backend: str = "te-fp8",
    enable_fp8: bool = True,
    params_dtype=torch.bfloat16,
) -> nn.Module:
    """
    Wrap an existing, weight-loaded DiT in a TE/FP8-aware module.

    NOTE: This is intentionally different from the *original* get_dit(conf)
    in kandinsky/models/dit.py. Here we expect to receive an actual model
    instance whose weights are already loaded by the upstream pipeline.
    """
    if not _TE_AVAILABLE:
        warnings.warn(
            "Transformer Engine could not be imported; returning baseline DiT without TE.",
            RuntimeWarning,
        )
        return base_dit

    return DiffusionTransformer3DTEFP8(
        base_dit=base_dit,
        backend=backend,
        enable_fp8=enable_fp8,
        params_dtype=params_dtype,
    )

