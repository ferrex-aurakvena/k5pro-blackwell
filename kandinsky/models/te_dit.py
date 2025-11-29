# kandinsky/models/te_dit.py
import warnings

import torch
from torch import nn

from .te_nn import (
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
    _TE_IMPORT_ERROR = None
except Exception as e:  # import failure path
    te = None
    Format = None
    DelayedScaling = None
    _TE_AVAILABLE = False
    _TE_IMPORT_ERROR = e


def _convert_linear_to_te(linear: nn.Linear, params_dtype=torch.bfloat16) -> nn.Module:
    """
    Replace a torch.nn.Linear with a Transformer Engine Linear while
    copying over weights and bias.

    Only converts plain nn.Linear; already-converted or non-Linear
    modules are left untouched.
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
    In-place convert *only* the heavy Linear layers in the existing DiT
    to Transformer Engine Linears:

      - Q/K/V/out in all encoder/decoder self-attention blocks
      - Q/K/V/out in all cross-attention blocks
      - FFN in/out projections

    We explicitly do *not* touch TimeEmbeddings, TextEmbeddings,
    VisualEmbeddings, Modulation, or other small Linear layers, which
    avoids FP8 shape assertions for 1xD tensors like time embeddings.
    """
    if not _TE_AVAILABLE:
        return base_dit

    for module in base_dit.modules():
        if isinstance(module, MultiheadSelfAttentionEnc):
            module.to_query = _convert_linear_to_te(
                module.to_query, params_dtype=params_dtype
            )
            module.to_key = _convert_linear_to_te(
                module.to_key, params_dtype=params_dtype
            )
            module.to_value = _convert_linear_to_te(
                module.to_value, params_dtype=params_dtype
            )
            module.out_layer = _convert_linear_to_te(
                module.out_layer, params_dtype=params_dtype
            )
        elif isinstance(module, MultiheadSelfAttentionDec):
            module.to_query = _convert_linear_to_te(
                module.to_query, params_dtype=params_dtype
            )
            module.to_key = _convert_linear_to_te(
                module.to_key, params_dtype=params_dtype
            )
            module.to_value = _convert_linear_to_te(
                module.to_value, params_dtype=params_dtype
            )
            module.out_layer = _convert_linear_to_te(
                module.out_layer, params_dtype=params_dtype
            )
        elif isinstance(module, MultiheadCrossAttention):
            module.to_query = _convert_linear_to_te(
                module.to_query, params_dtype=params_dtype
            )
            module.to_key = _convert_linear_to_te(
                module.to_key, params_dtype=params_dtype
            )
            module.to_value = _convert_linear_to_te(
                module.to_value, params_dtype=params_dtype
            )
            module.out_layer = _convert_linear_to_te(
                module.out_layer, params_dtype=params_dtype
            )
        elif isinstance(module, FeedForward):
            module.in_layer = _convert_linear_to_te(
                module.in_layer, params_dtype=params_dtype
            )
            module.out_layer = _convert_linear_to_te(
                module.out_layer, params_dtype=params_dtype
            )

    return base_dit


class DiffusionTransformer3DTEFP8(nn.Module):
    """
    Thin wrapper around an already-loaded baseline DiffusionTransformer3D:

      - Optionally converts key Linear layers to TE.Linears
      - Optionally runs forward under TE FP8 autocast

    We assume the upstream pipeline has already loaded weights.
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
            raise ValueError(f"Unsupported TE backend: {backend!r}")

        self.backend = backend
        self.params_dtype = params_dtype
        self.te_available = _TE_AVAILABLE
        self.enable_fp8 = bool(enable_fp8 and _TE_AVAILABLE)

        if not _TE_AVAILABLE:
            warnings.warn(
                f"Transformer Engine is not available; DiffusionTransformer3DTEFP8 "
                f"will fall back to the baseline DiT without TE acceleration. "
                f"Import error: {_TE_IMPORT_ERROR}",
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

        # Optional FP8 recipe (E4M3)
        self.fp8_recipe = None
        if self.enable_fp8 and DelayedScaling is not None and Format is not None:
            try:
                self.fp8_recipe = DelayedScaling(fp8_format=Format.E4M3)
            except Exception:
                self.fp8_recipe = None

        self.dit = base_dit

    def forward(self, *args, **kwargs):
        if self.enable_fp8 and self.te_available and te is not None:
            ctx_kwargs = {}
            if self.fp8_recipe is not None:
                ctx_kwargs["fp8_recipe"] = self.fp8_recipe
            # Only TE Linear modules participate in FP8; plain nn.Linear remains bf16.
            with te.fp8_autocast(enabled=True, **ctx_kwargs):
                return self.dit(*args, **kwargs)
        else:
            return self.dit(*args, **kwargs)

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying DiT so that
        code like `model.visual_cond` still works.

        The important part is that we *do not* look in self.__dict__ for
        `dit`, because PyTorch stores submodules in self._modules.
        We let nn.Module manage `self.dit` and just forward unknown
        attributes down to the wrapped DiT.
        """
        # Let nn.Module handle our own attributes & the registered submodule.
        if name in {
            "backend",
            "enable_fp8",
            "params_dtype",
            "te_available",
            "dit",
            "fp8_recipe",
        }:
            return super().__getattr__(name)

        # For everything else, try the underlying DiT.
        return getattr(self.dit, name)


def get_dit(
    base_dit: nn.Module,
    backend: str = "te-fp8",
    enable_fp8: bool = True,
    params_dtype=torch.bfloat16,
) -> nn.Module:
    """
    Wrap an existing, weight-loaded DiT in a TE/FP8-aware module.
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

