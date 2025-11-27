import torch
from torch import nn

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
except Exception as e:
    te = None
    DelayedScaling = None
    Format = None
    _TE_IMPORT_ERROR = e

from .nn import (
    TimeEmbeddings,
    TextEmbeddings,
    VisualEmbeddings,
    RoPE1D,
    RoPE3D,
    Modulation,
    OutLayer,
    apply_scale_shift_norm,
    apply_gate_sum,
    apply_rotary,
)
from .utils import fractal_flatten, fractal_unflatten, nablaT_v2
from .attention import SelfAttentionEngine
from torch.nn.attention.flex_attention import flex_attention

from .dit import DiffusionTransformer3D


class TEFeedForward(nn.Module):
    """
    Simple TE-backed MLP: Linear -> GELU -> Linear, with FP8 support via te.fp8_autocast().
    """

    def __init__(self, dim, ff_dim, params_dtype=torch.bfloat16):
        super().__init__()
        if te is None:
            raise ImportError(
                "TransformerEngine is required for TEFeedForward. "
                "Run test_ngc.py and ensure transformer_engine is importable."
            )
        self.in_layer = te.Linear(dim, ff_dim, bias=False, params_dtype=params_dtype)
        self.activation = nn.GELU()
        self.out_layer = te.Linear(ff_dim, dim, bias=False, params_dtype=params_dtype)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class TEMultiheadSelfAttentionEnc(nn.Module):
    """
    Text encoder self-attention with TE Linear layers for Q/K/V and output.
    Attention itself still uses the existing SelfAttentionEngine (sdpa/flash/sage).
    """

    def __init__(self, num_channels, head_dim, attn_engine=None, params_dtype=torch.bfloat16):
        super().__init__()
        if te is None:
            raise ImportError(
                "TransformerEngine is required for TEMultiheadSelfAttentionEnc."
            )
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)
        self.to_key = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)
        self.to_value = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)

        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)

        self.attn_engine = attn_engine or SelfAttentionEngine("sdpa")

    def get_qkv(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def scaled_dot_product_attention(self, query, key, value, attention_mask=None):
        out = self.attn_engine.get_attention()(
            q=query,
            k=key,
            v=value,
            attn_mask=attention_mask,
        )[0].flatten(-2, -1)
        return out

    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, rope, attention_mask=None):
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)
        out = self.scaled_dot_product_attention(query, key, value, attention_mask)
        out = self.out_l(out)
        return out


class TEMultiheadSelfAttentionDec(nn.Module):
    """
    Visual decoder self-attention with TE Linear Q/K/V/out, including the NABLA sparse path.
    """

    def __init__(self, num_channels, head_dim, attn_engine=None, params_dtype=torch.bfloat16):
        super().__init__()
        if te is None:
            raise ImportError(
                "TransformerEngine is required for TEMultiheadSelfAttentionDec."
            )
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)
        self.to_key = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)
        self.to_value = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)

        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)

        # Preserve whatever backend was configured on the original module (auto/sdpa/flash/sage).
        self.attn_engine = attn_engine or SelfAttentionEngine("auto")
        self._nabla_logged = False

    def get_qkv(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def attention(self, query, key, value):
        out = self.attn_engine.get_attention()(
            q=query.unsqueeze(0),
            k=key.unsqueeze(0),
            v=value.unsqueeze(0),
        )[0].flatten(-2, -1)
        return out

    def nabla(self, query, key, value, sparse_params=None):
        # Identical structure to the upstream implementation, but using TE-backed Q/K/V.
        query = query.unsqueeze(0).transpose(1, 2).contiguous()
        key = key.unsqueeze(0).transpose(1, 2).contiguous()
        value = value.unsqueeze(0).transpose(1, 2).contiguous()
        block_mask = nablaT_v2(
            query,
            key,
            sparse_params["sta_mask"],
            thr=sparse_params["P"],
        )
        out = (
            flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
            )
            .transpose(1, 2)
            .squeeze(0)
            .contiguous()
        )
        out = out.flatten(-2, -1)
        return out

    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, rope, sparse_params=None):
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        if sparse_params is not None:
            if not self._nabla_logged:
                print("[TE-DiT] Using NABLA sparse attention in visual self-attention.")
                self._nabla_logged = True
            out = self.nabla(query, key, value, sparse_params=sparse_params)
        else:
            out = self.attention(query, key, value)

        out = self.out_l(out)
        return out


class TEMultiheadCrossAttention(nn.Module):
    """
    Cross-attention (visual -> text) with TE Linear Q/K/V/out.
    """

    def __init__(self, num_channels, head_dim, attn_engine=None, params_dtype=torch.bfloat16):
        super().__init__()
        if te is None:
            raise ImportError(
                "TransformerEngine is required for TEMultiheadCrossAttention."
            )
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)
        self.to_key = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)
        self.to_value = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)

        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = te.Linear(num_channels, num_channels, bias=True, params_dtype=params_dtype)

        self.attn_engine = attn_engine or SelfAttentionEngine("sdpa")

    def get_qkv(self, x, cond):
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)

        shape, cond_shape = query.shape[:-1], key.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*cond_shape, self.num_heads, -1)
        value = value.reshape(*cond_shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def attention(self, query, key, value, attention_mask=None):
        out = self.attn_engine.get_attention()(
            q=query.unsqueeze(0),
            k=key,
            v=value,
            attn_mask=attention_mask,
        )[0].flatten(-2, -1)
        return out

    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, cond, attention_mask=None):
        query, key, value = self.get_qkv(x, cond)
        query, key = self.norm_qk(query, key)

        out = self.attention(query, key, value, attention_mask)
        out = self.out_l(out)
        return out


class DiffusionTransformer3DTEFP8(DiffusionTransformer3D):
    """
    Drop-in FP8 DiT variant:

    - Reuses all structural logic from DiffusionTransformer3D.
    - Swaps per-block attention + MLP linears for TE-backed versions.
    - Wraps DiT forward in te.fp8_autocast so those TE layers run with FP8 math.
    - Final OutLayer remains plain bf16 PyTorch (not TE), as requested.
    """

    def __init__(
        self,
        *args,
        enable_fp8: bool = True,
        fp8_recipe=None,
        params_dtype=torch.bfloat16,
        **kwargs,
    ):
        if te is None:
            raise ImportError(
                "transformer_engine is not available. "
                "The TE-FP8 DiT requires running inside the NGC PyTorch container "
                "with TransformerEngine installed. Run test_ngc.py to verify."
            )
        super().__init__(*args, **kwargs)

        if fp8_recipe is None:
            if DelayedScaling is None or Format is None:
                raise RuntimeError(
                    "TransformerEngine common.recipe.DelayedScaling / Format not available."
                )
            fp8_recipe = DelayedScaling(fp8_format=Format.E4M3)

        self.fp8_enabled = enable_fp8
        self.fp8_recipe = fp8_recipe
        self.params_dtype = params_dtype

        # Derive head_dim from RoPE3D axes config (matches original constructor).
        head_dim = sum(self.visual_rope_embeddings.axes_dims)

        print(
            f"[TE-DiT] Initializing DiffusionTransformer3DTEFP8 "
            f"(enable_fp8={self.fp8_enabled}, head_dim={head_dim}, params_dtype={self.params_dtype})"
        )

        # --- Text encoder blocks -> TE variants ---
        for block_idx, block in enumerate(self.text_transformer_blocks):
            old_attn = block.self_attention
            old_ff = block.feed_forward
            ff_dim = old_ff.out_layer.in_features

            block.self_attention = TEMultiheadSelfAttentionEnc(
                num_channels=self.model_dim,
                head_dim=head_dim,
                attn_engine=getattr(old_attn, "attn_engine", None),
                params_dtype=self.params_dtype,
            )
            block.feed_forward = TEFeedForward(
                dim=self.model_dim,
                ff_dim=ff_dim,
                params_dtype=self.params_dtype,
            )

        # --- Visual decoder blocks -> TE variants ---
        for block_idx, block in enumerate(self.visual_transformer_blocks):
            old_self = block.self_attention
            old_cross = block.cross_attention
            old_ff = block.feed_forward
            ff_dim = old_ff.out_layer.in_features

            block.self_attention = TEMultiheadSelfAttentionDec(
                num_channels=self.model_dim,
                head_dim=head_dim,
                attn_engine=getattr(old_self, "attn_engine", None),
                params_dtype=self.params_dtype,
            )
            block.cross_attention = TEMultiheadCrossAttention(
                num_channels=self.model_dim,
                head_dim=head_dim,
                attn_engine=getattr(old_cross, "attn_engine", None),
                params_dtype=self.params_dtype,
            )
            block.feed_forward = TEFeedForward(
                dim=self.model_dim,
                ff_dim=ff_dim,
                params_dtype=self.params_dtype,
            )

    # ---- Override compiled helpers with eager versions (less overhead with TE) ----

    def before_text_transformer_blocks(
        self,
        text_embed,
        time,
        pooled_text_embed,
        x,
        text_rope_pos,
    ):
        text_embed = self.text_embeddings(text_embed)
        time_embed = self.time_embeddings(time)
        time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        visual_embed = self.visual_embeddings(x)
        text_rope = self.text_rope_embeddings(text_rope_pos)
        return text_embed, time_embed, text_rope, visual_embed

    def before_visual_transformer_blocks(
        self,
        visual_embed,
        visual_rope_pos,
        scale_factor,
        sparse_params,
    ):
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(
            visual_shape,
            visual_rope_pos,
            scale_factor,
        )
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(
            visual_embed,
            visual_rope,
            visual_shape,
            block_mask=to_fractal,
        )
        return visual_embed, visual_shape, to_fractal, visual_rope

    def forward(self, *args, **kwargs):
        if not self.fp8_enabled or te is None:
            return super().forward(*args, **kwargs)

        # FP8 autocast only affects TE modules; OutLayer (final layer) stays in bf16.
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            return super().forward(*args, **kwargs)


def get_dit(
    conf,
    backend: str = "te-fp8",
    enable_fp8: bool = True,
    params_dtype=torch.bfloat16,
    **kwargs,
):
    """
    Factory for a TE-enhanced DiT.

    Parameters
    ----------
    conf:
        Dict-like configuration (typically `conf.model.dit_params`) that is
        normally passed to `kandinsky.models.dit.get_dit`.
    backend:
        Currently only "te-fp8" is supported. Reserved for future "te-nvfp4".
    enable_fp8:
        If False, the TE modules are kept but run without `fp8_autocast`.
    params_dtype:
        Parameter dtype for TE Linear layers (bf16 by default).
    """
    if backend not in {"te-fp8", "te_fp8"}:
        raise ValueError(f"Unsupported TE backend '{backend}'. Only 'te-fp8' is implemented for now.")

    return DiffusionTransformer3DTEFP8(
        **conf,
        enable_fp8=enable_fp8,
        params_dtype=params_dtype,
        **kwargs,
    )

