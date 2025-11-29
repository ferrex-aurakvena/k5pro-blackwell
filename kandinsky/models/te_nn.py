"""
te_nn.py

TE-facing view of the core transformer building blocks.

Right now this just re-exports the relevant modules from `nn.py` so we
don't touch upstream. If/when we need TE-specific variants, they should
live here without modifying `nn.py`.
"""

from .nn import (
    MultiheadSelfAttentionEnc as MultiheadSelfAttentionEnc,
    MultiheadSelfAttentionDec as MultiheadSelfAttentionDec,
    MultiheadCrossAttention as MultiheadCrossAttention,
    FeedForward as FeedForward,
)

__all__ = [
    "MultiheadSelfAttentionEnc",
    "MultiheadSelfAttentionDec",
    "MultiheadCrossAttention",
    "FeedForward",
]
