# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""K/V projections (M=B*kv_len, N=K=80) feeding learned-query attention.

Baseline: stock Inductor (two skinny mm kernels materialize bf16 K and V, stock flex
template reads them). Candidate: the same model with torchTLX on, the slot for an sm90
TLX flex template that projects K/V tiles in its prologue from the 80x80 weights in SMEM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "sm90_03_kv_proj_attention"
CANDIDATE_NAME = "tlx"
INTERLEAVE_VARIANTS = True
_COMMON_CONFIG: dict[str, object] = {
    "force_disable_caches": True,
    "enable_caching_generated_triton_templates": False,
}
BASELINE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": None}
# "force" has no eligible sm90 TLX template here (NoValidChoicesError); "allow" keeps stock fallbacks.
CANDIDATE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": "allow"}
ATOL = 3.0e-2
RTOL = 3.0e-2
REL_L2_TOL = 1.0e-2
# x, wk, wv
GRAD_INPUTS = (1, 2, 3)

B = 1365
Q_LEN = 64
KV_LEN = 1536
HEAD_DIM = 80
SHAPES = tuple({"b": B, "q_len": q, "kv_len": kv, "head_dim": d}
               for q, kv, d in ((64, 1536, 80), (64, 1920, 80), (200, 200, 256), (64, 200, 256)))


def add_arguments(parser) -> None:
    parser.add_argument("--b", type=int)
    parser.add_argument("--q-len", dest="q_len", type=int)
    parser.add_argument("--kv-len", dest="kv_len", type=int)
    parser.add_argument("--head-dim", dest="head_dim", type=int)


def configure(args) -> None:
    global B, Q_LEN, KV_LEN, HEAD_DIM
    B = args.b or B
    Q_LEN = args.q_len or Q_LEN
    KV_LEN = args.kv_len or KV_LEN
    HEAD_DIM = args.head_dim or HEAD_DIM


def problem() -> str:
    return f"B={B} q_len={Q_LEN} kv_len={KV_LEN} D={HEAD_DIM} dtype=bf16"


def _project(q, x, wk, wv):
    k = (x @ wk).unsqueeze(1)
    v = (x @ wv).unsqueeze(1)
    return q.expand(x.shape[0], -1, -1).unsqueeze(1), k, v


def model(q, x, wk, wv) -> torch.Tensor:
    from torch.nn.attention.flex_attention import flex_attention

    q, k, v = _project(q, x, wk, wv)
    return flex_attention(q.contiguous(), k, v)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    scale = HEAD_DIM**-0.5
    return (
        torch.randn((Q_LEN, HEAD_DIM), device="cuda", dtype=torch.bfloat16),
        torch.randn((B, KV_LEN, HEAD_DIM), device="cuda", dtype=torch.bfloat16),
        torch.randn((HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * scale,
        torch.randn((HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * scale,
    )
