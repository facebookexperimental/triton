# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Learned-query cross-attention whose shared K=V comes from a per-row RMSNorm chain.

K = V = RMSNorm(x + g * w * RMSNorm(c) + pos), Q = Linear(RMSNorm(seed)) broadcast
over the batch. Baseline: SDPA with torchTLX off. Candidate: flex attention with
torchTLX allowed to fuse the RMSNorm producer into the attention template.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from torchtlx_benchmark.run_torchtlx_fusions import attention

if TYPE_CHECKING:
    import torch

NAME = "sm90_01_rmsnorm_kv_attention"
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
# x, c, w, g, pos
GRAD_INPUTS = (3, 4, 5, 6, 7)

B = 1365
Q_LEN = 32
KV_LEN = 1920
HEAD_DIM = 80
EPS = 1.0e-5
ROWS = ((32, 1920, 80), (32, 1536, 80), (32, 200, 96), (32, 300, 96), (16, 100, 80), (16, 200, 256))
SHAPES = tuple({"b": B, "q_len": q, "kv_len": kv, "head_dim": d} for q, kv, d in ROWS)


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


def _project(seed, wq, bq, x, c, w, g, pos):
    rms_norm = torch.nn.functional.rms_norm
    d = (HEAD_DIM, )
    q = torch.nn.functional.linear(rms_norm(seed, d, eps=EPS).bfloat16(), wq, bq)
    q = q.expand(x.shape[0], -1, -1).unsqueeze(1)
    y = x + g * rms_norm(c.float(), d, w, eps=EPS)
    k = rms_norm(y + pos, d, eps=EPS).bfloat16().unsqueeze(1)
    return q, k


def model(*inputs, use_flex=False) -> torch.Tensor:
    q, k = _project(*inputs)
    return attention(q.contiguous(), k, k, use_flex=use_flex)


def comparison_models():
    return model, partial(model, use_flex=True)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    d = HEAD_DIM
    return (
        torch.randn((Q_LEN, d), device="cuda"),
        torch.randn((d, d), device="cuda", dtype=torch.bfloat16) * d**-0.5,
        torch.randn((d, ), device="cuda", dtype=torch.bfloat16),
        torch.randn((B, KV_LEN, d), device="cuda"),
        torch.randn((B, KV_LEN, d), device="cuda", dtype=torch.bfloat16),
        torch.randn((d, ), device="cuda"),
        torch.randn((1, ), device="cuda"),
        torch.randn((KV_LEN, d), device="cuda"),
    )
