# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Jagged -> padded pack feeding a per-row RMSNorm (shared K=V) and learned-query attention.

Baseline: SDPA with torchTLX off. Candidate: flex attention with torchTLX allowed
to gather and normalize K in its load path. Padded rows are RMSNorm(pos)
keys, so attention stays dense. FILL is the mean jagged length / kv_len.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from torchtlx_benchmark.run_torchtlx_fusions import attention

if TYPE_CHECKING:
    import torch

NAME = "sm90_02_jagged_pack_attention"
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
# values, pos
GRAD_INPUTS = (2, 4)

B = 1365
Q_LEN = 32
KV_LEN = 1920
HEAD_DIM = 80
FILL = 0.70
EPS = 1.0e-5
SHAPES = tuple({"b": B, "q_len": q, "kv_len": kv, "head_dim": d, "fill": f}
               for q, kv, d, f in ((32, 1920, 80, 0.70), (32, 1536, 80, 0.77), (32, 200, 96, 0.82)))


def add_arguments(parser) -> None:
    parser.add_argument("--b", type=int)
    parser.add_argument("--q-len", dest="q_len", type=int)
    parser.add_argument("--kv-len", dest="kv_len", type=int)
    parser.add_argument("--head-dim", dest="head_dim", type=int)
    parser.add_argument("--fill", type=float, help="mean jagged length / kv_len")


def configure(args) -> None:
    global B, Q_LEN, KV_LEN, HEAD_DIM, FILL
    B = args.b or B
    Q_LEN = args.q_len or Q_LEN
    KV_LEN = args.kv_len or KV_LEN
    HEAD_DIM = args.head_dim or HEAD_DIM
    FILL = args.fill or FILL


def problem() -> str:
    return f"B={B} q_len={Q_LEN} kv_len={KV_LEN} D={HEAD_DIM} fill={FILL} dtype=bf16"


def _gather_pack(values, offsets, max_len):
    pos = torch.arange(max_len, device=values.device)
    lengths = offsets[1:] - offsets[:-1]
    idx = (offsets[:-1, None] + pos).clamp(max=values.shape[0] - 1)
    return values[idx] * (pos < lengths[:, None]).unsqueeze(-1)


def _attention(q_seed, wq, padded, pos, use_flex):
    rms_norm = torch.nn.functional.rms_norm
    q = torch.nn.functional.linear(rms_norm(q_seed, (HEAD_DIM, ), eps=EPS).bfloat16(), wq)
    q = q.expand(padded.shape[0], -1, -1).unsqueeze(1)
    k = rms_norm(padded + pos, (HEAD_DIM, ), eps=EPS).bfloat16().unsqueeze(1)
    return attention(q.contiguous(), k, k, use_flex=use_flex)


def model(q_seed, wq, values, offsets, pos, *, use_flex=False) -> torch.Tensor:
    return _attention(q_seed, wq, _gather_pack(values, offsets, KV_LEN), pos, use_flex)


def comparison_models():
    return model, partial(model, use_flex=True)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    lengths = (KV_LEN * FILL * (0.5 + torch.rand(B))).round().clamp(1, KV_LEN).long()
    offsets = torch.zeros(B + 1, dtype=torch.long)
    offsets[1:] = lengths.cumsum(0)
    return (
        torch.randn((Q_LEN, HEAD_DIM), device="cuda"),
        torch.randn((HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * HEAD_DIM**-0.5,
        torch.randn((int(offsets[-1]), HEAD_DIM), device="cuda"),
        offsets.cuda(),
        torch.randn((KV_LEN, HEAD_DIM), device="cuda"),
    )
