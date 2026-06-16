"""Case 6 — LayerNorm forward, non-WS baseline (correctness + IR-dump source).

Per row x of length N:
    mu   = mean(x)
    var  = mean(x^2) - mu^2             # biased / population variance, single-pass
    rstd = 1 / sqrt(var + eps)
    y    = (x - mu) * rstd * w + b      # w, b are length-N learned params

Single-pass persistent design (matches the production reference
kernel_layernorm_fwd_persistent_tma_v2): one persistent loop over row-tiles,
each iteration loads the whole row in ONE TMA tile, computes mean/var/rstd,
normalizes, and stores. N is a power of two (512) so the row is a single
TMA-friendly tile — no feature-dim masking.

This is the modulo-schedule target: the load feeds tl.sum/tl.sum directly in
REGISTERS (no ttg.local_alloc staging), which is the "register-consumed TMA
load" pattern the emitter must support (distinct from GEMM/FA's
load → SMEM buffer → MMA). It is one loop, so it fits the single-loop emitter.
"""

import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_nows(
    X,  # [M, N] input
    W,  # [N] weight (gamma)
    B,  # [N] bias (beta)
    Y,  # [M, N] output
    M,  # rows
    eps,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(M, BLOCK_M)

    x_desc = tl.make_tensor_descriptor(X, [M, N], [N, 1], [BLOCK_M, N])
    y_desc = tl.make_tensor_descriptor(Y, [M, N], [N, 1], [BLOCK_M, N])

    w = tl.load(W + tl.arange(0, N))
    b = tl.load(B + tl.arange(0, N))

    for tile in range(pid, num_tiles, num_programs):
        row = tile * BLOCK_M
        x = x_desc.load([row, 0]).to(tl.float32)  # [BLOCK_M, N] in registers
        mu = tl.sum(x, axis=1) / N  # [BLOCK_M]
        var = tl.sum(x * x, axis=1) / N - mu * mu  # [BLOCK_M]
        rstd = 1.0 / tl.sqrt(var + eps)  # [BLOCK_M]
        y = (x - mu[:, None]) * rstd[:, None] * w[None, :] + b[None, :]
        y_desc.store([row, 0], y.to(Y.dtype.element_ty))
