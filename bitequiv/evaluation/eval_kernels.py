"""Suite 1 — hand-written evaluation kernels for the bitequiv framework.

This is one of the two kernel suites the framework measures: Suite 1 holds
HAND-WRITTEN kernels; Suite 2 (``realistic_inductor_kernels.py``) holds kernels
generated from torch / TorchInductor. Every kernel here is wrapped in a
:class:`KernelSpec` and collected in :data:`REGISTRY`. A spec knows everything
the driver (``evaluate.py`` / ``evaluate_opt.py``) needs and nothing about the
checker or the fuzzer:

  * which autotuner knobs the kernel accepts (``axes``) and how to expand them
    into a config list at a given effort (``config_space``);
  * how to ``compile`` a config to a ``CompiledKernel`` (the driver reads
    ``ck.asm[<artifact>]`` — ``ptx`` or ``ttgir`` per ``--artifact`` — for the checker);
  * how to ``run`` a config on a random seed and return the output ``bytes``
    (the fuzzer's ground-truth observation; welford concatenates Mean + Var);
  * a small precision size for Stages 1-2 and a perf size + ``benchmark`` hook so
    Stage 3 can time every kernel across its config space.

The kernels are organized into three comment-delimited GROUPS:

  GROUP 1 — PRIORITY (the shapes we want to support): the M1 reductions and the
    M3 wgmma GEMM kernels. These are what the checker + optimization pass must
    handle first.
  GROUP 2 — OPTIMIZATION LAYOUTS (M2): reductions whose operand is a genuinely
    multi-dim tile, so the compiler has a real layout choice and the M2
    reduction-layout pass has something to optimize. (Folded in from the former
    ``eval_kernels_layout.py``.)
  GROUP 3 — DIVERSITY / ROBUSTNESS: a placeholder for a broad, diverse kernel
    set that stresses both the checker and the pass. Next-step job (see banner).

Reduction inputs are float32 with a wide dynamic range and alternating signs
(``_adv_sum_2d`` / ``_adv_dot_2d`` / ``_adv_nd``): that is the regime where the
reduction ORDER actually changes the result bits, which is exactly what the
checker claims to track. Plain unit-scale data would round every order to the
same bits and hide the signal.

GROUP 1 kernels:
  * ``sum_dim1_simple``     — single-tile column sum (no loop).
  * ``sum_dim1_persistent`` — column sum looped over BLOCK_N chunks.
  * ``welford``             — one-pass mean/variance (2 outputs, custom combine).
  * ``sum``                 — single-tile row sum.
  * ``dot``                 — single-tile row dot product (mul-fed reduction).
  * ``cond_reduce``         — column sum behind a DATA-DEPENDENT branch (exercises
                              the checker's control-flow gap in Stage 1).
  * GEMM (M3: Hopper wgmma bitwise-equivalence) -- DRAFT, not GPU-verified:
    ``gemm_f16_free_tiling`` (free-var positive: M/N tiling + num_warps stay ONE
    class), ``gemm_kloop_block_k`` (block_k = fixed bit knob), ``gemm_input_precision``
    (tf32/ieee/tf32x3 -> 3 classes), ``gemm_bias_relu_fp_fusion`` (enable_fp_fusion
    on/off -> 2 classes), ``gemm_fp8_imprecise_acc`` (max_num_imprecise_acc flush),
    ``gemm_tma_store`` (known-limitation: C via TMA, no st.global root).

GROUP 2 kernels (large legal-layout space; the GROUP 2 banner explains the M2
win/control split): ``sum_2d_keepdim``, ``sum_2d_axis0``, ``sum_2d_col``,
``sum_2d_col_big``, ``sum_3d_outer``, ``col_sum_loop``, ``col_dot``,
``col_exp_sum``, ``col_bf16``, ``col_max``, ``sum_2d_deep``, ``sum_3d``,
``sum_3d_mid``, ``sum_4d``, ``sum_multiaxis``, ``layernorm``, ``softmax``,
``rmsnorm``.

Abbreviations are spelled out in full (``num_warps``, ``num_stages``, ``block_n``, ...).
"""

import itertools
from collections import namedtuple
from dataclasses import dataclass

import torch

import triton
import triton.language as tl

DEVICE = "cuda"

# String <-> compiler enum for the reduction-ordering knob (None = kernel default).
_ORDERING = {
    "unordered": tl.ReductionOrdering.UNORDERED,
    "inner_tree": tl.ReductionOrdering.INNER_TREE,
    None: None,
}

# Wide-dynamic-range exponents: order-of-magnitude spread that makes the
# reduction order decide the rounding (and thus the bits).
_SUM_LOGSPACE = (-6, 6)
_DOT_LOGSPACE = (0, 6)

# --------------------------------------------------------------------------- #
# Config space
# --------------------------------------------------------------------------- #
# One config = a point in autotuner-knob space. Unused knobs for a given kernel
# are None (e.g. welford has no reduction_ordering; single-tile kernels have no
# block_n). Full names, no abbreviations.
_AXIS_ORDER = ("reduction_ordering", "num_warps", "num_stages", "enable_fp_fusion", "block_n", "gemm_block_m",
               "gemm_block_n", "gemm_block_k", "input_precision", "max_num_imprecise_acc")
Config = namedtuple("Config", _AXIS_ORDER)

# Per-effort value lists. "light" (~10 configs) is the quick smoke / CI grid;
# "heavy" is the full sweep (the looped kernels reach ~1300 configs via block_n).
_AXIS_VALUES = {
    "light": {
        "reduction_ordering": ("unordered", "inner_tree"),
        "num_warps": (1, 2, 4, 8),
        "num_stages": (1, ),
        "enable_fp_fusion": (True, ),
        "block_n": (4096, ),
        # GEMM axes (M3). gemm_block_* are the wgmma tile; note they are DISTINCT
        # from the reduction `block_n` (a chunk size like 4096) to avoid collision.
        # num_warps is shared, but GEMM specs filter it to {4, 8} (wgmma needs %4==0).
        "gemm_block_m": (64, 128),
        "gemm_block_n": (64, 128),
        "gemm_block_k": (32, ),
        "input_precision": ("tf32", "ieee", "tf32x3"),
        "max_num_imprecise_acc": (32, 128),
    },
    "heavy": {
        "reduction_ordering": ("unordered", "inner_tree"),
        "num_warps": (1, 2, 4, 8, 16, 32),
        "num_stages": (1, 2, 3, 4, 5, 6),
        "enable_fp_fusion": (True, False),
        "block_n": (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
        "gemm_block_m": (64, 128),
        "gemm_block_n": (64, 128, 256),
        "gemm_block_k": (16, 32, 64),
        "input_precision": ("tf32", "ieee", "tf32x3"),
        "max_num_imprecise_acc": (32, 64, 128),
    },
}


def build_configs(axes, config_effort, overrides=None):
    """Cartesian product of the kernel's ``axes`` at the given effort.

    Axes the kernel does not use are pinned to ``None`` so every kernel yields a
    uniform :class:`Config`. ``overrides`` (``{axis: values}``) replaces the
    per-effort value list for specific axes, so a spec's class-splitting knob can
    vary at every effort without perturbing the shared ``_AXIS_VALUES``.
    """
    try:
        values = _AXIS_VALUES[config_effort]
    except KeyError:
        raise ValueError(f"unknown config effort {config_effort!r}; expected one of {sorted(_AXIS_VALUES)}")
    overrides = overrides or {}
    per_axis = [(overrides.get(a, values[a]) if a in axes else (None, )) for a in _AXIS_ORDER]
    return [Config(*combo) for combo in itertools.product(*per_axis)]


def config_label(config):
    """Human-readable one-line label for a config (only the knobs it actually uses)."""
    parts = []
    if config.reduction_ordering is not None:
        parts.append(f"reduction_ordering={config.reduction_ordering}")
    parts.append(f"num_warps={config.num_warps}")
    parts.append(f"num_stages={config.num_stages}")
    if config.enable_fp_fusion is not None:
        parts.append(f"enable_fp_fusion={'on' if config.enable_fp_fusion else 'off'}")
    if config.block_n is not None:
        parts.append(f"block_n={config.block_n}")
    for axis in ("gemm_block_m", "gemm_block_n", "gemm_block_k", "input_precision", "max_num_imprecise_acc"):
        value = getattr(config, axis)
        if value is not None:
            parts.append(f"{axis}={value}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Inputs — seeded, reproducible, float32 with order-sensitive dynamic range.
# --------------------------------------------------------------------------- #
def _signs(cols):
    return torch.where(torch.arange(cols) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0)).unsqueeze(0)


def _adv_sum_2d(rows, cols, seed):
    """Wide dynamic range + alternating signs: reduction ORDER strongly affects the bits."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(rows, cols, generator=g, dtype=torch.float32)
    scale = torch.logspace(_SUM_LOGSPACE[0], _SUM_LOGSPACE[1], cols, dtype=torch.float32).unsqueeze(0)
    return (base * scale * _signs(cols)).to(DEVICE, torch.float32)


def _full_mantissa_2d(rows, cols, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return 1.0 + torch.randint(0, 1 << 23, (rows, cols), generator=g).float() / (1 << 23)  # [1, 2)


def _adv_dot_2d(rows, cols, seed):
    """Large near-cancelling products: residual dominated by product-rounding (fma axis)
    and reduction order (num_warps axis). Returns (a, b)."""
    mant = _full_mantissa_2d(rows, cols, seed)
    mag = torch.logspace(_DOT_LOGSPACE[0], _DOT_LOGSPACE[1], cols, dtype=torch.float32).unsqueeze(0)
    a = (mant * mag * _signs(cols)).to(DEVICE, torch.float32)
    b = (mant * mag).to(DEVICE, torch.float32)
    return a, b


def _randn_2d(rows, cols, seed):
    """Plain unit-scale data — used for perf timing (Stage 3), where only speed matters."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(rows, cols, generator=g, dtype=torch.float32).to(DEVICE)


def _adv_nd(numel, seed):
    """Flat wide-dynamic-range, alternating-sign f32 buffer — reduction ORDER (over any
    axis, once viewed multi-dim) strongly affects the result bits. Used by GROUP 2."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(numel, generator=g, dtype=torch.float32)
    scale = torch.logspace(_SUM_LOGSPACE[0], _SUM_LOGSPACE[1], numel, dtype=torch.float32)
    signs = torch.where(torch.arange(numel) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
    return (base * scale * signs).to(DEVICE, torch.float32)


def _plain_nd(numel, seed):
    """Plain unit-scale flat f32 — GROUP 2 perf timing (Stage 3) and realistic fusions."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(numel, generator=g, dtype=torch.float32).to(DEVICE)


def _to_bytes(t):
    """Exact output bits as a ``bytes`` object (the fuzzer's equality unit)."""
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


# =========================================================================== #
# GROUP 1 — PRIORITY KERNELS TO SUPPORT: M1 reductions + M3 wgmma GEMM.
# The shapes the checker + optimization pass must handle first. The M1 reductions
# come first (kernels, then compile/run/bench), then the M3 GEMM kernels.
# =========================================================================== #
@triton.jit
def sum_dim1_kernel(input_ptr, output_ptr, M, N, input_stride_m, input_stride_n, BLOCK_N: tl.constexpr,
                    ROWS_PER_BLOCK: tl.constexpr, REDUCTION_ORDERING: tl.constexpr = None):
    """Sum a 2D tensor along dim1 (columns). Single [ROWS_PER_BLOCK, BLOCK_N] tile per
    program, no column loop (BLOCK_N must be >= N)."""
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offsets = (row_start + tl.arange(0, ROWS_PER_BLOCK)).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_N).to(tl.int64)
    offsets = row_offsets[:, None] * input_stride_m + col_offsets[None, :] * input_stride_n
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sums = tl.sum(vals.to(tl.float32), axis=1, reduction_ordering=REDUCTION_ORDERING)
    tl.store(output_ptr + row_offsets, sums, mask=row_offsets < M)


@triton.jit
def sum_dim1_kernel_batched(input_ptr, output_ptr, M, N, input_stride_m, input_stride_n, BLOCK_N: tl.constexpr,
                            ROWS_PER_BLOCK: tl.constexpr, REDUCTION_ORDERING: tl.constexpr = None):
    """Batched sum along dim1: loop over BLOCK_N column chunks, tree-reduce within each chunk
    (reduction_ordering), then linearly accumulate across chunks."""
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offsets = (row_start + tl.arange(0, ROWS_PER_BLOCK)).to(tl.int64)
    row_mask = row_offsets < M
    acc = tl.zeros([ROWS_PER_BLOCK], dtype=tl.float32)
    for col_start in range(0, N, BLOCK_N):
        col_offsets = (col_start + tl.arange(0, BLOCK_N)).to(tl.int64)
        offsets = row_offsets[:, None] * input_stride_m + col_offsets[None, :] * input_stride_n
        mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32), axis=1, reduction_ordering=REDUCTION_ORDERING)
    tl.store(output_ptr + row_offsets, acc, mask=row_mask)


@triton.jit
def welford_combine(mean_a, m2_a, count_a, mean_b, m2_b, count_b):
    """Standard parallel/pairwise Welford merge."""
    count = count_a + count_b
    delta = mean_b - mean_a
    new_mean = mean_a + delta * (count_b / tl.maximum(count, 1.0))
    new_m2 = m2_a + m2_b + delta * delta * count_a * count_b / tl.maximum(count, 1.0)
    return new_mean, new_m2, count


@triton.jit
def welford_kernel(X, Mean, Var, N, stride_x, BLOCK_SIZE: tl.constexpr):
    """Welford one-pass reduction over the last dim of a 2D tensor; one row per program."""
    row_idx = tl.program_id(0)
    x_ptr = X + row_idx * stride_x
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    m2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        new_count = count + tl.where(mask, 1.0, 0.0)
        delta = x - mean
        new_mean = mean + tl.where(mask, delta / tl.maximum(new_count, 1.0), 0.0)
        delta2 = x - new_mean
        new_m2 = m2 + tl.where(mask, delta * delta2, 0.0)
        mean, m2, count = new_mean, new_m2, new_count
    final_mean, final_m2, final_count = tl.reduce((mean, m2, count), axis=0, combine_fn=welford_combine)
    tl.store(Mean + row_idx, final_mean)
    tl.store(Var + row_idx, final_m2 / final_count)


@triton.jit
def rowsum_kernel(src, dst, n_cols, stride, BLOCK: tl.constexpr, ORD: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(src + row * stride + offs, mask=mask, other=0.0)
    tl.store(dst + row, tl.sum(x, axis=0, reduction_ordering=ORD))


@triton.jit
def rowdot_kernel(a, b, dst, n_cols, stride, BLOCK: tl.constexpr, ORD: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(a + row * stride + offs, mask=mask, other=0.0)
    y = tl.load(b + row * stride + offs, mask=mask, other=0.0)
    tl.store(dst + row, tl.sum(x * y, axis=0, reduction_ordering=ORD))


@triton.jit
def cond_reduce_kernel(input_ptr, output_ptr, M, N, input_stride_m, input_stride_n, BLOCK_N: tl.constexpr,
                       ROWS_PER_BLOCK: tl.constexpr, REDUCTION_ORDERING: tl.constexpr = None):
    """Column sum behind a DATA-DEPENDENT branch. The branch predicate is a runtime scalar
    (the row's plain sum), so PTX emits a real control-flow branch and the stored result
    depends on which side is taken. The tree reconstruction is branch-blind (last-writer by
    stream position, no CFG), so this kernel sits in the checker's control-flow gap -- it
    exists so Stage 1 can demonstrate a LIMITED / unsupported verdict."""
    pid = tl.program_id(0)
    row_offsets = (pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_N).to(tl.int64)
    offsets = row_offsets[:, None] * input_stride_m + col_offsets[None, :] * input_stride_n
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    plain = tl.sum(vals, axis=1, reduction_ordering=REDUCTION_ORDERING)
    gate = tl.sum(plain)  # reduce the per-row sums to one runtime scalar
    if gate > 0.0:
        result = tl.sum(vals * vals, axis=1, reduction_ordering=REDUCTION_ORDERING)
    else:
        result = plain
    tl.store(output_ptr + row_offsets, result, mask=row_offsets < M)


# --------------------------------------------------------------------------- #
# Per-kernel compile / run (and perf for the looped kernel)
# --------------------------------------------------------------------------- #
def _simple_compile(config, size, maxnreg=None):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return sum_dim1_kernel.warmup(src, out, rows, cols, src.stride(0), src.stride(1), grid=(rows, ), BLOCK_N=cols,
                                  ROWS_PER_BLOCK=1, REDUCTION_ORDERING=_ORDERING[config.reduction_ordering],
                                  num_warps=config.num_warps, num_stages=config.num_stages,
                                  enable_fp_fusion=config.enable_fp_fusion, maxnreg=maxnreg)


def _simple_run(config, ck, seed, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, seed)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](src, out, rows, cols, src.stride(0), src.stride(1))
    torch.cuda.synchronize()
    return _to_bytes(out)


def _persist_compile(config, size, maxnreg=None):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return sum_dim1_kernel_batched.warmup(src, out, rows, cols, src.stride(0), src.stride(1), grid=(rows, ),
                                          BLOCK_N=config.block_n, ROWS_PER_BLOCK=1,
                                          REDUCTION_ORDERING=_ORDERING[config.reduction_ordering],
                                          num_warps=config.num_warps, num_stages=config.num_stages,
                                          enable_fp_fusion=config.enable_fp_fusion, maxnreg=maxnreg)


def _persist_run(config, ck, seed, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, seed)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](src, out, rows, cols, src.stride(0), src.stride(1))
    torch.cuda.synchronize()
    return _to_bytes(out)


def _welford_compile(config, size, maxnreg=None):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    mean = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    var = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return welford_kernel.warmup(src, mean, var, cols, src.stride(0), grid=(rows, ), BLOCK_SIZE=config.block_n,
                                 num_warps=config.num_warps, num_stages=config.num_stages,
                                 enable_fp_fusion=config.enable_fp_fusion, maxnreg=maxnreg)


def _welford_run(config, ck, seed, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, seed)
    mean = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    var = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](src, mean, var, cols, src.stride(0))
    torch.cuda.synchronize()
    return _to_bytes(mean) + _to_bytes(var)  # both outputs must match for bit-equivalence


def _rowsum_compile(config, size, maxnreg=None):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return rowsum_kernel.warmup(src, out, cols, src.stride(0), grid=(rows, ), BLOCK=cols,
                                ORD=_ORDERING[config.reduction_ordering], num_warps=config.num_warps,
                                num_stages=config.num_stages, enable_fp_fusion=config.enable_fp_fusion, maxnreg=maxnreg)


def _rowsum_run(config, ck, seed, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, seed)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](src, out, cols, src.stride(0))
    torch.cuda.synchronize()
    return _to_bytes(out)


def _rowdot_compile(config, size, maxnreg=None):
    rows, cols = size
    a, b = _adv_dot_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return rowdot_kernel.warmup(a, b, out, cols, a.stride(0), grid=(rows, ), BLOCK=cols,
                                ORD=_ORDERING[config.reduction_ordering], num_warps=config.num_warps,
                                num_stages=config.num_stages, enable_fp_fusion=config.enable_fp_fusion, maxnreg=maxnreg)


def _rowdot_run(config, ck, seed, size):
    rows, cols = size
    a, b = _adv_dot_2d(rows, cols, seed)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](a, b, out, cols, a.stride(0))
    torch.cuda.synchronize()
    return _to_bytes(out)


def _cond_compile(config, size, maxnreg=None):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return cond_reduce_kernel.warmup(src, out, rows, cols, src.stride(0), src.stride(1), grid=(rows, ), BLOCK_N=cols,
                                     ROWS_PER_BLOCK=1, REDUCTION_ORDERING=_ORDERING[config.reduction_ordering],
                                     num_warps=config.num_warps, num_stages=config.num_stages,
                                     enable_fp_fusion=config.enable_fp_fusion, maxnreg=maxnreg)


def _cond_run(config, ck, seed, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, seed)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](src, out, rows, cols, src.stride(0), src.stride(1))
    torch.cuda.synchronize()
    return _to_bytes(out)


def _bench_ms(thunk):
    """Min-of-medians over 3 do_bench runs (each runs many launches), to suppress noise."""
    from triton.testing import do_bench
    return min(do_bench(thunk, warmup=50, rep=200) for _ in range(3))


def _persist_benchmark(config, size):
    """Compile + time the persistent kernel on plain data; returns (ms, output_bytes, asm).

    Returns the full ``ck.asm`` dict so the driver can hand the checker whichever
    artifact ``--artifact`` selects (``ptx`` or ``ttgir``)."""
    rows, cols = size
    src = _randn_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck = sum_dim1_kernel_batched.warmup(src, out, rows, cols, src.stride(0), src.stride(1), grid=(rows, ),
                                        BLOCK_N=config.block_n, ROWS_PER_BLOCK=1,
                                        REDUCTION_ORDERING=_ORDERING[config.reduction_ordering],
                                        num_warps=config.num_warps, num_stages=config.num_stages,
                                        enable_fp_fusion=config.enable_fp_fusion)

    def thunk():
        ck[(rows, 1, 1)](src, out, rows, cols, src.stride(0), src.stride(1))

    thunk()
    torch.cuda.synchronize()
    return _bench_ms(thunk), _to_bytes(out), ck.asm


# Stage-3 benchmarks for the remaining kernels. Lean: build inputs/outputs ONCE (seed 0)
# and time only the launch — input generation must stay out of the timed region. Each
# returns (ms, output_bytes, asm_dict), the same shape as _persist_benchmark.
def _bench(ck, thunk, out_bytes):
    thunk()
    torch.cuda.synchronize()
    return _bench_ms(thunk), out_bytes, ck.asm


def _simple_benchmark(config, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck = _simple_compile(config, size)
    return _bench(ck, lambda: ck[(rows, 1, 1)](src, out, rows, cols, src.stride(0), src.stride(1)), _to_bytes(out))


def _welford_benchmark(config, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    mean = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    var = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck = _welford_compile(config, size)
    return _bench(ck, lambda: ck[(rows, 1, 1)](src, mean, var, cols, src.stride(0)), _to_bytes(mean) + _to_bytes(var))


def _rowsum_benchmark(config, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck = _rowsum_compile(config, size)
    return _bench(ck, lambda: ck[(rows, 1, 1)](src, out, cols, src.stride(0)), _to_bytes(out))


def _rowdot_benchmark(config, size):
    rows, cols = size
    a, b = _adv_dot_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck = _rowdot_compile(config, size)
    return _bench(ck, lambda: ck[(rows, 1, 1)](a, b, out, cols, a.stride(0)), _to_bytes(out))


def _cond_benchmark(config, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck = _cond_compile(config, size)
    return _bench(ck, lambda: ck[(rows, 1, 1)](src, out, rows, cols, src.stride(0), src.stride(1)), _to_bytes(out))


# --------------------------------------------------------------------------- #
# GEMM kernels (M3: Hopper wgmma bitwise-equivalence)
# --------------------------------------------------------------------------- #
# DRAFT -- these kernels are NOT yet GPU-verified. Every "lowers to wgmma" and
# "order-sensitive" claim below rests on a static code audit, not a live H100
# compile+run. Each needs a smoke test (dump PTX -> confirm wgmma.mma_async with
# no MMAv2/FMA fallback; run the fuzzer -> confirm the expected class split or
# collapse) before it can be trusted. See ~/bitwise-equiv/M3_PLAN.md.
#
# The M3 bit model: for C[m, n] = sum_k A[m, k] * B[k, n] the FIXED (bit-deciding)
# knobs are the K-loop order, the wgmma instruction K-shape (block_k / dtype), the
# accumulate mode (max_num_imprecise_acc), the input precision, and
# enable_fp_fusion. M/N tiling (gemm_block_m / gemm_block_n) and warp arrangement
# are FREE -- they must NOT change a given C[m, n]'s bits. The specs below are a
# free-variable positive anchor, one negative per fixed knob, and a
# known-limitation (TMA-store) example.
#
# wgmma v3 needs num_warps % 4 == 0 (1/2 fall back to MMAv2/FMA), so every GEMM
# spec filters num_warps to {4, 8} via _gemm_num_warps_filter. All precision sizes
# are exact multiples of every block value, so no K-tail masking is needed (a
# masked tail would perturb the accumulation and muddy the bitwise signal).
_TORCH_OUT_DTYPE = {tl.float16: torch.float16, tl.float32: torch.float32}

# Value with nontrivial low mantissa bits so `acc * alpha` fuses (fma.rn) vs
# not (mul.rn + add.rn) observably differ under enable_fp_fusion.
_GEMM_ALPHA = 1.0009765625

# Per-dtype operand dynamic range; fp16/fp8 use a narrower range to stay in-range
# / representable after the cast. Keyed by torch dtype; fp8 falls back to default.
_GEMM_LOGSPACE = {
    torch.float32: (0.0, 6.0),
    torch.bfloat16: (0.0, 4.0),
    torch.float16: (-2.0, 2.0),
}


def _gemm_operands(M, N, K, seed, dtype):
    """A[M, K], B[K, N] with wide magnitude + alternating sign ALONG K (the
    contraction axis), so the sum over K catastrophically cancels and its ORDER
    decides the result bits."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    logspace = _GEMM_LOGSPACE.get(dtype, (-1.0, 1.0))  # fallback = fp8 e4m3: narrow, stays inside the ~448 range
    base_a = torch.randn(M, K, generator=g, dtype=torch.float32)
    mant_b = 1.0 + torch.randint(0, 1 << 23, (K, N), generator=g).float() / (1 << 23)
    k_scale = torch.logspace(logspace[0], logspace[1], K, dtype=torch.float32)
    k_sign = torch.where(torch.arange(K) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
    a = base_a * (k_scale * k_sign).unsqueeze(0)
    b = mant_b * k_scale.unsqueeze(1)
    return a.to(DEVICE, dtype), b.to(DEVICE, dtype)


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT_DTYPE: tl.constexpr,
                INPUT_PRECISION: tl.constexpr = None, MAX_NUM_IMPRECISE_ACC: tl.constexpr = None):
    """Tiled GEMM: one [BLOCK_M, BLOCK_N] output tile per program, K looped in
    BLOCK_K chunks accumulating into one f32 accumulator (the canonical K order).
    C is written with a plain tl.store (st.global root the PTX checker can see).
    Shared by the free-tiling / K-loop / input-precision / fp8 specs."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, input_precision=INPUT_PRECISION,
                     max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(OUT_DTYPE))


@triton.jit
def gemm_bias_relu_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, alpha, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                          stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          OUT_DTYPE: tl.constexpr):
    """GEMM with a fused `relu(acc * alpha + bias)` epilogue. The scale+bias
    affine is fma.rn (enable_fp_fusion on) vs mul.rn + add.rn (off) -- the
    enable_fp_fusion negative lives in this modeled f32 epilogue DAG, since the
    fusion knob cannot change the wgmma itself."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    bias = tl.load(bias_ptr + offs_n)
    acc = acc * alpha + bias[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(OUT_DTYPE))


@triton.jit
def gemm_tma_store_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                          stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          OUT_DTYPE: tl.constexpr):
    """Known-limitation: identical to gemm_kernel except C is written through a
    TMA descriptor (cp.async.bulk.tensor via shared memory), so NO st.global is
    emitted -- builder.roots() finds no root -> empty descriptor -> Stage-1
    UNSUPPORTED. The cond_reduce analog for the epilogue-store gap. NOTE: the TMA
    descriptor API + host allocator wiring is UNVERIFIED on the target Triton."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
                                       block_shape=[BLOCK_M, BLOCK_N])
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(OUT_DTYPE))


# --------------------------------------------------------------------------- #
# GEMM compile / run
# --------------------------------------------------------------------------- #
def _gemm_num_warps_filter(config):
    """wgmma v3 needs num_warps % 4 == 0; 1/2 fall back to MMAv2/FMA and would
    leave the wgmma path (and merge into the wrong equivalence class)."""
    return config.num_warps in (4, 8)


def _gemm_blocks(config, default_bk):
    """Resolve the tile from the config, defaulting axes the spec does not sweep."""
    bm = config.gemm_block_m if config.gemm_block_m is not None else 128
    bn = config.gemm_block_n if config.gemm_block_n is not None else 128
    bk = config.gemm_block_k if config.gemm_block_k is not None else default_bk
    return bm, bn, bk


def _gemm_launch_kwargs(config):
    """Shared launch knobs, with GEMM-safe defaults when an axis is unused."""
    return {
        "num_warps": config.num_warps,
        "num_stages": config.num_stages if config.num_stages is not None else 1,
        "enable_fp_fusion": config.enable_fp_fusion if config.enable_fp_fusion is not None else True,
    }


def _gemm_compile_generic(config, size, maxnreg, in_dtype, out_dtype, default_bk, input_precision,
                          max_num_imprecise_acc):
    M, N, K = size
    bm, bn, bk = _gemm_blocks(config, default_bk)
    a, b = _gemm_operands(M, N, K, 0, in_dtype)
    c = torch.empty((M, N), device=DEVICE, dtype=_TORCH_OUT_DTYPE[out_dtype])
    return gemm_kernel.warmup(a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                              c.stride(1), grid=(M // bm, N // bn), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                              OUT_DTYPE=out_dtype, INPUT_PRECISION=input_precision,
                              MAX_NUM_IMPRECISE_ACC=max_num_imprecise_acc, maxnreg=maxnreg, **_gemm_launch_kwargs(config))


def _gemm_run_generic(config, ck, seed, size, in_dtype, out_dtype, default_bk):
    M, N, K = size
    bm, bn, _bk = _gemm_blocks(config, default_bk)
    a, b = _gemm_operands(M, N, K, seed, in_dtype)
    c = torch.empty((M, N), device=DEVICE, dtype=_TORCH_OUT_DTYPE[out_dtype])
    ck[(M // bm, N // bn, 1)](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                              c.stride(1))
    torch.cuda.synchronize()
    return _to_bytes(c)


def _gemm_free_tiling_compile(config, size, maxnreg=None):
    return _gemm_compile_generic(config, size, maxnreg, torch.float16, tl.float16, 32, None, None)


def _gemm_free_tiling_run(config, ck, seed, size):
    return _gemm_run_generic(config, ck, seed, size, torch.float16, tl.float16, 32)


def _gemm_kloop_compile(config, size, maxnreg=None):
    return _gemm_compile_generic(config, size, maxnreg, torch.bfloat16, tl.float32, 32, None, None)


def _gemm_kloop_run(config, ck, seed, size):
    return _gemm_run_generic(config, ck, seed, size, torch.bfloat16, tl.float32, 32)


def _gemm_prec_compile(config, size, maxnreg=None):
    return _gemm_compile_generic(config, size, maxnreg, torch.float32, tl.float32, 32, config.input_precision, None)


def _gemm_prec_run(config, ck, seed, size):
    return _gemm_run_generic(config, ck, seed, size, torch.float32, tl.float32, 32)


def _gemm_fp8_compile(config, size, maxnreg=None):
    return _gemm_compile_generic(config, size, maxnreg, torch.float8_e4m3fn, tl.float32, 256, None,
                                 config.max_num_imprecise_acc)


def _gemm_fp8_run(config, ck, seed, size):
    return _gemm_run_generic(config, ck, seed, size, torch.float8_e4m3fn, tl.float32, 256)


def _gemm_bias(N, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    return (torch.randn(N, generator=g, dtype=torch.float32) * 1e3).to(DEVICE, torch.float32)


def _gemm_bias_relu_compile(config, size, maxnreg=None):
    M, N, K = size
    bm, bn, bk = _gemm_blocks(config, 32)
    a, b = _gemm_operands(M, N, K, 0, torch.bfloat16)
    bias = _gemm_bias(N, 0)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
    return gemm_bias_relu_kernel.warmup(a, b, c, bias, _GEMM_ALPHA, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                        b.stride(1), c.stride(0), c.stride(1), grid=(M // bm, N // bn), BLOCK_M=bm,
                                        BLOCK_N=bn, BLOCK_K=bk, OUT_DTYPE=tl.float32, maxnreg=maxnreg,
                                        **_gemm_launch_kwargs(config))


def _gemm_bias_relu_run(config, ck, seed, size):
    M, N, K = size
    bm, bn, _bk = _gemm_blocks(config, 32)
    a, b = _gemm_operands(M, N, K, seed, torch.bfloat16)
    bias = _gemm_bias(N, seed)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
    ck[(M // bm, N // bn, 1)](a, b, c, bias, _GEMM_ALPHA, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1))
    torch.cuda.synchronize()
    return _to_bytes(c)


def _gemm_tma_store_compile(config, size, maxnreg=None):
    M, N, K = size
    bm, bn, bk = _gemm_blocks(config, 32)
    a, b = _gemm_operands(M, N, K, 0, torch.float16)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    return gemm_tma_store_kernel.warmup(a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                        c.stride(0), c.stride(1), grid=(M // bm, N // bn), BLOCK_M=bm, BLOCK_N=bn,
                                        BLOCK_K=bk, OUT_DTYPE=tl.float16, maxnreg=maxnreg, **_gemm_launch_kwargs(config))


def _gemm_tma_store_run(config, ck, seed, size):
    # The TMA descriptor store needs a global scratch allocator; install one
    # before launching. (Stage 1's UNSUPPORTED verdict only needs the compiled
    # PTX, but Stages 2/4 also RUN the kernel, so it must be launchable.)
    triton.set_allocator(lambda nbytes, alignment, stream: torch.empty(nbytes, device=DEVICE, dtype=torch.int8))
    M, N, K = size
    bm, bn, _bk = _gemm_blocks(config, 32)
    a, b = _gemm_operands(M, N, K, seed, torch.float16)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    ck[(M // bm, N // bn, 1)](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                              c.stride(1))
    torch.cuda.synchronize()
    return _to_bytes(c)


# --------------------------------------------------------------------------- #
# KernelSpec + registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KernelSpec:
    """Everything the framework needs to evaluate one kernel (see module docstring)."""

    name: str
    description: str
    output_arity: int  # number of output tensors (welford = 2)
    axes: tuple  # autotuner knobs this kernel accepts (subset of _AXIS_ORDER)
    precision_size: tuple  # (ROWS, COLS) for Stages 1-2
    perf_size: tuple  # (ROWS, COLS) for Stage 3
    known_limitation: str  # non-empty => Stage 1 reports LIMITED with this note
    compile_fn: object  # (config, size) -> CompiledKernel
    run_fn: object  # (config, ck, seed, size) -> bytes
    perf_fn: object = None  # (config, size) -> (ms, bytes, asm_dict); None => skipped in Stage 3
    config_filter: object = None  # optional (config) -> bool; drop configs that fail it (e.g. GEMM num_warps in {4, 8})
    axis_values: object = None  # optional {axis: values} override so a splitting knob varies at every effort

    @property
    def supports_perf(self):
        return self.perf_fn is not None

    def config_space(self, config_effort):
        configs = build_configs(self.axes, config_effort, self.axis_values)
        if self.config_filter is not None:
            configs = [c for c in configs if self.config_filter(c)]
        return configs

    def compile(self, config, size, maxnreg=None):
        return self.compile_fn(config, size, maxnreg=maxnreg)

    def run(self, config, ck, seed, size):
        return self.run_fn(config, ck, seed, size)

    def benchmark(self, config, size):
        return self.perf_fn(config, size)


_REDUCTION_AXES = ("reduction_ordering", "num_warps", "num_stages", "enable_fp_fusion")

REGISTRY = {
    "sum_dim1_simple":
    KernelSpec(
        name="sum_dim1_simple",
        description="single-tile column sum (no loop); BLOCK_N covers the whole row",
        output_arity=1,
        axes=_REDUCTION_AXES,
        precision_size=(128, 8192),
        perf_size=(128, 8192),
        known_limitation="",
        compile_fn=_simple_compile,
        run_fn=_simple_run,
        perf_fn=_simple_benchmark,
    ),
    "sum_dim1_persistent":
    KernelSpec(
        name="sum_dim1_persistent",
        description="column sum looped over BLOCK_N chunks (tree-reduce per chunk, accumulate across)",
        output_arity=1,
        axes=_REDUCTION_AXES + ("block_n", ),
        precision_size=(128, 16384),
        perf_size=(128, 262144),
        known_limitation="",
        compile_fn=_persist_compile,
        run_fn=_persist_run,
        perf_fn=_persist_benchmark,
    ),
    "welford":
    KernelSpec(
        name="welford",
        description="one-pass mean/variance (2 outputs, custom Welford combine; no reduction_ordering)",
        output_arity=2,
        axes=("num_warps", "num_stages", "enable_fp_fusion", "block_n"),
        precision_size=(128, 16384),
        perf_size=(128, 16384),
        known_limitation="",
        compile_fn=_welford_compile,
        run_fn=_welford_run,
        perf_fn=_welford_benchmark,
    ),
    "sum":
    KernelSpec(
        name="sum",
        description="single-tile row sum (pure add reduction)",
        output_arity=1,
        axes=_REDUCTION_AXES,
        precision_size=(128, 8192),
        perf_size=(128, 8192),
        known_limitation="",
        compile_fn=_rowsum_compile,
        run_fn=_rowsum_run,
        perf_fn=_rowsum_benchmark,
    ),
    "dot":
    KernelSpec(
        name="dot",
        description="single-tile row dot product (mul-fed reduction; exercises FMA contraction)",
        output_arity=1,
        axes=_REDUCTION_AXES,
        precision_size=(128, 8192),
        perf_size=(128, 8192),
        known_limitation="",
        compile_fn=_rowdot_compile,
        run_fn=_rowdot_run,
        perf_fn=_rowdot_benchmark,
    ),
    "cond_reduce":
    KernelSpec(
        name="cond_reduce",
        description="column sum behind a data-dependent branch (control-flow example)",
        output_arity=1,
        axes=_REDUCTION_AXES,
        precision_size=(128, 8192),
        perf_size=(128, 8192),
        known_limitation=("data-dependent control flow: the PTX reduction reconstruction is branch-blind "
                          "(no CFG model), so a runtime branch over the reduction is a soundness gap"),
        compile_fn=_cond_compile,
        run_fn=_cond_run,
        perf_fn=_cond_benchmark,
    ),
    # GEMM (M3) -- DRAFT, not GPU-verified. Stage-3 perf deferred (perf_fn=None).
    "gemm_f16_free_tiling":
    KernelSpec(
        name="gemm_f16_free_tiling",
        description="f16 wgmma GEMM; free-var positive -- only M/N tiling + num_warps vary, must stay ONE bitwise class",
        output_arity=1,
        axes=("num_warps", "gemm_block_m", "gemm_block_n"),
        precision_size=(256, 256, 256),
        perf_size=(256, 256, 256),
        known_limitation="",
        compile_fn=_gemm_free_tiling_compile,
        run_fn=_gemm_free_tiling_run,
        config_filter=_gemm_num_warps_filter,
    ),
    "gemm_kloop_block_k":
    KernelSpec(
        name="gemm_kloop_block_k",
        description="bf16 K-looped GEMM; M/N tiling FREE, block_k is a FIXED bit knob (splits classes)",
        output_arity=1,
        axes=("num_warps", "num_stages", "gemm_block_m", "gemm_block_n", "gemm_block_k"),
        precision_size=(256, 256, 256),
        perf_size=(256, 256, 256),
        known_limitation="",
        compile_fn=_gemm_kloop_compile,
        run_fn=_gemm_kloop_run,
        config_filter=_gemm_num_warps_filter,
        axis_values={"gemm_block_k": (16, 32)},  # sweep the splitting knob at every effort (light gemm_block_k is single-valued)
    ),
    "gemm_input_precision":
    KernelSpec(
        name="gemm_input_precision",
        description="f32 GEMM; input_precision tf32/ieee/tf32x3 must land in 3 distinct classes",
        output_arity=1,
        axes=("num_warps", "gemm_block_m", "gemm_block_n", "input_precision"),
        precision_size=(256, 256, 256),
        perf_size=(256, 256, 256),
        known_limitation="",
        compile_fn=_gemm_prec_compile,
        run_fn=_gemm_prec_run,
        config_filter=_gemm_num_warps_filter,
    ),
    "gemm_bias_relu_fp_fusion":
    KernelSpec(
        name="gemm_bias_relu_fp_fusion",
        description="bf16 GEMM + relu(acc*alpha+bias); enable_fp_fusion on/off -> 2 classes (fma vs mul+add)",
        output_arity=1,
        axes=("enable_fp_fusion", "num_warps", "gemm_block_m", "gemm_block_n"),
        precision_size=(256, 256, 256),
        perf_size=(256, 256, 256),
        known_limitation="",
        compile_fn=_gemm_bias_relu_compile,
        run_fn=_gemm_bias_relu_run,
        config_filter=_gemm_num_warps_filter,
        axis_values={"enable_fp_fusion": (True, False)},  # on/off split at every effort (light enable_fp_fusion is single-valued)
    ),
    "gemm_fp8_imprecise_acc":
    KernelSpec(
        name="gemm_fp8_imprecise_acc",
        description="fp8 e4m3 GEMM; max_num_imprecise_acc (partial-accumulator flush) is a FIXED bit knob",
        output_arity=1,
        axes=("num_warps", "gemm_block_m", "gemm_block_n", "max_num_imprecise_acc"),
        precision_size=(256, 256, 512),
        perf_size=(256, 256, 512),
        known_limitation="",
        compile_fn=_gemm_fp8_compile,
        run_fn=_gemm_fp8_run,
        config_filter=_gemm_num_warps_filter,
    ),
    "gemm_tma_store":
    KernelSpec(
        name="gemm_tma_store",
        description="f16 wgmma GEMM with a TMA-descriptor epilogue (no st.global root)",
        output_arity=1,
        axes=("num_warps", "gemm_block_m", "gemm_block_n"),
        precision_size=(256, 256, 256),
        perf_size=(256, 256, 256),
        known_limitation=("C is stored via a TMA descriptor (cp.async.bulk.tensor), not st.global; the PTX checker's "
                          "builder.roots() finds no root -> empty descriptor -> UNSUPPORTED. (TMA API + host "
                          "allocator wiring is unverified on the target Triton.)"),
        compile_fn=_gemm_tma_store_compile,
        run_fn=_gemm_tma_store_run,
        config_filter=_gemm_num_warps_filter,
    ),
}

# fp8 e4m3 is absent on older torch builds; drop the fp8 spec if so, to keep the
# import safe and avoid a missing-dtype AttributeError masquerading as an
# UNSUPPORTED checker verdict.
if not hasattr(torch, "float8_e4m3fn"):
    REGISTRY.pop("gemm_fp8_imprecise_acc", None)


# =========================================================================== #
# GROUP 2 — OPTIMIZATION-LAYOUT KERNELS (M2): large legal-layout space.
# =========================================================================== #
# WHY: the GROUP 1 reductions are effectively 1-D (e.g. sum_dim1_simple uses a
# [1, N] tile), so all lanes + warps are FORCED onto the reduce axis and there is
# no better layout to pick — an M2 reduction-layout pass has nothing to optimize.
# The kernels below reduce a genuinely MULTI-DIM tile with real KEPT dims, so the
# compiler has a real layout choice (warps/lanes can move OFF the reduce axis;
# warpsPerCTA[axis]=1 deletes the whole cross-warp shared-memory stage). Higher
# rank (3-D/4-D) and fusions make the legal layout space large and diverse.
#
# TARGETS vs CONTROLS (empirical, H100, via the reduce operand's warpsPerCTA): the
# compiler ALREADY sets warpsPerCTA[axis]=1 for inner-contiguous-axis reductions
# with kept dims big enough to absorb the warps (sum_2d_keepdim, sum_3d, sum_4d,
# sum_multiaxis, the fusions) — CONTROLS M2 must leave alone (gap ~0). The WIN case
# is a NON-contiguous / OUTER axis reduce the compiler under-parallelizes so the
# cross-warp stage dominates: sum_2d_axis0 / sum_2d_col / sum_2d_col_big (axis 0),
# sum_3d_outer, sum_3d_mid. The win GROWS with the reduce extent. col_dot is a
# mul-fed reduce that FMA-contracts under enable_fp_fusion (layout-sensitive, NOT
# covered by inner_tree) so the pass SKIPS it — a bit-safety control. Every kernel
# processes ONE multi-dim tile per program (grid = tile count); tile rank/shape is
# fixed as tl.constexpr args, so `size` stays the 2-tuple (grid, reduce_extent).
@triton.jit
def _k_sum2d_keepdim(X, Out, R: tl.constexpr, N: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    r = tl.arange(0, R)[:, None]
    n = tl.arange(0, N)[None, :]
    x = tl.load(X + g * (R * N) + r * N + n)  # [R, N]
    s = tl.sum(x, axis=1, reduction_ordering=ORD)  # [R]
    tl.store(Out + g * R + tl.arange(0, R), s)


@triton.jit
def _k_sum2d_axis0(X, Out, M: tl.constexpr, C: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    m = tl.arange(0, M)[:, None]
    c = tl.arange(0, C)[None, :]
    x = tl.load(X + g * (M * C) + m * C + c)  # [M, C]
    s = tl.sum(x, axis=0, reduction_ordering=ORD)  # [C] (reduce non-contiguous axis)
    tl.store(Out + g * C + tl.arange(0, C), s)


@triton.jit
def _k_sum3d(X, Out, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    b = tl.arange(0, B)[:, None, None]
    m = tl.arange(0, M)[None, :, None]
    n = tl.arange(0, N)[None, None, :]
    x = tl.load(X + g * (B * M * N) + b * (M * N) + m * N + n)  # [B, M, N]
    s = tl.sum(x, axis=2, reduction_ordering=ORD)  # [B, M]
    bo = tl.arange(0, B)[:, None]
    mo = tl.arange(0, M)[None, :]
    tl.store(Out + g * (B * M) + bo * M + mo, s)


@triton.jit
def _k_sum3d_mid(X, Out, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    b = tl.arange(0, B)[:, None, None]
    m = tl.arange(0, M)[None, :, None]
    n = tl.arange(0, N)[None, None, :]
    x = tl.load(X + g * (B * M * N) + b * (M * N) + m * N + n)  # [B, M, N]
    s = tl.sum(x, axis=1, reduction_ordering=ORD)  # [B, N] (reduce the middle axis)
    bo = tl.arange(0, B)[:, None]
    no = tl.arange(0, N)[None, :]
    tl.store(Out + g * (B * N) + bo * N + no, s)


@triton.jit
def _k_sum4d(X, Out, A: tl.constexpr, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    a = tl.arange(0, A)[:, None, None, None]
    b = tl.arange(0, B)[None, :, None, None]
    m = tl.arange(0, M)[None, None, :, None]
    n = tl.arange(0, N)[None, None, None, :]
    x = tl.load(X + g * (A * B * M * N) + a * (B * M * N) + b * (M * N) + m * N + n)  # [A, B, M, N]
    s = tl.sum(x, axis=3, reduction_ordering=ORD)  # [A, B, M]
    ao = tl.arange(0, A)[:, None, None]
    bo = tl.arange(0, B)[None, :, None]
    mo = tl.arange(0, M)[None, None, :]
    tl.store(Out + g * (A * B * M) + ao * (B * M) + bo * M + mo, s)


@triton.jit
def _k_sum_multiaxis(X, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    m = tl.arange(0, M)[:, None, None]
    n = tl.arange(0, N)[None, :, None]
    k = tl.arange(0, K)[None, None, :]
    x = tl.load(X + g * (M * N * K) + m * (N * K) + n * K + k)  # [M, N, K]
    s1 = tl.sum(x, axis=2, reduction_ordering=ORD)  # [M, N]
    s = tl.sum(s1, axis=1, reduction_ordering=ORD)  # [M]
    tl.store(Out + g * M + tl.arange(0, M), s)


@triton.jit
def _k_sum3d_outer(X, Out, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    b = tl.arange(0, B)[:, None, None]
    m = tl.arange(0, M)[None, :, None]
    n = tl.arange(0, N)[None, None, :]
    x = tl.load(X + g * (B * M * N) + b * (M * N) + m * N + n)  # [B, M, N]
    s = tl.sum(x, axis=0, reduction_ordering=ORD)  # [M, N] (reduce the OUTER axis)
    mo = tl.arange(0, M)[:, None]
    no = tl.arange(0, N)[None, :]
    tl.store(Out + g * (M * N) + mo * N + no, s)


@triton.jit
def _k_layernorm(X, Out, R: tl.constexpr, N: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    r = tl.arange(0, R)[:, None]
    n = tl.arange(0, N)[None, :]
    off = g * (R * N) + r * N + n
    x = tl.load(X + off)  # [R, N]
    mean = tl.sum(x, axis=1, reduction_ordering=ORD) / N  # [R]
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1, reduction_ordering=ORD) / N  # [R]
    y = xc / tl.sqrt(var[:, None] + 1e-6)
    tl.store(Out + off, y)


@triton.jit
def _k_softmax(X, Out, R: tl.constexpr, N: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    r = tl.arange(0, R)[:, None]
    n = tl.arange(0, N)[None, :]
    off = g * (R * N) + r * N + n
    x = tl.load(X + off)  # [R, N]
    mx = tl.max(x, axis=1)  # [R] — max is order-invariant, so no reduction_ordering
    e = tl.exp(x - mx[:, None])
    s = tl.sum(e, axis=1, reduction_ordering=ORD)  # [R] — the order-sensitive reduce
    y = e / s[:, None]
    tl.store(Out + off, y)


@triton.jit
def _k_rmsnorm(X, Out, R: tl.constexpr, N: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    r = tl.arange(0, R)[:, None]
    n = tl.arange(0, N)[None, :]
    off = g * (R * N) + r * N + n
    x = tl.load(X + off)  # [R, N]
    ms = tl.sum(x * x, axis=1, reduction_ordering=ORD) / N  # [R]
    y = x / tl.sqrt(ms[:, None] + 1e-6)
    tl.store(Out + off, y)


# Diverse structure/compute (not just single-tile column sums): a for-loop, a
# two-input mul-fed reduce, a transcendental, a bf16 input, and a max combine. All
# reduce the non-contiguous axis 0 so they exercise M2 (except max, which is
# order-invariant -> a control the pass must leave to the existing heuristics).
@triton.jit
def _k_col_sum_loop(X, Out, M: tl.constexpr, C: tl.constexpr, CHUNK: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    c = tl.arange(0, C)[None, :]
    acc = tl.zeros([C], dtype=tl.float32)
    for r0 in range(0, M, CHUNK):
        r = (r0 + tl.arange(0, CHUNK))[:, None]
        x = tl.load(X + g * (M * C) + r * C + c)  # [CHUNK, C]
        acc += tl.sum(x, axis=0, reduction_ordering=ORD)  # per-chunk column sum (inner_tree within chunk)
    tl.store(Out + g * C + tl.arange(0, C), acc)


@triton.jit
def _k_col_dot(A, B, Out, M: tl.constexpr, C: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    m = tl.arange(0, M)[:, None]
    c = tl.arange(0, C)[None, :]
    off = g * (M * C) + m * C + c
    a = tl.load(A + off)
    b = tl.load(B + off)
    s = tl.sum(a * b, axis=0, reduction_ordering=ORD)  # [C] column-wise dot (mul-fed reduce)
    tl.store(Out + g * C + tl.arange(0, C), s)


@triton.jit
def _k_col_exp_sum(X, Out, M: tl.constexpr, C: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    m = tl.arange(0, M)[:, None]
    c = tl.arange(0, C)[None, :]
    x = tl.load(X + g * (M * C) + m * C + c)
    s = tl.sum(tl.exp(x), axis=0, reduction_ordering=ORD)  # [C] transcendental then reduce
    tl.store(Out + g * C + tl.arange(0, C), s)


@triton.jit
def _k_col_bf16(X, Out, M: tl.constexpr, C: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    m = tl.arange(0, M)[:, None]
    c = tl.arange(0, C)[None, :]
    x = tl.load(X + g * (M * C) + m * C + c)  # bf16 input
    s = tl.sum(x.to(tl.float32), axis=0, reduction_ordering=ORD)  # accumulate in fp32
    tl.store(Out + g * C + tl.arange(0, C), s)


@triton.jit
def _k_col_max(X, Out, M: tl.constexpr, C: tl.constexpr, ORD: tl.constexpr):
    g = tl.program_id(0)
    m = tl.arange(0, M)[:, None]
    c = tl.arange(0, C)[None, :]
    x = tl.load(X + g * (M * C) + m * C + c)
    s = tl.max(x, axis=0)  # max is order-invariant: no reduction_ordering, so M2 leaves it alone
    tl.store(Out + g * C + tl.arange(0, C), s)


# One record per GROUP 2 kernel + a factory that builds compile/run/benchmark from it.
@dataclass(frozen=True)
class _LK:
    jit: object  # the @triton.jit function
    consts: dict  # constexpr tile dims (kwargs to warmup), e.g. {"R": 32, "N": 128}
    out_numel: int  # output elements per program
    reduce_extent: int  # reduce-axis size, for display only
    fusion: bool = False  # fusion kernels use plain (realistic) inputs
    n_inputs: int = 1  # number of input tensors the kernel takes (dot = 2)
    in_dtype: object = None  # input tensor dtype (None -> float32; e.g. torch.bfloat16)
    input_numel: int = 0  # override per-program input size (looped kernels: total loaded != prod(consts))

    @property
    def tile_numel(self):
        if self.input_numel:
            return self.input_numel
        n = 1
        for v in self.consts.values():
            n *= v
        return n

    @property
    def input_fn(self):
        return _plain_nd if self.fusion else _adv_nd


def _mk_compile(lk):
    def compile_fn(config, size, maxnreg=None):
        grid = size[0]
        dt = lk.in_dtype or torch.float32
        # warmup does not launch, so the data is irrelevant — use cheap buffers.
        ins = [torch.empty(grid * lk.tile_numel, device=DEVICE, dtype=dt) for _ in range(lk.n_inputs)]
        out = torch.empty(grid * lk.out_numel, device=DEVICE, dtype=torch.float32)
        return lk.jit.warmup(*ins, out, grid=(grid, ), ORD=_ORDERING[config.reduction_ordering],
                             num_warps=config.num_warps, num_stages=config.num_stages,
                             enable_fp_fusion=config.enable_fp_fusion, maxnreg=maxnreg, **lk.consts)

    return compile_fn


def _mk_run(lk):
    def run_fn(config, ck, seed, size):
        grid = size[0]
        dt = lk.in_dtype or torch.float32
        ins = [lk.input_fn(grid * lk.tile_numel, seed + i).to(dt) for i in range(lk.n_inputs)]
        out = torch.empty(grid * lk.out_numel, device=DEVICE, dtype=torch.float32)
        ck[(grid, 1, 1)](*ins, out)
        torch.cuda.synchronize()
        return _to_bytes(out)

    return run_fn


def _mk_bench(lk):
    def perf_fn(config, size):
        grid = size[0]
        dt = lk.in_dtype or torch.float32
        ins = [_plain_nd(grid * lk.tile_numel, i).to(dt) for i in range(lk.n_inputs)]
        out = torch.empty(grid * lk.out_numel, device=DEVICE, dtype=torch.float32)
        ck = _mk_compile(lk)(config, size)

        def thunk():
            ck[(grid, 1, 1)](*ins, out)

        thunk()
        torch.cuda.synchronize()
        return _bench_ms(thunk), _to_bytes(out), ck.asm

    return perf_fn


def _layout_spec(name, description, lk, precision_grid=64, perf_grid=2048):
    return KernelSpec(
        name=name,
        description=description,
        output_arity=1,
        axes=_REDUCTION_AXES,
        precision_size=(precision_grid, lk.reduce_extent),
        perf_size=(perf_grid, lk.reduce_extent),
        known_limitation="",
        compile_fn=_mk_compile(lk),
        run_fn=_mk_run(lk),
        perf_fn=_mk_bench(lk),
    )


REGISTRY.update({
    "sum_2d_keepdim":
    _layout_spec("sum_2d_keepdim", "[R=32,N=128] reduce axis=1; warps can move to the kept R axis (canonical M2 win)",
                 _LK(_k_sum2d_keepdim, {"R": 32, "N": 128}, out_numel=32, reduce_extent=128)),
    "sum_2d_axis0":
    _layout_spec("sum_2d_axis0", "[M=128,C=32] reduce axis=0 (non-contiguous reduce; coalesce-vs-fold conflict)",
                 _LK(_k_sum2d_axis0, {"M": 128, "C": 32}, out_numel=32, reduce_extent=128)),
    # Exposed-reduce M2 TARGETS: non-contiguous / outer-axis reductions at increasing size.
    # The compiler under-parallelizes the reduce axis within the warp, so the cross-warp
    # stage dominates -> the pass wins, and the win grows with the reduce extent.
    "sum_2d_col":
    _layout_spec("sum_2d_col", "[M=256,C=32] reduce axis=0 (bigger non-contiguous reduce)",
                 _LK(_k_sum2d_axis0, {"M": 256, "C": 32}, out_numel=32, reduce_extent=256)),
    "sum_2d_col_big":
    _layout_spec("sum_2d_col_big", "[M=1024,C=32] reduce axis=0 (large 32K-tile non-contiguous reduce)",
                 _LK(_k_sum2d_axis0, {"M": 1024, "C": 32}, out_numel=32, reduce_extent=1024)),
    "sum_3d_outer":
    _layout_spec("sum_3d_outer", "[B=64,M=8,N=16] reduce OUTER axis=0 (non-contiguous reduce in 3-D)",
                 _LK(_k_sum3d_outer, {"B": 64, "M": 8, "N": 16}, out_numel=8 * 16, reduce_extent=64)),
    # Diverse structure / compute (a for-loop, mul-fed dot, transcendental, bf16, max op).
    "col_sum_loop":
    _layout_spec("col_sum_loop", "[M=512,C=32] LOOPED column sum over CHUNK=128 row-blocks (for-loop structure)",
                 _LK(_k_col_sum_loop, {"M": 512, "C": 32, "CHUNK": 128}, out_numel=32, reduce_extent=128,
                     input_numel=512 * 32)),
    "col_dot":
    _layout_spec("col_dot", "[M=256,C=32] column-wise dot sum(a*b) axis=0 (two inputs, mul-fed reduce)",
                 _LK(_k_col_dot, {"M": 256, "C": 32}, out_numel=32, reduce_extent=256, n_inputs=2)),
    "col_exp_sum":
    _layout_spec("col_exp_sum", "[M=256,C=32] sum(exp(x)) axis=0 (transcendental compute then reduce)",
                 _LK(_k_col_exp_sum, {"M": 256, "C": 32}, out_numel=32, reduce_extent=256, fusion=True)),
    "col_bf16":
    _layout_spec("col_bf16", "[M=256,C=32] bf16-input column sum axis=0, fp32 accumulate (dtype diversity)",
                 _LK(_k_col_bf16, {"M": 256, "C": 32}, out_numel=32, reduce_extent=256, in_dtype=torch.bfloat16)),
    "col_max":
    _layout_spec("col_max", "[M=256,C=32] column MAX axis=0 (order-invariant op; M2 control)",
                 _LK(_k_col_max, {"M": 256, "C": 32}, out_numel=32, reduce_extent=256)),
    "sum_2d_deep":
    _layout_spec("sum_2d_deep", "[R=8,N=512] deep reduce axis=1 (large reduce extent, small kept dim)",
                 _LK(_k_sum2d_keepdim, {"R": 8, "N": 512}, out_numel=8, reduce_extent=512)),
    "sum_3d":
    _layout_spec("sum_3d", "[B=8,M=8,N=64] reduce inner axis=2 (rank-3 layout space)",
                 _LK(_k_sum3d, {"B": 8, "M": 8, "N": 64}, out_numel=64, reduce_extent=64)),
    "sum_3d_mid":
    _layout_spec("sum_3d_mid", "[B=8,M=32,N=16] reduce MIDDLE axis=1 (non-contiguous reduce in 3-D)",
                 _LK(_k_sum3d_mid, {"B": 8, "M": 32, "N": 16}, out_numel=8 * 16, reduce_extent=32)),
    "sum_4d":
    _layout_spec("sum_4d", "[A=4,B=4,M=8,N=32] reduce inner axis=3 (rank-4 layout space)",
                 _LK(_k_sum4d, {"A": 4, "B": 4, "M": 8, "N": 32}, out_numel=4 * 4 * 8, reduce_extent=32)),
    "sum_multiaxis":
    _layout_spec("sum_multiaxis", "[M=32,N=8,K=16] reduce axes 2 then 1 (multi-axis reduce)",
                 _LK(_k_sum_multiaxis, {"M": 32, "N": 8, "K": 16}, out_numel=32, reduce_extent=8 * 16)),
    "layernorm":
    _layout_spec("layernorm", "[R=16,N=128] mean/var reduce + normalize (reduce+broadcast+elementwise fusion)",
                 _LK(_k_layernorm, {"R": 16, "N": 128}, out_numel=16 * 128, reduce_extent=128, fusion=True)),
    "softmax":
    _layout_spec("softmax", "[R=16,N=128] row-max + row-sum reduce + broadcast div (fusion; sum is order-sensitive)",
                 _LK(_k_softmax, {"R": 16, "N": 128}, out_numel=16 * 128, reduce_extent=128, fusion=True)),
    "rmsnorm":
    _layout_spec("rmsnorm", "[R=16,N=128] mean-square reduce + broadcast mul (fusion)",
                 _LK(_k_rmsnorm, {"R": 16, "N": 128}, out_numel=16 * 128, reduce_extent=128, fusion=True)),
})


# =========================================================================== #
# GROUP 3 — DIVERSITY / ROBUSTNESS KERNELS (placeholder — next-step job).
# =========================================================================== #
# Intentionally empty for now. This group will hold a broad, diverse set of
# reduction kernels (varied purpose / control-flow / dtype / rank / op) chosen so
# that most have a real base(inner_tree) != ceiling(unordered) gap, to stress both
# the checker (soundness) and the M2 pass (robustness). A 29-kernel draft of these
# lives in the history of D111025844; re-adding + curating them here is a follow-up.


def resolve_kernels(selector):
    """Map a ``--kernels`` selector ("all" or a comma list) to a list of KernelSpec."""
    if selector in (None, "all"):
        return list(REGISTRY.values())
    out = []
    for name in selector.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in REGISTRY:
            raise ValueError(f"unknown kernel {name!r}; available: {', '.join(REGISTRY)}")
        out.append(REGISTRY[name])
    if not out:
        raise ValueError("no kernels selected")
    return out
