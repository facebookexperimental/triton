"""Test kernels for the bitequiv evaluation framework, one place for all of them.

Every kernel the framework measures lives here, wrapped in a :class:`KernelSpec`
and collected in :data:`REGISTRY`. A spec knows everything the driver
(``evaluate.py``) needs and nothing about the checker or the fuzzer:

  * which autotuner knobs the kernel accepts (``axes``) and how to expand them
    into a config list at a given effort (``config_space``);
  * how to ``compile`` a config to a ``CompiledKernel`` (the driver reads
    ``ck.asm[<artifact>]`` — ``ptx`` or ``ttgir`` per ``--artifact`` — for the checker);
  * how to ``run`` a config on a random seed and return the output ``bytes``
    (the fuzzer's ground-truth observation; welford concatenates Mean + Var);
  * a small precision size for Stages 1-2 and a perf size + ``benchmark`` hook so
    Stage 3 can time every kernel across its config space.

All inputs are float32 with a wide dynamic range and alternating signs
(``_adv_sum_2d`` / ``_adv_dot_2d``): that is the regime where reduction ORDER
actually changes the result bits, which is exactly what the checker claims to
track. Plain unit-scale data would round every order to the same bits and hide
the signal.

Kernels:
  * ``sum_dim1_simple``     — single-tile column sum (no loop).
  * ``sum_dim1_persistent`` — column sum looped over BLOCK_N chunks.
  * ``welford``             — one-pass mean/variance (2 outputs, custom combine).
  * ``sum``                 — single-tile row sum.
  * ``dot``                 — single-tile row dot product (mul-fed reduction).
  * ``cond_reduce``         — column sum behind a DATA-DEPENDENT branch; exists to
                              exercise the checker's control-flow gap (Stage 1).

Kernels and the input helpers are ported from the former
``evaluate_ptx_kernel_suite.py`` monolith; the abbreviations there are spelled
out in full here (``num_warps``, ``num_stages``, ``block_n``, ...).
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
_AXIS_ORDER = ("reduction_ordering", "num_warps", "num_stages", "enable_fp_fusion", "block_n")
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
    },
    "heavy": {
        "reduction_ordering": ("unordered", "inner_tree"),
        "num_warps": (1, 2, 4, 8, 16, 32),
        "num_stages": (1, 2, 3, 4, 5, 6),
        "enable_fp_fusion": (True, False),
        "block_n": (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
    },
}


def build_configs(axes, config_effort):
    """Cartesian product of the kernel's ``axes`` at the given effort.

    Axes the kernel does not use are pinned to ``None`` so every kernel yields a
    uniform :class:`Config`.
    """
    try:
        values = _AXIS_VALUES[config_effort]
    except KeyError:
        raise ValueError(f"unknown config effort {config_effort!r}; expected one of {sorted(_AXIS_VALUES)}")
    per_axis = [values[a] if a in axes else (None, ) for a in _AXIS_ORDER]
    return [Config(*combo) for combo in itertools.product(*per_axis)]


def config_label(config):
    """Human-readable one-line label for a config (only the knobs it actually uses)."""
    parts = []
    if config.reduction_ordering is not None:
        parts.append(f"reduction_ordering={config.reduction_ordering}")
    parts.append(f"num_warps={config.num_warps}")
    parts.append(f"num_stages={config.num_stages}")
    parts.append(f"enable_fp_fusion={'on' if config.enable_fp_fusion else 'off'}")
    if config.block_n is not None:
        parts.append(f"block_n={config.block_n}")
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


def _to_bytes(t):
    """Exact output bits as a ``bytes`` object (the fuzzer's equality unit)."""
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
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
def _simple_compile(config, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return sum_dim1_kernel.warmup(src, out, rows, cols, src.stride(0), src.stride(1), grid=(rows, ), BLOCK_N=cols,
                                  ROWS_PER_BLOCK=1, REDUCTION_ORDERING=_ORDERING[config.reduction_ordering],
                                  num_warps=config.num_warps, num_stages=config.num_stages,
                                  enable_fp_fusion=config.enable_fp_fusion)


def _simple_run(config, ck, seed, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, seed)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](src, out, rows, cols, src.stride(0), src.stride(1))
    torch.cuda.synchronize()
    return _to_bytes(out)


def _persist_compile(config, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return sum_dim1_kernel_batched.warmup(src, out, rows, cols, src.stride(0), src.stride(1), grid=(rows, ),
                                          BLOCK_N=config.block_n, ROWS_PER_BLOCK=1,
                                          REDUCTION_ORDERING=_ORDERING[config.reduction_ordering],
                                          num_warps=config.num_warps, num_stages=config.num_stages,
                                          enable_fp_fusion=config.enable_fp_fusion)


def _persist_run(config, ck, seed, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, seed)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](src, out, rows, cols, src.stride(0), src.stride(1))
    torch.cuda.synchronize()
    return _to_bytes(out)


def _welford_compile(config, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    mean = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    var = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return welford_kernel.warmup(src, mean, var, cols, src.stride(0), grid=(rows, ), BLOCK_SIZE=config.block_n,
                                 num_warps=config.num_warps, num_stages=config.num_stages,
                                 enable_fp_fusion=config.enable_fp_fusion)


def _welford_run(config, ck, seed, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, seed)
    mean = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    var = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](src, mean, var, cols, src.stride(0))
    torch.cuda.synchronize()
    return _to_bytes(mean) + _to_bytes(var)  # both outputs must match for bit-equivalence


def _rowsum_compile(config, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return rowsum_kernel.warmup(src, out, cols, src.stride(0), grid=(rows, ), BLOCK=cols,
                                ORD=_ORDERING[config.reduction_ordering], num_warps=config.num_warps,
                                num_stages=config.num_stages, enable_fp_fusion=config.enable_fp_fusion)


def _rowsum_run(config, ck, seed, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, seed)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](src, out, cols, src.stride(0))
    torch.cuda.synchronize()
    return _to_bytes(out)


def _rowdot_compile(config, size):
    rows, cols = size
    a, b = _adv_dot_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return rowdot_kernel.warmup(a, b, out, cols, a.stride(0), grid=(rows, ), BLOCK=cols,
                                ORD=_ORDERING[config.reduction_ordering], num_warps=config.num_warps,
                                num_stages=config.num_stages, enable_fp_fusion=config.enable_fp_fusion)


def _rowdot_run(config, ck, seed, size):
    rows, cols = size
    a, b = _adv_dot_2d(rows, cols, seed)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    ck[(rows, 1, 1)](a, b, out, cols, a.stride(0))
    torch.cuda.synchronize()
    return _to_bytes(out)


def _cond_compile(config, size):
    rows, cols = size
    src = _adv_sum_2d(rows, cols, 0)
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    return cond_reduce_kernel.warmup(src, out, rows, cols, src.stride(0), src.stride(1), grid=(rows, ), BLOCK_N=cols,
                                     ROWS_PER_BLOCK=1, REDUCTION_ORDERING=_ORDERING[config.reduction_ordering],
                                     num_warps=config.num_warps, num_stages=config.num_stages,
                                     enable_fp_fusion=config.enable_fp_fusion)


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

    @property
    def supports_perf(self):
        return self.perf_fn is not None

    def config_space(self, config_effort):
        return build_configs(self.axes, config_effort)

    def compile(self, config, size):
        return self.compile_fn(config, size)

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
}


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
