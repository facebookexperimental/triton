"""benchmark_kernels.py — a standalone, runnable Triton kernel benchmark zoo.

A broad, project-agnostic collection of Triton kernels covering the general usage of
Triton across the ecosystem (pointwise, activations, dropout, normalization, softmax,
reductions, scan/SSM, GEMM, quantization, attention, losses, embedding, sort, rope,
convolution, pooling, MoE, special functions). It is intentionally SELF-CONTAINED:

  * it depends on **torch + triton ONLY** — nothing from the libraries the kernels were
    adapted from, and nothing from any surrounding project. Drop this one file into any
    repo and it runs.
  * every kernel ships an input builder, a launch, an optional torch reference (for a
    correctness check), and is timed with ``triton.testing.do_bench``.

The kernel bodies are adapted (faithfully — library decorators/wrappers stripped, helper
functions inlined) from these open-source projects; each GROUP banner links its source:

  * Triton tutorials       — https://github.com/triton-lang/triton
  * tritonbench            — https://github.com/pytorch-labs/tritonbench
  * FlagGems               — https://github.com/FlagOpen/FlagGems
  * flash-linear-attention — https://github.com/fla-org/flash-linear-attention
  * torchao                — https://github.com/pytorch/ao

Usage:
  python benchmark_kernels.py                 # run + time (and check) every kernel
  python benchmark_kernels.py --list          # list kernels (category, name, source)
  python benchmark_kernels.py --filter gemm   # only kernels whose name/category matches
  python benchmark_kernels.py --only vector_add
  python benchmark_kernels.py --no-check      # skip correctness, time only

Adding a kernel: write a plain ``@triton.jit`` body (torch+triton only — inline any helper),
then ``register(name=..., category=..., source=<github link>, make_inputs=..., run=..., ref=...)``.
Keep shapes modest but representative so the whole suite runs quickly.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import torch

import triton
import triton.language as tl

# libdevice (erf/tanh/rsqrt/... ) — import robustly across Triton versions; kernels that use
# it fail gracefully (caught per-kernel by the runner) if it is unavailable.
try:
    from triton.language.extra import libdevice
except Exception:  # noqa: BLE001
    try:
        import triton.language.math as libdevice  # older layout
    except Exception:  # noqa: BLE001
        libdevice = None

DEVICE = "cuda"


# --------------------------------------------------------------------------- #
# Input helpers (seeded, reproducible)
# --------------------------------------------------------------------------- #
def randn(*shape, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(DEVICE, dtype)


def rand(*shape, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float32).to(DEVICE, dtype)


def randint(high, shape, seed=0, dtype=torch.int32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, high, shape, generator=g, dtype=torch.int64).to(DEVICE, dtype)


def ones(*shape, dtype=torch.float32):
    return torch.ones(*shape, device=DEVICE, dtype=dtype)


# --------------------------------------------------------------------------- #
# Benchmark spec + registry
# --------------------------------------------------------------------------- #
@dataclass
class Bench:
    name: str
    category: str
    source: str  # github link to the kernel this was adapted from
    make_inputs: Callable  # () -> tuple of args passed to run/ref
    run: Callable  # (inputs) -> output tensor(s); builds outputs + launches the jit
    ref: Optional[Callable] = None  # (inputs) -> expected output(s) via torch eager; None => time-only
    rtol: float = 1e-2
    atol: float = 1e-2
    shape_note: str = ""
    runnable: bool = True  # False => SKIP (e.g. needs Blackwell sm100 / a HW path not in portable Triton)
    note: str = ""  # reason shown when skipped, or any caveat


BENCHMARKS: list = []


def register(**kwargs):
    BENCHMARKS.append(Bench(**kwargs))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _bench_ms(thunk):
    from triton.testing import do_bench
    return do_bench(thunk, warmup=25, rep=100)


def _allclose(out, exp, rtol, atol):
    if isinstance(out, (tuple, list)):
        return all(_allclose(o, e, rtol, atol) for o, e in zip(out, exp))
    return torch.allclose(out.float(), exp.float(), rtol=rtol, atol=atol)


def main(argv):
    ap = argparse.ArgumentParser(description="Standalone Triton kernel benchmark zoo.")
    ap.add_argument("--filter", default=None, help="substring match on kernel name OR category")
    ap.add_argument("--only", default=None, help="exact kernel name")
    ap.add_argument("--list", action="store_true", help="list kernels and exit")
    ap.add_argument("--no-check", action="store_true", help="skip correctness, time only")
    args = ap.parse_args(argv)

    benches = BENCHMARKS
    if args.only:
        benches = [b for b in benches if b.name == args.only]
    elif args.filter:
        benches = [b for b in benches if args.filter in b.name or args.filter in b.category]

    if args.list:
        for b in benches:
            print(f"{b.category:20s} {b.name:34s} {b.source}")
        print(f"\n{len(benches)} kernels ({len(BENCHMARKS)} total).")
        return

    if not torch.cuda.is_available():
        raise SystemExit("benchmark_kernels needs a CUDA device.")

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"{'category':20s} {'kernel':34s} {'shape':24s} {'ms':>10s} {'correct':>8s}")
    n_ok = n_fail = n_skip = 0
    for b in benches:
        if not b.runnable:
            print(f"{b.category:20s} {b.name:34s} {b.shape_note:24s} {'SKIP':>10s}  {b.note}")
            n_skip += 1
            continue
        try:
            inp = b.make_inputs()
            out = b.run(inp)  # warmup + compile
            torch.cuda.synchronize()
            correct = "-"
            if b.ref is not None and not args.no_check:
                correct = str(_allclose(out, b.ref(inp), b.rtol, b.atol))
            ms = _bench_ms(lambda: b.run(inp))
            print(f"{b.category:20s} {b.name:34s} {b.shape_note:24s} {ms:10.4f} {correct:>8s}")
            n_ok += 1
        except Exception as e:  # noqa: BLE001 - one bad kernel must not abort the suite
            msg = str(e).splitlines()[0][:52] if str(e) else ""
            print(f"{b.category:20s} {b.name:34s} {'':24s} {'FAIL':>10s}  {type(e).__name__}: {msg}")
            n_fail += 1
    print(f"\n{n_ok} ran, {n_fail} failed, {n_skip} skipped (see notes), of {len(benches)} selected.")


# ######################################################################################### #
# ## GROUP 1 — POINTWISE / ELEMENTWISE                                                     ## #
# ## source: https://github.com/triton-lang/triton (python/tutorials/01-vector-add.py)    ## #
# ##         https://github.com/FlagOpen/FlagGems (src/flag_gems/ops)                      ## #
# ######################################################################################### #
@triton.jit
def _add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


def _run_add(inp):
    x, y = inp
    o = torch.empty_like(x)
    n = x.numel()
    _add_kernel[(triton.cdiv(n, 1024), )](x, y, o, n, BLOCK=1024)
    return o


register(name="vector_add", category="pointwise",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py",
         make_inputs=lambda: (randn(1 << 20, seed=1), randn(1 << 20, seed=2)),
         run=_run_add, ref=lambda inp: inp[0] + inp[1], shape_note="2^20")


@triton.jit
def _mul_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) * tl.load(y_ptr + offs, mask=m), mask=m)


def _run_mul(inp):
    x, y = inp
    o = torch.empty_like(x)
    n = x.numel()
    _mul_kernel[(triton.cdiv(n, 1024), )](x, y, o, n, BLOCK=1024)
    return o


register(name="vector_mul", category="pointwise",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/mul.py",
         make_inputs=lambda: (randn(1 << 20, seed=1), randn(1 << 20, seed=2)),
         run=_run_mul, ref=lambda inp: inp[0] * inp[1], shape_note="2^20")


@triton.jit
def _exp_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.exp(tl.load(x_ptr + offs, mask=m)), mask=m)


def _run_exp(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _exp_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="vector_exp", category="pointwise",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/vector_exp/kernels.py",
         make_inputs=lambda: (randn(1 << 20, seed=3), ),
         run=_run_exp, ref=lambda inp: torch.exp(inp[0]), shape_note="2^20")


@triton.jit
def _cast_bf16_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m).to(tl.bfloat16), mask=m)


def _run_cast_bf16(inp):
    x, = inp
    o = torch.empty(x.shape, device=DEVICE, dtype=torch.bfloat16)
    n = x.numel()
    _cast_bf16_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="cast_fp32_to_bf16", category="pointwise",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops",
         make_inputs=lambda: (randn(1 << 20, seed=4), ),
         run=_run_cast_bf16, ref=lambda inp: inp[0].to(torch.bfloat16), shape_note="2^20")


# ######################################################################################### #
# ## GROUP 2 — ACTIVATIONS (forward)                                                       ## #
# ## source: https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{gelu,silu,relu,...}) ## #
# ######################################################################################### #
@triton.jit
def _gelu_tanh_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)  # sqrt(2/pi)*(x+0.044715 x^3)
    tl.store(o_ptr + offs, 0.5 * x * (1.0 + libdevice.tanh(inner)), mask=m)


def _run_gelu_tanh(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _gelu_tanh_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="gelu_tanh", category="activation",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/gelu.py",
         make_inputs=lambda: (randn(1 << 20, seed=5), ),
         run=_run_gelu_tanh,
         ref=lambda inp: torch.nn.functional.gelu(inp[0], approximate="tanh"),
         rtol=2e-3, atol=2e-3, shape_note="2^20")


@triton.jit
def _gelu_erf_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, 0.5 * x * (1.0 + libdevice.erf(x * 0.7071067811865476)), mask=m)  # x/sqrt(2)


def _run_gelu_erf(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _gelu_erf_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="gelu_erf", category="activation",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/gelu.py",
         make_inputs=lambda: (randn(1 << 20, seed=6), ),
         run=_run_gelu_erf, ref=lambda inp: torch.nn.functional.gelu(inp[0], approximate="none"),
         rtol=2e-3, atol=2e-3, shape_note="2^20")


@triton.jit
def _silu_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x * tl.sigmoid(x), mask=m)


def _run_silu(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _silu_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="silu", category="activation",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/silu.py",
         make_inputs=lambda: (randn(1 << 20, seed=7), ),
         run=_run_silu, ref=lambda inp: torch.nn.functional.silu(inp[0]),
         rtol=2e-3, atol=2e-3, shape_note="2^20")


@triton.jit
def _relu_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, tl.where(x > 0, x, 0.0), mask=m)


def _run_relu(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _relu_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="relu", category="activation",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/relu.py",
         make_inputs=lambda: (randn(1 << 20, seed=8), ),
         run=_run_relu, ref=lambda inp: torch.nn.functional.relu(inp[0]), shape_note="2^20")


@triton.jit
def _sigmoid_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.sigmoid(tl.load(x_ptr + offs, mask=m)), mask=m)


def _run_sigmoid(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _sigmoid_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="sigmoid", category="activation",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/sigmoid.py",
         make_inputs=lambda: (randn(1 << 20, seed=9), ),
         run=_run_sigmoid, ref=lambda inp: torch.sigmoid(inp[0]),
         rtol=2e-3, atol=2e-3, shape_note="2^20")


@triton.jit
def _tanh_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, libdevice.tanh(tl.load(x_ptr + offs, mask=m)), mask=m)


def _run_tanh(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _tanh_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="tanh", category="activation",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/tanh.py",
         make_inputs=lambda: (randn(1 << 20, seed=10), ),
         run=_run_tanh, ref=lambda inp: torch.tanh(inp[0]), rtol=2e-3, atol=2e-3, shape_note="2^20")


@triton.jit
def _softplus_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    # numerically stable softplus: max(x,0) + log1p(exp(-|x|))
    tl.store(o_ptr + offs, tl.maximum(x, 0.0) + libdevice.log1p(tl.exp(-tl.abs(x))), mask=m)


def _run_softplus(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _softplus_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="softplus", category="activation",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/softplus.py",
         make_inputs=lambda: (randn(1 << 20, seed=11), ),
         run=_run_softplus, ref=lambda inp: torch.nn.functional.softplus(inp[0]),
         rtol=2e-3, atol=2e-3, shape_note="2^20")


# ######################################################################################### #
# ## GROUP 3 — FUSED ACTIVATIONS (gate * up, per-row tile)                                 ## #
# ## source: https://github.com/pytorch-labs/tritonbench (operators/swiglu, operators/geglu)## #
# ##         https://github.com/FlagOpen/FlagGems (src/flag_gems/fused)                    ## #
# ######################################################################################### #
@triton.jit
def _swiglu_kernel(g_ptr, u_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    off = row * N + cols
    g = tl.load(g_ptr + off, mask=m)
    u = tl.load(u_ptr + off, mask=m)
    tl.store(o_ptr + off, (g * tl.sigmoid(g)) * u, mask=m)


def _run_swiglu(inp):
    g, u = inp
    M, N = g.shape
    o = torch.empty_like(g)
    _swiglu_kernel[(M, )](g, u, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="swiglu", category="activation-fused",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/swiglu",
         make_inputs=lambda: (randn(1024, 4096, seed=12), randn(1024, 4096, seed=13)),
         run=_run_swiglu, ref=lambda inp: torch.nn.functional.silu(inp[0]) * inp[1],
         rtol=2e-3, atol=2e-3, shape_note="1024x4096")


@triton.jit
def _geglu_kernel(g_ptr, u_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    off = row * N + cols
    g = tl.load(g_ptr + off, mask=m)
    u = tl.load(u_ptr + off, mask=m)
    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    tl.store(o_ptr + off, (0.5 * g * (1.0 + libdevice.tanh(inner))) * u, mask=m)


def _run_geglu(inp):
    g, u = inp
    M, N = g.shape
    o = torch.empty_like(g)
    _geglu_kernel[(M, )](g, u, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="geglu", category="activation-fused",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/geglu",
         make_inputs=lambda: (randn(1024, 4096, seed=14), randn(1024, 4096, seed=15)),
         run=_run_geglu,
         ref=lambda inp: torch.nn.functional.gelu(inp[0], approximate="tanh") * inp[1],
         rtol=2e-3, atol=2e-3, shape_note="1024x4096")


@triton.jit
def _silu_and_mul_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    g = tl.load(x_ptr + row * (2 * N) + cols, mask=m)
    u = tl.load(x_ptr + row * (2 * N) + N + cols, mask=m)
    tl.store(o_ptr + row * N + cols, (g * tl.sigmoid(g)) * u, mask=m)


def _run_silu_and_mul(inp):
    x, = inp
    M, twoN = x.shape
    N = twoN // 2
    o = torch.empty((M, N), device=DEVICE, dtype=x.dtype)
    _silu_and_mul_kernel[(M, )](x, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


def _ref_silu_and_mul(inp):
    x, = inp
    n = x.shape[1] // 2
    return torch.nn.functional.silu(x[:, :n]) * x[:, n:]


register(name="silu_and_mul", category="activation-fused",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/fused/silu_and_mul.py",
         make_inputs=lambda: (randn(1024, 8192, seed=16), ),
         run=_run_silu_and_mul, ref=_ref_silu_and_mul, rtol=2e-3, atol=2e-3, shape_note="1024x(2*4096)")


# ######################################################################################### #
# ## GROUP 4 — DROPOUT (seeded philox RNG)                                                 ## #
# ## source: https://github.com/triton-lang/triton (python/tutorials/04-low-memory-dropout.py)## #
# ######################################################################################### #
@triton.jit
def _dropout_kernel(x_ptr, o_ptr, n, p, seed, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    keep = tl.rand(seed, offs) > p
    tl.store(o_ptr + offs, tl.where(keep, x / (1.0 - p), 0.0), mask=m)


def _run_dropout(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _dropout_kernel[(triton.cdiv(n, 1024), )](x, o, n, 0.3, 123, BLOCK=1024)
    return o


# Triton philox RNG != torch RNG, so no exact torch reference; time-only.
register(name="seeded_dropout", category="dropout",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/04-low-memory-dropout.py",
         make_inputs=lambda: (randn(1 << 20, seed=17), ), run=_run_dropout, ref=None,
         shape_note="2^20 p=0.3")


# ######################################################################################### #
# ## GROUP 5 — SPECIAL FUNCTIONS (libdevice)                                               ## #
# ## source: https://github.com/triton-lang/triton (python/tutorials/07-extern-functions.py)## #
# ##         https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{erf,log,...}.py)     ## #
# ######################################################################################### #
@triton.jit
def _erf_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, libdevice.erf(tl.load(x_ptr + offs, mask=m)), mask=m)


def _run_erf(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _erf_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="special_erf", category="special-fn",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/07-extern-functions.py",
         make_inputs=lambda: (randn(1 << 20, seed=18), ),
         run=_run_erf, ref=lambda inp: torch.erf(inp[0]), rtol=2e-3, atol=2e-3, shape_note="2^20")


@triton.jit
def _log_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, libdevice.log(tl.load(x_ptr + offs, mask=m)), mask=m)


def _run_log(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _log_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="special_log", category="special-fn",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/log.py",
         make_inputs=lambda: (rand(1 << 20, seed=19) * 10.0 + 0.1, ),
         run=_run_log, ref=lambda inp: torch.log(inp[0]), rtol=2e-3, atol=2e-3, shape_note="2^20 (>0)")


@triton.jit
def _rsqrt_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, libdevice.rsqrt(tl.load(x_ptr + offs, mask=m)), mask=m)


def _run_rsqrt(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _rsqrt_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="special_rsqrt", category="special-fn",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/rsqrt.py",
         make_inputs=lambda: (rand(1 << 20, seed=20) * 10.0 + 0.1, ),
         run=_run_rsqrt, ref=lambda inp: torch.rsqrt(inp[0]), rtol=2e-3, atol=2e-3, shape_note="2^20 (>0)")


@triton.jit
def _expm1_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, libdevice.expm1(tl.load(x_ptr + offs, mask=m)), mask=m)


def _run_expm1(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _expm1_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="special_expm1", category="special-fn",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/07-extern-functions.py",
         make_inputs=lambda: (randn(1 << 20, seed=21), ),
         run=_run_expm1, ref=lambda inp: torch.expm1(inp[0]), rtol=2e-3, atol=2e-3, shape_note="2^20")


@triton.jit
def _cos_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, libdevice.cos(tl.load(x_ptr + offs, mask=m)), mask=m)


def _run_cos(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _cos_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="special_cos", category="special-fn",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/07-extern-functions.py",
         make_inputs=lambda: (randn(1 << 20, seed=22), ),
         run=_run_cos, ref=lambda inp: torch.cos(inp[0]), rtol=2e-3, atol=2e-3, shape_note="2^20")


@triton.jit
def _pow_kernel(x_ptr, o_ptr, n, e, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, libdevice.pow(tl.load(x_ptr + offs, mask=m), e), mask=m)


def _run_pow(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _pow_kernel[(triton.cdiv(n, 1024), )](x, o, n, 2.5, BLOCK=1024)
    return o


register(name="special_pow2p5", category="special-fn",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/pow.py",
         make_inputs=lambda: (rand(1 << 20, seed=23) * 4.0 + 0.1, ),
         run=_run_pow, ref=lambda inp: torch.pow(inp[0], 2.5), rtol=2e-3, atol=2e-3, shape_note="2^20 (>0)")


# ######################################################################################### #
# ## GROUP 6 — SOFTMAX FAMILY (per-row, numerically stable)                                ## #
# ## source: https://github.com/triton-lang/triton (python/tutorials/02-fused-softmax.py) ## #
# ##         https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{softmax,log_softmax})## #
# ######################################################################################### #
@triton.jit
def _softmax_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(o_ptr + row * N + cols, e / tl.sum(e, axis=0), mask=m)


def _run_softmax(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _softmax_kernel[(M, )](x, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="softmax_row", category="softmax",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py",
         make_inputs=lambda: (randn(1024, 4096, seed=24), ),
         run=_run_softmax, ref=lambda inp: torch.softmax(inp[0], dim=1),
         rtol=2e-3, atol=2e-3, shape_note="1024x4096")


@triton.jit
def _log_softmax_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    lse = tl.log(tl.sum(tl.exp(x), axis=0))
    tl.store(o_ptr + row * N + cols, x - lse, mask=m)


def _run_log_softmax(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _log_softmax_kernel[(M, )](x, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="log_softmax_row", category="softmax",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/log_softmax.py",
         make_inputs=lambda: (randn(1024, 4096, seed=25), ),
         run=_run_log_softmax, ref=lambda inp: torch.log_softmax(inp[0], dim=1),
         rtol=2e-3, atol=2e-3, shape_note="1024x4096")


@triton.jit
def _softmax_bwd_kernel(y_ptr, dy_ptr, dx_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    y = tl.load(y_ptr + row * N + cols, mask=m, other=0.0)
    dy = tl.load(dy_ptr + row * N + cols, mask=m, other=0.0)
    dx = (dy - tl.sum(dy * y, axis=0)) * y
    tl.store(dx_ptr + row * N + cols, dx, mask=m)


def _run_softmax_bwd(inp):
    y, dy = inp
    M, N = y.shape
    dx = torch.empty_like(y)
    _softmax_bwd_kernel[(M, )](y, dy, dx, M, N, BLOCK_N=triton.next_power_of_2(N))
    return dx


def _ref_softmax_bwd(inp):
    y, dy = inp
    return (dy - (dy * y).sum(1, keepdim=True)) * y


register(name="softmax_bwd", category="softmax",
         source="https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/softmax.py",
         make_inputs=lambda: (torch.softmax(randn(1024, 4096, seed=26), 1), randn(1024, 4096, seed=27)),
         run=_run_softmax_bwd, ref=_ref_softmax_bwd, rtol=2e-3, atol=2e-3, shape_note="1024x4096")


# ######################################################################################### #
# ## GROUP 7 — NORMALIZATION                                                               ## #
# ## source: https://github.com/triton-lang/triton (python/tutorials/05-layer-norm.py)    ## #
# ##         https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{groupnorm,...})      ## #
# ##         https://github.com/fla-org/flash-linear-attention (fla/modules/l2norm.py)     ## #
# ######################################################################################### #
@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, o_ptr, M, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(m, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0)
    b = tl.load(b_ptr + cols, mask=m, other=0.0)
    tl.store(o_ptr + row * N + cols, xc * rstd * w + b, mask=m)


def _run_layernorm(inp):
    x, w, b = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _layernorm_kernel[(M, )](x, w, b, o, M, N, 1e-5, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="layernorm_fwd", category="norm",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py",
         make_inputs=lambda: (randn(2048, 2048, seed=28), randn(2048, seed=29), randn(2048, seed=30)),
         run=_run_layernorm,
         ref=lambda inp: torch.nn.functional.layer_norm(inp[0], (inp[0].shape[1], ), inp[1], inp[2], 1e-5),
         rtol=2e-3, atol=2e-3, shape_note="2048x2048")


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, o_ptr, M, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0)
    tl.store(o_ptr + row * N + cols, x * rstd * w, mask=m)


def _run_rmsnorm(inp):
    x, w = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _rmsnorm_kernel[(M, )](x, w, o, M, N, 1e-5, BLOCK_N=triton.next_power_of_2(N))
    return o


def _ref_rmsnorm(inp):
    x, w = inp
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w


register(name="rmsnorm_fwd", category="norm",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/rms_norm/fused_triton.py",
         make_inputs=lambda: (randn(2048, 2048, seed=31), randn(2048, seed=32)),
         run=_run_rmsnorm, ref=_ref_rmsnorm, rtol=2e-3, atol=2e-3, shape_note="2048x2048")


@triton.jit
def _l2norm_kernel(x_ptr, o_ptr, M, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    nrm = tl.sqrt(tl.sum(x * x, axis=0))
    tl.store(o_ptr + row * N + cols, x / tl.maximum(nrm, eps), mask=m)


def _run_l2norm(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _l2norm_kernel[(M, )](x, o, M, N, 1e-12, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="l2norm_row", category="norm",
         source="https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py",
         make_inputs=lambda: (randn(2048, 2048, seed=33), ),
         run=_run_l2norm, ref=lambda inp: torch.nn.functional.normalize(inp[0], dim=1),
         rtol=2e-3, atol=2e-3, shape_note="2048x2048")


@triton.jit
def _groupnorm_kernel(x_ptr, w_ptr, b_ptr, o_ptr, C, HW, G, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # n*G + g
    g = pid % G
    cpg = C // G
    numel = cpg * HW
    base = pid * numel  # groups are contiguous in an [N, C, HW] tensor
    cols = tl.arange(0, BLOCK)
    m = cols < numel
    x = tl.load(x_ptr + base + cols, mask=m, other=0.0)
    mean = tl.sum(x, axis=0) / numel
    xc = tl.where(m, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, axis=0) / numel + eps)
    ch = g * cpg + (cols // HW)  # per-channel affine
    w = tl.load(w_ptr + ch, mask=m, other=0.0)
    b = tl.load(b_ptr + ch, mask=m, other=0.0)
    tl.store(o_ptr + base + cols, xc * rstd * w + b, mask=m)


def _run_groupnorm(inp):
    x, w, b = inp
    N, C, HW = x.shape
    G = 8
    o = torch.empty_like(x)
    _groupnorm_kernel[(N * G, )](x, w, b, o, C, HW, G, 1e-5, BLOCK=triton.next_power_of_2((C // G) * HW))
    return o


register(name="groupnorm_fwd", category="norm",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/groupnorm.py",
         make_inputs=lambda: (randn(16, 64, 128, seed=34), randn(64, seed=35), randn(64, seed=36)),
         run=_run_groupnorm,
         ref=lambda inp: torch.nn.functional.group_norm(inp[0], 8, inp[1], inp[2], 1e-5),
         rtol=2e-3, atol=2e-3, shape_note="16x64x128 G=8")


# ######################################################################################### #
# ## GROUP 8 — REDUCTIONS (per-row [M,N] -> [M])                                           ## #
# ## source: https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{sum,max,argmax,...}) ## #
# ##         https://github.com/pytorch-labs/tritonbench (operators/sum, operators/welford)## #
# ######################################################################################### #
@triton.jit
def _mul_combine(a, b):
    return a * b


# ORD is an optional reduction_ordering (default None = compiler default). It lets a
# determinism-focused caller (e.g. the bitequiv M2 eval) request inner_tree; standalone
# benchmark use leaves it None, so behavior is unchanged. These are 1-D per-row reduces
# (no kept dim), so they are order-invariant to layout and M2 does not fire on them.
@triton.jit
def _rowreduce_kernel(x_ptr, o_ptr, M, N, KIND: tl.constexpr, BLOCK_N: tl.constexpr, ORD: tl.constexpr = None):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    if KIND == 0:  # sum
        r = tl.sum(tl.load(x_ptr + row * N + cols, mask=m, other=0.0), axis=0, reduction_ordering=ORD)
    elif KIND == 1:  # mean
        r = tl.sum(tl.load(x_ptr + row * N + cols, mask=m, other=0.0), axis=0, reduction_ordering=ORD) / N
    elif KIND == 2:  # max
        r = tl.max(tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf")), axis=0)
    elif KIND == 3:  # min
        r = tl.min(tl.load(x_ptr + row * N + cols, mask=m, other=float("inf")), axis=0)
    elif KIND == 4:  # var (population)
        x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
        mean = tl.sum(x, axis=0, reduction_ordering=ORD) / N
        xc = tl.where(m, x - mean, 0.0)
        r = tl.sum(xc * xc, axis=0, reduction_ordering=ORD) / N
    else:  # KIND == 5: logsumexp
        x = tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf"))
        mx = tl.max(x, axis=0)
        r = mx + tl.log(tl.sum(tl.exp(x - mx), axis=0, reduction_ordering=ORD))
    tl.store(o_ptr + row, r)


def _mk_rowreduce(kind):
    def run(inp):
        x, = inp
        M, N = x.shape
        o = torch.empty(M, device=DEVICE, dtype=torch.float32)
        _rowreduce_kernel[(M, )](x, o, M, N, KIND=kind, BLOCK_N=triton.next_power_of_2(N))
        return o
    return run


_REDUCE_SRC = "https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops"
register(name="reduce_sum", category="reduction", source=_REDUCE_SRC + "/sum.py",
         make_inputs=lambda: (randn(2048, 4096, seed=37), ), run=_mk_rowreduce(0),
         ref=lambda inp: inp[0].sum(1), rtol=2e-3, atol=2e-3, shape_note="2048x4096")
register(name="reduce_mean", category="reduction", source=_REDUCE_SRC + "/mean.py",
         make_inputs=lambda: (randn(2048, 4096, seed=38), ), run=_mk_rowreduce(1),
         ref=lambda inp: inp[0].mean(1), rtol=2e-3, atol=2e-3, shape_note="2048x4096")
register(name="reduce_max", category="reduction", source=_REDUCE_SRC + "/max.py",
         make_inputs=lambda: (randn(2048, 4096, seed=39), ), run=_mk_rowreduce(2),
         ref=lambda inp: inp[0].max(1).values, shape_note="2048x4096")
register(name="reduce_min", category="reduction", source=_REDUCE_SRC + "/min.py",
         make_inputs=lambda: (randn(2048, 4096, seed=40), ), run=_mk_rowreduce(3),
         ref=lambda inp: inp[0].min(1).values, shape_note="2048x4096")
register(name="reduce_var", category="reduction", source=_REDUCE_SRC + "/var_mean.py",
         make_inputs=lambda: (randn(2048, 4096, seed=41), ), run=_mk_rowreduce(4),
         ref=lambda inp: inp[0].var(1, unbiased=False), rtol=2e-3, atol=2e-3, shape_note="2048x4096")
register(name="reduce_logsumexp", category="reduction", source=_REDUCE_SRC + "/logsumexp.py",
         make_inputs=lambda: (randn(2048, 4096, seed=42), ), run=_mk_rowreduce(5),
         ref=lambda inp: torch.logsumexp(inp[0], 1), rtol=2e-3, atol=2e-3, shape_note="2048x4096")


@triton.jit
def _argmax_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + cols, mask=cols < N, other=-float("inf"))
    tl.store(o_ptr + row, tl.argmax(x, axis=0))


def _run_argmax(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty(M, device=DEVICE, dtype=torch.int32)
    _argmax_kernel[(M, )](x, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="reduce_argmax", category="reduction", source=_REDUCE_SRC + "/argmax.py",
         make_inputs=lambda: (randn(2048, 4096, seed=43), ),
         run=_run_argmax, ref=lambda inp: inp[0].argmax(1), shape_note="2048x4096")


@triton.jit
def _prod_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + cols, mask=cols < N, other=1.0)
    tl.store(o_ptr + row, tl.reduce(x, 0, _mul_combine))


def _run_prod(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty(M, device=DEVICE, dtype=torch.float32)
    _prod_kernel[(M, )](x, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="reduce_prod", category="reduction", source=_REDUCE_SRC + "/prod.py",
         make_inputs=lambda: (rand(2048, 128, seed=44) * 0.02 + 0.99, ),
         run=_run_prod, ref=lambda inp: inp[0].prod(1), rtol=5e-3, atol=5e-3, shape_note="2048x128 (~1)")


@triton.jit
def _anyall_kernel(x_ptr, o_ptr, M, N, IS_ANY: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    if IS_ANY:  # masked lanes -> 0 (do not trigger any)
        pred = (tl.load(x_ptr + row * N + cols, mask=m, other=0.0) > 0).to(tl.int32)
        tl.store(o_ptr + row, tl.max(pred, axis=0))
    else:  # all: masked lanes -> 1 (do not break all)
        pred = (tl.load(x_ptr + row * N + cols, mask=m, other=1.0) > 0).to(tl.int32)
        tl.store(o_ptr + row, tl.min(pred, axis=0))


def _mk_anyall(is_any):
    def run(inp):
        x, = inp
        M, N = x.shape
        o = torch.empty(M, device=DEVICE, dtype=torch.int32)
        _anyall_kernel[(M, )](x, o, M, N, IS_ANY=is_any, BLOCK_N=triton.next_power_of_2(N))
        return o
    return run


register(name="reduce_any", category="reduction", source=_REDUCE_SRC + "/any.py",
         make_inputs=lambda: (randn(2048, 4096, seed=45), ), run=_mk_anyall(True),
         ref=lambda inp: (inp[0] > 0).any(1), shape_note="2048x4096")
register(name="reduce_all", category="reduction", source=_REDUCE_SRC + "/all.py",
         make_inputs=lambda: (randn(2048, 4096, seed=46), ), run=_mk_anyall(False),
         ref=lambda inp: (inp[0] > 0).all(1), shape_note="2048x4096")


@triton.jit
def _welford_kernel(x_ptr, mean_ptr, var_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(m, x - mean, 0.0)
    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, tl.sum(xc * xc, axis=0) / N)


def _run_welford(inp):
    x, = inp
    M, N = x.shape
    mean = torch.empty(M, device=DEVICE, dtype=torch.float32)
    var = torch.empty(M, device=DEVICE, dtype=torch.float32)
    _welford_kernel[(M, )](x, mean, var, M, N, BLOCK_N=triton.next_power_of_2(N))
    return mean, var


register(name="welford_mean_var", category="reduction",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/welford/triton_welford.py",
         make_inputs=lambda: (randn(2048, 4096, seed=47), ), run=_run_welford,
         ref=lambda inp: (inp[0].mean(1), inp[0].var(1, unbiased=False)), rtol=2e-3, atol=2e-3,
         shape_note="2048x4096")


# ######################################################################################### #
# ## GROUP 9 — SCAN (per-row prefix)                                                       ## #
# ## source: https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{cumsum,cummax,...})  ## #
# ##         https://github.com/fla-org/flash-linear-attention (fla/ops/utils/cumsum.py)   ## #
# ######################################################################################### #
@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def _cumsum_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + cols)
    tl.store(o_ptr + row * N + cols, tl.cumsum(x, axis=0))


def _run_cumsum(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _cumsum_kernel[(M, )](x, o, M, N, BLOCK_N=N)
    return o


register(name="cumsum_row", category="scan", source=_REDUCE_SRC + "/cumsum.py",
         make_inputs=lambda: (randn(4096, 512, seed=48), ),
         run=_run_cumsum, ref=lambda inp: torch.cumsum(inp[0], 1), rtol=2e-3, atol=2e-3, shape_note="4096x512")


@triton.jit
def _cummax_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + cols)
    tl.store(o_ptr + row * N + cols, tl.associative_scan(x, 0, _max_combine))


def _run_cummax(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _cummax_kernel[(M, )](x, o, M, N, BLOCK_N=N)
    return o


register(name="cummax_row", category="scan", source=_REDUCE_SRC + "/cummax.py",
         make_inputs=lambda: (randn(4096, 512, seed=49), ),
         run=_run_cummax, ref=lambda inp: torch.cummax(inp[0], 1).values, shape_note="4096x512")


@triton.jit
def _cumprod_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + cols)
    tl.store(o_ptr + row * N + cols, tl.associative_scan(x, 0, _mul_combine))


def _run_cumprod(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _cumprod_kernel[(M, )](x, o, M, N, BLOCK_N=N)
    return o


register(name="cumprod_row", category="scan", source=_REDUCE_SRC + "/cumprod.py",
         make_inputs=lambda: (rand(4096, 256, seed=50) * 0.02 + 0.99, ),
         run=_run_cumprod, ref=lambda inp: torch.cumprod(inp[0], 1), rtol=5e-3, atol=5e-3, shape_note="4096x256 (~1)")


# ######################################################################################### #
# ## GROUP 10 — GEMM / MATMUL (tl.dot; grouped-L2 swizzle, K-loop accumulate)              ## #
# ## source: https://github.com/triton-lang/triton (python/tutorials/03-matrix-multiplication.py)## #
# ##         https://github.com/pytorch-labs/tritonbench (operators/gemm, operators/addmm) ## #
# ######################################################################################### #
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr, ACT: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    gsize = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % gsize)
    pid_n = (pid % num_pid_in_group) // gsize
    offs_am = (pid_m * BM + tl.arange(0, BM)) % M
    offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_am[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_bn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        km = offs_k[None, :] < K - k * BK
        a = tl.load(a_ptrs, mask=km, other=0.0)
        kn = offs_k[:, None] < K - k * BK
        b = tl.load(b_ptrs, mask=kn, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    if ACT == 1:  # fused relu epilogue
        acc = tl.where(acc > 0, acc, 0.0)
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
    cm = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=cm)


def _mk_matmul(act, out_dtype):
    def run(inp):
        a, b = inp
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty((M, N), device=DEVICE, dtype=out_dtype)
        grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), )
        _matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                             c.stride(0), c.stride(1), BM=64, BN=64, BK=32, GM=8, ACT=act)
        return c
    return run


register(name="gemm_fp16", category="gemm",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py",
         make_inputs=lambda: (randn(1024, 1024, dtype=torch.float16, seed=51),
                              randn(1024, 1024, dtype=torch.float16, seed=52)),
         run=_mk_matmul(0, torch.float16), ref=lambda inp: inp[0] @ inp[1],
         rtol=1e-2, atol=2e-1, shape_note="1024^3 fp16")
register(name="gemm_bf16", category="gemm",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/gemm",
         make_inputs=lambda: (randn(1024, 1024, dtype=torch.bfloat16, seed=53),
                              randn(1024, 1024, dtype=torch.bfloat16, seed=54)),
         run=_mk_matmul(0, torch.bfloat16), ref=lambda inp: inp[0] @ inp[1],
         rtol=2e-2, atol=5e-1, shape_note="1024^3 bf16")
register(name="gemm_relu", category="gemm",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py",
         make_inputs=lambda: (randn(1024, 1024, dtype=torch.float16, seed=55),
                              randn(1024, 1024, dtype=torch.float16, seed=56)),
         run=_mk_matmul(1, torch.float16), ref=lambda inp: torch.relu(inp[0] @ inp[1]),
         rtol=1e-2, atol=2e-1, shape_note="1024^3 fp16 +relu")


@triton.jit
def _addmm_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    gsize = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % gsize)
    pid_n = (pid % num_pid_in_group) // gsize
    offs_am = (pid_m * BM + tl.arange(0, BM)) % M
    offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_am[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_bn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    acc = acc + bias[None, :]
    c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
    cm = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=cm)


def _run_addmm(inp):
    a, b, bias = inp
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), )
    _addmm_kernel[grid](a, b, bias, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), BM=64, BN=64, BK=32, GM=8)
    return c


register(name="addmm_bias", category="gemm",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/addmm",
         make_inputs=lambda: (randn(1024, 1024, dtype=torch.float16, seed=57),
                              randn(1024, 1024, dtype=torch.float16, seed=58),
                              randn(1024, dtype=torch.float16, seed=59)),
         run=_run_addmm, ref=lambda inp: inp[2] + inp[0] @ inp[1],
         rtol=1e-2, atol=2e-1, shape_note="1024^3 fp16 +bias")


@triton.jit
def _bmm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    bid = tl.program_id(1)
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_am = pid_m * BM + tl.arange(0, BM)
    offs_bn = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + bid * sab + offs_am[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + bid * sbb + offs_k[:, None] * sbk + offs_bn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BK), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BK) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + bid * scb + offs_am[:, None] * scm + offs_bn[None, :] * scn
    cm = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=cm)


def _run_bmm(inp):
    a, b = inp
    B, M, K = a.shape
    N = b.shape[2]
    c = torch.empty((B, M, N), device=DEVICE, dtype=torch.float16)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), B)
    _bmm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1),
                      b.stride(2), c.stride(0), c.stride(1), c.stride(2), BM=64, BN=64, BK=32)
    return c


register(name="bmm", category="gemm",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/gemm",
         make_inputs=lambda: (randn(16, 512, 512, dtype=torch.float16, seed=60),
                              randn(16, 512, 512, dtype=torch.float16, seed=61)),
         run=_run_bmm, ref=lambda inp: torch.bmm(inp[0], inp[1]),
         rtol=1e-2, atol=2e-1, shape_note="16x512x512 fp16")


# ######################################################################################### #
# ## GROUP 11 — ATTENTION (flash-attention v2 forward; online softmax + 2 dots)            ## #
# ## source: https://github.com/triton-lang/triton (python/tutorials/06-fused-attention.py)## #
# ##         https://github.com/pytorch-labs/tritonbench (operators/flash_attention)       ## #
# ######################################################################################### #
@triton.jit
def _attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, scale, S, D: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)
    base = bh * S * D
    offs_m = m_block * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + base + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < S, other=0.0)
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for start_n in range(0, S, BN):
        offs_n = start_n + tl.arange(0, BN)
        k = tl.load(k_ptr + base + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < S, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(offs_n[None, :] < S, qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + base + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < S, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * D + offs_d[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=offs_m[:, None] < S)


def _mk_attn(causal):
    def run(inp):
        q, k, v = inp
        B, H, S, D = q.shape
        o = torch.empty_like(q)
        qf, kf, vf, of = (t.reshape(B * H, S, D) for t in (q, k, v, o))
        grid = (triton.cdiv(S, 64), B * H)
        _attn_fwd_kernel[grid](qf, kf, vf, of, 1.0 / (D ** 0.5), S, D=D, BM=64, BN=64, CAUSAL=causal)
        return o
    return run


def _mk_attn_ref(causal):
    def ref(inp):
        q, k, v = inp
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return ref


register(name="flash_attention_fwd", category="attention",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py",
         make_inputs=lambda: (randn(4, 8, 1024, 64, dtype=torch.float16, seed=62),
                              randn(4, 8, 1024, 64, dtype=torch.float16, seed=63),
                              randn(4, 8, 1024, 64, dtype=torch.float16, seed=64)),
         run=_mk_attn(False), ref=_mk_attn_ref(False), rtol=1e-2, atol=2e-2, shape_note="4x8x1024x64")
register(name="flash_attention_causal", category="attention",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py",
         make_inputs=lambda: (randn(4, 8, 1024, 64, dtype=torch.float16, seed=65),
                              randn(4, 8, 1024, 64, dtype=torch.float16, seed=66),
                              randn(4, 8, 1024, 64, dtype=torch.float16, seed=67)),
         run=_mk_attn(True), ref=_mk_attn_ref(True), rtol=1e-2, atol=2e-2, shape_note="4x8x1024x64 causal")


# ######################################################################################### #
# ## GROUP 12 — LOSSES (per-row)                                                           ## #
# ## source: https://github.com/pytorch-labs/tritonbench (operators/cross_entropy, kl_div)## #
# ##         https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{mse_loss,...})       ## #
# ######################################################################################### #
@triton.jit
def _cross_entropy_kernel(x_ptr, tgt_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf"))
    mx = tl.max(x, axis=0)
    lse = mx + tl.log(tl.sum(tl.exp(x - mx), axis=0))
    t = tl.load(tgt_ptr + row)
    lt = tl.load(x_ptr + row * N + t)
    tl.store(o_ptr + row, lse - lt)


def _run_cross_entropy(inp):
    x, tgt = inp
    M, N = x.shape
    o = torch.empty(M, device=DEVICE, dtype=torch.float32)
    _cross_entropy_kernel[(M, )](x, tgt, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="cross_entropy", category="loss",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/cross_entropy",
         make_inputs=lambda: (randn(4096, 4096, seed=68), randint(4096, (4096, ), seed=69)),
         run=_run_cross_entropy,
         ref=lambda inp: torch.nn.functional.cross_entropy(inp[0], inp[1].long(), reduction="none"),
         rtol=2e-3, atol=2e-3, shape_note="4096x4096")


@triton.jit
def _kl_div_kernel(logp_ptr, tgt_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    logp = tl.load(logp_ptr + row * N + cols, mask=m, other=0.0)
    t = tl.load(tgt_ptr + row * N + cols, mask=m, other=0.0)
    term = tl.where(t > 0, t * (tl.log(t) - logp), 0.0)
    tl.store(o_ptr + row, tl.sum(term, axis=0))


def _run_kl_div(inp):
    logp, t = inp
    M, N = logp.shape
    o = torch.empty(M, device=DEVICE, dtype=torch.float32)
    _kl_div_kernel[(M, )](logp, t, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="kl_div", category="loss",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/kl_div",
         make_inputs=lambda: (torch.log_softmax(randn(4096, 2048, seed=70), 1),
                              torch.softmax(randn(4096, 2048, seed=71), 1)),
         run=_run_kl_div,
         ref=lambda inp: torch.nn.functional.kl_div(inp[0], inp[1], reduction="none").sum(1),
         rtol=2e-3, atol=2e-3, shape_note="4096x2048")


@triton.jit
def _mse_kernel(x_ptr, y_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    d = tl.load(x_ptr + row * N + cols, mask=m, other=0.0) - tl.load(y_ptr + row * N + cols, mask=m, other=0.0)
    tl.store(o_ptr + row, tl.sum(d * d, axis=0) / N)


def _run_mse(inp):
    x, y = inp
    M, N = x.shape
    o = torch.empty(M, device=DEVICE, dtype=torch.float32)
    _mse_kernel[(M, )](x, y, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="mse_loss_row", category="loss",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/mse_loss.py",
         make_inputs=lambda: (randn(4096, 2048, seed=72), randn(4096, 2048, seed=73)),
         run=_run_mse, ref=lambda inp: ((inp[0] - inp[1]) ** 2).mean(1), rtol=2e-3, atol=2e-3, shape_note="4096x2048")


# ######################################################################################### #
# ## GROUP 13 — POSITIONAL (rotary embedding, rotate-half)                                 ## #
# ## source: https://github.com/fla-org/flash-linear-attention (fla/modules/rotary.py)    ## #
# ######################################################################################### #
@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, o_ptr, M, D, BLOCK_D2: tl.constexpr):
    row = tl.program_id(0)
    d2 = D // 2
    j = tl.arange(0, BLOCK_D2)
    m = j < d2
    x1 = tl.load(x_ptr + row * D + j, mask=m, other=0.0)
    x2 = tl.load(x_ptr + row * D + d2 + j, mask=m, other=0.0)
    c = tl.load(cos_ptr + row * d2 + j, mask=m, other=0.0)
    s = tl.load(sin_ptr + row * d2 + j, mask=m, other=0.0)
    tl.store(o_ptr + row * D + j, x1 * c - x2 * s, mask=m)
    tl.store(o_ptr + row * D + d2 + j, x2 * c + x1 * s, mask=m)


def _run_rope(inp):
    x, cos, sin = inp
    M, D = x.shape
    o = torch.empty_like(x)
    _rope_kernel[(M, )](x, cos, sin, o, M, D, BLOCK_D2=triton.next_power_of_2(D // 2))
    return o


def _ref_rope(inp):
    x, cos, sin = inp
    d2 = x.shape[1] // 2
    x1, x2 = x[:, :d2], x[:, d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=1)


register(name="rope_rotate_half", category="positional",
         source="https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/rotary.py",
         make_inputs=lambda: (randn(8192, 128, seed=74), randn(8192, 64, seed=75), randn(8192, 64, seed=76)),
         run=_run_rope, ref=_ref_rope, rtol=2e-3, atol=2e-3, shape_note="8192x128")


# ######################################################################################### #
# ## GROUP 14 — EMBEDDING / GATHER / SCATTER                                               ## #
# ## source: https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{embedding,gather,...})## #
# ######################################################################################### #
@triton.jit
def _embedding_kernel(tbl_ptr, idx_ptr, o_ptr, D, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    t = tl.load(idx_ptr + row)
    d = tl.arange(0, BLOCK_D)
    m = d < D
    tl.store(o_ptr + row * D + d, tl.load(tbl_ptr + t * D + d, mask=m, other=0.0), mask=m)


def _run_embedding(inp):
    tbl, idx = inp
    M = idx.shape[0]
    D = tbl.shape[1]
    o = torch.empty((M, D), device=DEVICE, dtype=tbl.dtype)
    _embedding_kernel[(M, )](tbl, idx, o, D, BLOCK_D=triton.next_power_of_2(D))
    return o


register(name="embedding", category="embedding",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/embedding.py",
         make_inputs=lambda: (randn(50000, 256, seed=77), randint(50000, (8192, ), seed=78)),
         run=_run_embedding, ref=lambda inp: inp[0][inp[1].long()], shape_note="V=50k D=256 M=8192")


@triton.jit
def _gather1d_kernel(src_ptr, idx_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    idx = tl.load(idx_ptr + offs, mask=m, other=0)
    tl.store(o_ptr + offs, tl.load(src_ptr + idx, mask=m, other=0.0), mask=m)


def _run_gather1d(inp):
    src, idx = inp
    n = idx.shape[0]
    o = torch.empty(n, device=DEVICE, dtype=src.dtype)
    _gather1d_kernel[(triton.cdiv(n, 1024), )](src, idx, o, n, BLOCK=1024)
    return o


register(name="gather_1d", category="embedding",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/gather.py",
         make_inputs=lambda: (randn(1 << 20, seed=79), randint(1 << 20, (1 << 20, ), seed=80)),
         run=_run_gather1d, ref=lambda inp: inp[0][inp[1].long()], shape_note="2^20")


@triton.jit
def _index_add_kernel(out_ptr, idx_ptr, src_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    idx = tl.load(idx_ptr + offs, mask=m, other=0)
    val = tl.load(src_ptr + offs, mask=m, other=0.0)
    tl.atomic_add(out_ptr + idx, val, mask=m)


def _run_index_add(inp):
    idx, src, v = inp
    out = torch.zeros(v, device=DEVICE, dtype=torch.float32)
    n = idx.shape[0]
    _index_add_kernel[(triton.cdiv(n, 1024), )](out, idx, src, n, BLOCK=1024)
    return out


def _ref_index_add(inp):
    idx, src, v = inp
    return torch.zeros(v, device=DEVICE, dtype=torch.float32).index_add_(0, idx.long(), src)


register(name="scatter_index_add", category="embedding",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/scatter.py",
         make_inputs=lambda: (randint(4096, (1 << 20, ), seed=81), randn(1 << 20, seed=82), 4096),
         run=_run_index_add, ref=_ref_index_add, rtol=1e-2, atol=1e-2, shape_note="2^20 -> 4096 (atomic)")


# ######################################################################################### #
# ## GROUP 15 — POOLING (2x2 stride-2)                                                     ## #
# ## source: https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{avg_pool2d,max_pool2d})## #
# ######################################################################################### #
@triton.jit
def _avgpool2d_kernel(x_ptr, o_ptr, H, W, OH, OW, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)  # nc*OH + oh
    nc = pid // OH
    oh = pid % OH
    ow = tl.arange(0, BLOCK_OW)
    m = ow < OW
    base = nc * H * W
    a = tl.load(x_ptr + base + (2 * oh) * W + 2 * ow, mask=m, other=0.0)
    b = tl.load(x_ptr + base + (2 * oh) * W + 2 * ow + 1, mask=m, other=0.0)
    c = tl.load(x_ptr + base + (2 * oh + 1) * W + 2 * ow, mask=m, other=0.0)
    d = tl.load(x_ptr + base + (2 * oh + 1) * W + 2 * ow + 1, mask=m, other=0.0)
    tl.store(o_ptr + nc * OH * OW + oh * OW + ow, (a + b + c + d) * 0.25, mask=m)


@triton.jit
def _maxpool2d_kernel(x_ptr, o_ptr, H, W, OH, OW, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    nc = pid // OH
    oh = pid % OH
    ow = tl.arange(0, BLOCK_OW)
    m = ow < OW
    base = nc * H * W
    a = tl.load(x_ptr + base + (2 * oh) * W + 2 * ow, mask=m, other=-float("inf"))
    b = tl.load(x_ptr + base + (2 * oh) * W + 2 * ow + 1, mask=m, other=-float("inf"))
    c = tl.load(x_ptr + base + (2 * oh + 1) * W + 2 * ow, mask=m, other=-float("inf"))
    d = tl.load(x_ptr + base + (2 * oh + 1) * W + 2 * ow + 1, mask=m, other=-float("inf"))
    tl.store(o_ptr + nc * OH * OW + oh * OW + ow, tl.maximum(tl.maximum(a, b), tl.maximum(c, d)), mask=m)


def _mk_pool(kern):
    def run(inp):
        x, = inp
        N, C, H, W = x.shape
        nc, oh, ow = N * C, H // 2, W // 2
        o = torch.empty((nc, oh, ow), device=DEVICE, dtype=x.dtype)
        kern[(nc * oh, )](x.reshape(nc, H, W), o, H, W, oh, ow, BLOCK_OW=triton.next_power_of_2(ow))
        return o.reshape(N, C, oh, ow)
    return run


register(name="avg_pool2d", category="pooling",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/avg_pool2d.py",
         make_inputs=lambda: (randn(16, 32, 128, 128, seed=83), ),
         run=_mk_pool(_avgpool2d_kernel),
         ref=lambda inp: torch.nn.functional.avg_pool2d(inp[0], 2), rtol=2e-3, atol=2e-3, shape_note="16x32x128^2")
register(name="max_pool2d", category="pooling",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/max_pool2d_with_indices.py",
         make_inputs=lambda: (randn(16, 32, 128, 128, seed=84), ),
         run=_mk_pool(_maxpool2d_kernel),
         ref=lambda inp: torch.nn.functional.max_pool2d(inp[0], 2), shape_note="16x32x128^2")


# ######################################################################################### #
# ## GROUP 16 — CONVOLUTION (depthwise causal conv1d)                                      ## #
# ## source: https://github.com/fla-org/flash-linear-attention (fla/modules/conv)          ## #
# ######################################################################################### #
@triton.jit
def _causal_conv1d_kernel(x_ptr, w_ptr, o_ptr, L, K: tl.constexpr, BLOCK_L: tl.constexpr):
    c = tl.program_id(0)
    offs = tl.arange(0, BLOCK_L)
    m = offs < L
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)
    for k in range(K):
        idx = offs - (K - 1) + k
        xk = tl.load(x_ptr + c * L + idx, mask=m & (idx >= 0), other=0.0)
        acc += xk * tl.load(w_ptr + c * K + k)
    tl.store(o_ptr + c * L + offs, acc, mask=m)


def _run_causal_conv1d(inp):
    x, w = inp
    C, L = x.shape
    K = w.shape[1]
    o = torch.empty_like(x)
    _causal_conv1d_kernel[(C, )](x, w, o, L, K=K, BLOCK_L=triton.next_power_of_2(L))
    return o


def _ref_causal_conv1d(inp):
    x, w = inp
    C, L = x.shape
    K = w.shape[1]
    xp = torch.nn.functional.pad(x, (K - 1, 0))
    out = torch.zeros_like(x)
    for k in range(K):
        out += xp[:, k:k + L] * w[:, k:k + 1]
    return out


register(name="causal_conv1d", category="convolution",
         source="https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/conv.py",
         make_inputs=lambda: (randn(256, 4096, seed=85), randn(256, 4, seed=86)),
         run=_run_causal_conv1d, ref=_ref_causal_conv1d, rtol=2e-3, atol=2e-3, shape_note="C=256 L=4096 K=4")


# ######################################################################################### #
# ## GROUP 17 — DATA MOVEMENT (transpose, flip)                                            ## #
# ## source: https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/{permute_copy,flip})  ## #
# ######################################################################################### #
@triton.jit
def _transpose_kernel(x_ptr, o_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    tl.store(o_ptr + rn[:, None] * M + rm[None, :], tl.trans(x), mask=(rn[:, None] < N) & (rm[None, :] < M))


def _run_transpose(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty((N, M), device=DEVICE, dtype=x.dtype)
    _transpose_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32))](x, o, M, N, BM=32, BN=32)
    return o


register(name="transpose_2d", category="data-movement",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/permute_copy.py",
         make_inputs=lambda: (randn(2048, 1024, seed=87), ),
         run=_run_transpose, ref=lambda inp: inp[0].t().contiguous(), shape_note="2048x1024")


@triton.jit
def _flip1d_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + (n - 1 - offs), mask=m, other=0.0), mask=m)


def _run_flip1d(inp):
    x, = inp
    o = torch.empty_like(x)
    n = x.numel()
    _flip1d_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o


register(name="flip_1d", category="data-movement",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/flip.py",
         make_inputs=lambda: (randn(1 << 20, seed=88), ),
         run=_run_flip1d, ref=lambda inp: torch.flip(inp[0], [0]), shape_note="2^20")


# ######################################################################################### #
# ## GROUP 18 — GEMM STRUCTURAL VARIANTS (autotune+pipeline / split-K / persistent /       ## #
# ##            block-pointers / fp8 tensor-core)                                          ## #
# ## source: https://github.com/triton-lang/triton (tutorials 03/09/12) + tritonbench gemm## #
# ######################################################################################### #
# --- autotuned matmul (multiple configs + num_stages async pipelining) --- #
_GEMM_AUTOTUNE_CFGS = [
    triton.Config({"BM": 128, "BN": 128, "BK": 32, "GM": 8}, num_warps=4, num_stages=3),
    triton.Config({"BM": 128, "BN": 64, "BK": 32, "GM": 8}, num_warps=4, num_stages=4),
    triton.Config({"BM": 64, "BN": 64, "BK": 32, "GM": 8}, num_warps=4, num_stages=3),
    triton.Config({"BM": 128, "BN": 256, "BK": 64, "GM": 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_GEMM_AUTOTUNE_CFGS, key=["M", "N", "K"])
@triton.jit
def _gemm_autotune_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    gsize = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % gsize)
    pid_n = (pid % num_pid_in_group) // gsize
    offs_am = (pid_m * BM + tl.arange(0, BM)) % M
    offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_am[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_bn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
    cm = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=cm)


def _run_gemm_autotune(inp):
    a, b = inp
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    _gemm_autotune_kernel[lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]), )](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


register(name="gemm_autotune", category="gemm-structural",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py",
         make_inputs=lambda: (randn(2048, 2048, dtype=torch.float16, seed=89),
                              randn(2048, 2048, dtype=torch.float16, seed=90)),
         run=_run_gemm_autotune, ref=lambda inp: inp[0] @ inp[1], rtol=1e-2, atol=3e-1, shape_note="2048^3 autotune")


# --- split-K matmul (atomic accumulation across K partitions) --- #
@triton.jit
def _gemm_splitk_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_am = pid_m * BM + tl.arange(0, BM)
    offs_bn = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * BK + tl.arange(0, BK)
    a_ptrs = a_ptr + offs_am[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_bn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK * SPLIT_K)):
        ka = offs_k[None, :] + k * BK * SPLIT_K
        kb = offs_k[:, None] + k * BK * SPLIT_K
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (ka < K), other=0.0)
        b = tl.load(b_ptrs, mask=(kb < K) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * SPLIT_K * sak
        b_ptrs += BK * SPLIT_K * sbk
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
    cm = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, acc, mask=cm)


def _run_gemm_splitk(inp):
    a, b = inp
    M, K = a.shape
    N = b.shape[1]
    c = torch.zeros((M, N), device=DEVICE, dtype=torch.float32)
    BM = BN = 64
    BK = 32
    split_k = 4
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), split_k)
    _gemm_splitk_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, SPLIT_K=split_k)
    return c


register(name="gemm_splitk", category="gemm-structural",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/12-split-k-matmul.py",
         make_inputs=lambda: (randn(512, 512, dtype=torch.float16, seed=91),
                              randn(512, 512, dtype=torch.float16, seed=92)),
         run=_run_gemm_splitk, ref=lambda inp: inp[0].float() @ inp[1].float(),
         rtol=1e-2, atol=5e-1, shape_note="512^3 split-K=4 (atomic)")


# --- persistent matmul (grid = #SMs, loop over output tiles) --- #
@triton.jit
def _gemm_persistent_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn, NUM_SMS,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GM * num_pid_n
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GM
        gsize = min(num_pid_m - first_pid_m, GM)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % gsize)
        pid_n = (tile_id % num_pid_in_group) // gsize
        offs_am = (pid_m * BM + tl.arange(0, BM)) % M
        offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + offs_am[:, None] * sam + offs_k[None, :] * sak
        b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_bn[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        offs_cm = pid_m * BM + tl.arange(0, BM)
        offs_cn = pid_n * BN + tl.arange(0, BN)
        c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
        cm = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=cm)


def _run_gemm_persistent(inp):
    a, b = inp
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    _gemm_persistent_kernel[(num_sms, )](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                         c.stride(0), c.stride(1), num_sms, BM=128, BN=128, BK=32, GM=8)
    return c


register(name="gemm_persistent", category="gemm-structural",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py",
         make_inputs=lambda: (randn(2048, 2048, dtype=torch.float16, seed=93),
                              randn(2048, 2048, dtype=torch.float16, seed=94)),
         run=_run_gemm_persistent, ref=lambda inp: inp[0] @ inp[1], rtol=1e-2, atol=3e-1,
         shape_note="2048^3 persistent")


# --- matmul via block pointers (tl.make_block_ptr / tl.advance) --- #
@triton.jit
def _gemm_block_ptr_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_bp = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(sam, sak), offsets=(pid_m * BM, 0),
                             block_shape=(BM, BK), order=(1, 0))
    b_bp = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(sbk, sbn), offsets=(0, pid_n * BN),
                             block_shape=(BK, BN), order=(1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_bp, boundary_check=(0, 1))
        b = tl.load(b_bp, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_bp = tl.advance(a_bp, (0, BK))
        b_bp = tl.advance(b_bp, (BK, 0))
    c_bp = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(scm, scn), offsets=(pid_m * BM, pid_n * BN),
                             block_shape=(BM, BN), order=(1, 0))
    tl.store(c_bp, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


def _run_gemm_block_ptr(inp):
    a, b = inp
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    _gemm_block_ptr_kernel[(triton.cdiv(M, 128), triton.cdiv(N, 128))](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=128, BN=128, BK=32)
    return c


register(name="gemm_block_ptr", category="gemm-structural",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py",
         make_inputs=lambda: (randn(2048, 2048, dtype=torch.float16, seed=95),
                              randn(2048, 2048, dtype=torch.float16, seed=96)),
         run=_run_gemm_block_ptr, ref=lambda inp: inp[0] @ inp[1], rtol=1e-2, atol=3e-1,
         shape_note="2048^3 block-ptr")


# --- fp8 tensor-core matmul (fp8 e4m3 inputs -> fp32 accumulate) --- #
@triton.jit
def _gemm_fp8_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BM + tl.arange(0, BM)
    offs_bn = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_am[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_bn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BK), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BK) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
    cm = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=cm)


def _run_gemm_fp8(inp):
    a, b = inp
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    _gemm_fp8_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=64, BN=64, BK=64)
    return c


def _make_fp8_gemm_inputs():
    a = (randn(512, 512, seed=97) * 0.5).to(torch.float8_e4m3fn)
    b = (randn(512, 512, seed=98) * 0.5).to(torch.float8_e4m3fn)
    return a, b


register(name="gemm_fp8", category="gemm-structural",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/fp8_gemm",
         make_inputs=_make_fp8_gemm_inputs,
         run=_run_gemm_fp8, ref=lambda inp: inp[0].float() @ inp[1].float(),
         rtol=2e-1, atol=2.0, shape_note="512^3 fp8-e4m3")


# ######################################################################################### #
# ## GROUP 19 — TMA (Tensor Memory Accelerator: tl.make_tensor_descriptor + set_allocator) ## #
# ## source: https://github.com/triton-lang/triton (tutorials 09-persistent / 10-block-scaled)## #
# ######################################################################################### #
def _install_tma_allocator():
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device=DEVICE, dtype=torch.int8))


@triton.jit
def _tma_copy_kernel(x_ptr, o_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    o_desc.store([pid_m * BM, pid_n * BN], x_desc.load([pid_m * BM, pid_n * BN]))


def _run_tma_copy(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _install_tma_allocator()
    _tma_copy_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](x, o, M, N, BM=64, BN=64)
    return o


register(name="tma_copy_2d", category="tma",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py",
         make_inputs=lambda: (randn(2048, 2048, dtype=torch.float16, seed=99), ),
         run=_run_tma_copy, ref=lambda inp: inp[0], shape_note="2048x2048 fp16 TMA")


@triton.jit
def _gemm_tma_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = a_desc.load([pid_m * BM, k])
        b = b_desc.load([k, pid_n * BN])
        acc += tl.dot(a, b)
    c_desc.store([pid_m * BM, pid_n * BN], acc.to(c_ptr.dtype.element_ty))


def _run_gemm_tma(inp):
    a, b = inp
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    _install_tma_allocator()
    _gemm_tma_kernel[(triton.cdiv(M, 128), triton.cdiv(N, 128))](a, b, c, M, N, K, BM=128, BN=128, BK=64)
    return c


register(name="gemm_tma", category="tma",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py",
         make_inputs=lambda: (randn(1024, 1024, dtype=torch.float16, seed=100),
                              randn(1024, 1024, dtype=torch.float16, seed=101)),
         run=_run_gemm_tma, ref=lambda inp: inp[0] @ inp[1], rtol=1e-2, atol=3e-1, shape_note="1024^3 TMA")


# ######################################################################################### #
# ## GROUP 20 — COOPERATIVE / CROSS-CTA + DATA-DEPENDENT CONTROL FLOW                      ## #
# ## source: https://github.com/triton-lang/triton (tutorials 15-multi-cta-layer-norm)    ## #
# ######################################################################################### #
@triton.jit
def _coop_global_sum_kernel(x_ptr, acc_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    partial = tl.sum(tl.load(x_ptr + offs, mask=m, other=0.0), axis=0)
    tl.atomic_add(acc_ptr, partial)  # cross-CTA cooperative accumulation into one scalar


def _run_coop_global_sum(inp):
    x, = inp
    acc = torch.zeros(1, device=DEVICE, dtype=torch.float32)
    n = x.numel()
    _coop_global_sum_kernel[(triton.cdiv(n, 4096), )](x, acc, n, BLOCK=4096)
    return acc


register(name="cooperative_global_sum", category="cooperative",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/15-multi-cta-layer-norm.py",
         make_inputs=lambda: (randn(1 << 20, seed=102), ),
         run=_run_coop_global_sum, ref=lambda inp: inp[0].sum().reshape(1), rtol=1e-2, atol=1e-1,
         shape_note="2^20 -> scalar (atomic)")


@triton.jit
def _data_dependent_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    gate = tl.sum(x, axis=0)  # runtime scalar
    if gate > 0.0:  # data-dependent branch (real CFG in PTX)
        r = tl.sum(x * x, axis=0)
    else:
        r = tl.sum(tl.abs(x), axis=0)
    tl.store(o_ptr + row, r)


def _run_data_dependent(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty(M, device=DEVICE, dtype=torch.float32)
    _data_dependent_kernel[(M, )](x, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


def _ref_data_dependent(inp):
    x, = inp
    gate = x.sum(1)
    return torch.where(gate > 0, (x * x).sum(1), x.abs().sum(1))


register(name="data_dependent_branch", category="control-flow",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials",
         make_inputs=lambda: (randn(4096, 4096, seed=103), ),
         run=_run_data_dependent, ref=_ref_data_dependent, rtol=2e-3, atol=2e-3, shape_note="4096x4096")


# ######################################################################################### #
# ## GROUP 21 — WARP SPECIALIZATION (persistent matmul, warp_specialize producer/consumer) ## #
# ## source: https://github.com/triton-lang/triton (tutorials 09-persistent-matmul)       ## #
# ######################################################################################### #
@triton.jit
def _gemm_ws_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn, NUM_SMS,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GM * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GM
        gsize = min(num_pid_m - first_pid_m, GM)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % gsize)
        pid_n = (tile_id % num_pid_in_group) // gsize
        offs_am = (pid_m * BM + tl.arange(0, BM)) % M
        offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + offs_am[:, None] * sam + offs_k[None, :] * sak
        b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_bn[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        offs_cm = pid_m * BM + tl.arange(0, BM)
        offs_cn = pid_n * BN + tl.arange(0, BN)
        c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
        cm = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=cm)


def _run_gemm_ws(inp):
    a, b = inp
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    _gemm_ws_kernel[(num_sms, )](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                 c.stride(0), c.stride(1), num_sms, BM=128, BN=128, BK=64, GM=8, num_warps=4)
    return c


register(name="gemm_warp_specialized", category="warp-spec",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py",
         make_inputs=lambda: (randn(2048, 2048, dtype=torch.float16, seed=104),
                              randn(2048, 2048, dtype=torch.float16, seed=105)),
         run=_run_gemm_ws, ref=lambda inp: inp[0] @ inp[1], rtol=1e-2, atol=3e-1, shape_note="2048^3 warp-spec")


# ######################################################################################### #
# ## GROUP 22 — BACKWARD (weight-grad / input-grad reductions)                             ## #
# ## source: https://github.com/triton-lang/triton (tutorials/05-layer-norm.py backward)  ## #
# ##         https://github.com/pytorch-labs/tritonbench (operators/{layer_norm,rms_norm}) ## #
# ######################################################################################### #
@triton.jit
def _layernorm_bwd_dx_kernel(dy_ptr, x_ptr, w_ptr, dx_ptr, M, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    dy = tl.load(dy_ptr + row * N + cols, mask=m, other=0.0)
    w = tl.load(w_ptr + cols, mask=m, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(m, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, axis=0) / N + eps)
    xhat = xc * rstd
    dyw = dy * w
    c1 = tl.sum(tl.where(m, dyw * xhat, 0.0), axis=0) / N
    c2 = tl.sum(tl.where(m, dyw, 0.0), axis=0) / N
    tl.store(dx_ptr + row * N + cols, (dyw - xhat * c1 - c2) * rstd, mask=m)


def _run_layernorm_bwd(inp):
    dy, x, w = inp
    M, N = x.shape
    dx = torch.empty_like(x)
    _layernorm_bwd_dx_kernel[(M, )](dy, x, w, dx, M, N, 1e-5, BLOCK_N=triton.next_power_of_2(N))
    return dx


def _ref_layernorm_bwd(inp):
    dy, x, w = inp
    mean = x.mean(1, keepdim=True)
    xc = x - mean
    rstd = torch.rsqrt((xc * xc).mean(1, keepdim=True) + 1e-5)
    xhat = xc * rstd
    dyw = dy * w
    c1 = (dyw * xhat).mean(1, keepdim=True)
    c2 = dyw.mean(1, keepdim=True)
    return (dyw - xhat * c1 - c2) * rstd


register(name="layernorm_bwd_dx", category="backward",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py",
         make_inputs=lambda: (randn(2048, 2048, seed=106), randn(2048, 2048, seed=107), randn(2048, seed=108)),
         run=_run_layernorm_bwd, ref=_ref_layernorm_bwd, rtol=2e-3, atol=2e-3, shape_note="2048x2048")


@triton.jit
def _rmsnorm_bwd_dx_kernel(dy_ptr, x_ptr, w_ptr, dx_ptr, M, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    dy = tl.load(dy_ptr + row * N + cols, mask=m, other=0.0)
    w = tl.load(w_ptr + cols, mask=m, other=0.0)
    r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    dyw = dy * w
    s = tl.sum(tl.where(m, dyw * x, 0.0), axis=0) / N
    tl.store(dx_ptr + row * N + cols, r * dyw - x * (r * r * r) * s, mask=m)


def _run_rmsnorm_bwd(inp):
    dy, x, w = inp
    M, N = x.shape
    dx = torch.empty_like(x)
    _rmsnorm_bwd_dx_kernel[(M, )](dy, x, w, dx, M, N, 1e-5, BLOCK_N=triton.next_power_of_2(N))
    return dx


def _ref_rmsnorm_bwd(inp):
    dy, x, w = inp
    r = torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-5)
    dyw = dy * w
    s = (dyw * x).mean(1, keepdim=True)
    return r * dyw - x * (r ** 3) * s


register(name="rmsnorm_bwd_dx", category="backward",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/rms_norm/fused_triton.py",
         make_inputs=lambda: (randn(2048, 2048, seed=109), randn(2048, 2048, seed=110), randn(2048, seed=111)),
         run=_run_rmsnorm_bwd, ref=_ref_rmsnorm_bwd, rtol=2e-3, atol=2e-3, shape_note="2048x2048")


# Final stage of LayerNorm/RMSNorm backward: sum the per-row-block partial buffers over the ROW
# axis (dim 0, strided by N) to get dw[N] and db[N]. This reduces a NON-contiguous axis with the
# kept feature axis contiguous -- the M2 firing shape (the weight-grad companion to the dx kernels
# above; the zoo had input-grad reductions but not this weight-grad one). ORD defaults to None so
# the standalone benchmark is unchanged; the bitequiv M2 eval passes inner_tree.
@triton.jit
def _norm_bwd_dwdb_kernel(dw_partial_ptr, db_partial_ptr, dw_ptr, db_ptr, M, N,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ORD: tl.constexpr = None):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]
    dw = tl.sum(tl.load(dw_partial_ptr + offs, mask=mask, other=0.0), axis=0, reduction_ordering=ORD)
    db = tl.sum(tl.load(db_partial_ptr + offs, mask=mask, other=0.0), axis=0, reduction_ordering=ORD)
    cmask = cols < N
    tl.store(dw_ptr + cols, dw, mask=cmask)
    tl.store(db_ptr + cols, db, mask=cmask)


def _run_norm_bwd_dwdb(inp):
    dwp, dbp = inp
    M, N = dwp.shape
    dw = torch.empty(N, device=DEVICE, dtype=torch.float32)
    db = torch.empty(N, device=DEVICE, dtype=torch.float32)
    BLOCK_N = 256
    _norm_bwd_dwdb_kernel[(triton.cdiv(N, BLOCK_N), )](dwp, dbp, dw, db, M, N,
                                                       BLOCK_M=triton.next_power_of_2(M), BLOCK_N=BLOCK_N)
    return torch.cat([dw, db])


def _ref_norm_bwd_dwdb(inp):
    dwp, dbp = inp
    return torch.cat([dwp.sum(0), dbp.sum(0)])


register(name="layernorm_bwd_dwdb", category="backward",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py",
         make_inputs=lambda: (randn(64, 2048, seed=112), randn(64, 2048, seed=113)),
         run=_run_norm_bwd_dwdb, ref=_ref_norm_bwd_dwdb, rtol=2e-3, atol=2e-3, shape_note="64x2048 partials")


@triton.jit
def _gemm_dx_kernel(dy_ptr, b_ptr, dx_ptr, M, K, N, sdym, sdyn, sbk, sbn, sdxm, sdxk,
                    BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_n = tl.arange(0, BN)
    dy_ptrs = dy_ptr + offs_m[:, None] * sdym + offs_n[None, :] * sdyn
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BK), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BN)):
        nmask = offs_n[None, :] < N - n * BN
        dyt = tl.load(dy_ptrs, mask=(offs_m[:, None] < M) & nmask, other=0.0)
        bt = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & nmask, other=0.0)
        acc += tl.dot(dyt, tl.trans(bt))
        dy_ptrs += BN * sdyn
        b_ptrs += BN * sbn
    offs_dm = pid_m * BM + tl.arange(0, BM)
    offs_dk = pid_k * BK + tl.arange(0, BK)
    dx_ptrs = dx_ptr + offs_dm[:, None] * sdxm + offs_dk[None, :] * sdxk
    tl.store(dx_ptrs, acc.to(dx_ptr.dtype.element_ty), mask=(offs_dm[:, None] < M) & (offs_dk[None, :] < K))


def _run_gemm_dx(inp):
    dy, b = inp  # dy [M,N], b [K,N] -> dx [M,K] = dy @ b^T
    M, N = dy.shape
    K = b.shape[0]
    dx = torch.empty((M, K), device=DEVICE, dtype=torch.float16)
    _gemm_dx_kernel[(triton.cdiv(M, 64), triton.cdiv(K, 64))](
        dy, b, dx, M, K, N, dy.stride(0), dy.stride(1), b.stride(0), b.stride(1), dx.stride(0), dx.stride(1),
        BM=64, BK=64, BN=32)
    return dx


register(name="gemm_bwd_dx", category="backward",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py",
         make_inputs=lambda: (randn(1024, 1024, dtype=torch.float16, seed=112),
                              randn(1024, 1024, dtype=torch.float16, seed=113)),
         run=_run_gemm_dx, ref=lambda inp: inp[0] @ inp[1].t(), rtol=1e-2, atol=2e-1, shape_note="dy@bT 1024^3")


# ######################################################################################### #
# ## GROUP 23 — QUANTIZATION (dynamic scale + quantize/dequantize round-trip)              ## #
# ## source: https://github.com/FlagOpen/FlagGems (ops/per_token_group_quant_fp8.py)      ## #
# ##         https://github.com/pytorch/ao (torchao/kernel, prototype quant kernels)       ## #
# ######################################################################################### #
@triton.jit
def _int8_qdq_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    scale = tl.max(tl.abs(x), axis=0) / 127.0 + 1e-12
    q = tl.floor(x / scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(o_ptr + row * N + cols, q * scale, mask=m)


def _run_int8_qdq(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty_like(x)
    _int8_qdq_kernel[(M, )](x, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


def _ref_int8_qdq(inp):
    x, = inp
    scale = x.abs().amax(1, keepdim=True) / 127.0 + 1e-12
    q = torch.floor(x / scale + 0.5).clamp(-127.0, 127.0)
    return q * scale


# Correctness = "dequant recovers x within one quant step" (bit-exact vs a torch round-trip is
# boundary-sensitive: a few elements flip a level at .5 boundaries). Quant error <= scale/2.
register(name="quant_int8_dynamic", category="quant",
         source="https://github.com/pytorch/ao/blob/main/torchao/kernel/intmm_triton.py",
         make_inputs=lambda: (randn(4096, 4096, seed=114), ),
         run=_run_int8_qdq, ref=lambda inp: inp[0], rtol=0.0, atol=0.06,
         shape_note="4096x4096 per-row int8")


@triton.jit
def _fp8_quant_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m, other=0.0).to(o_ptr.dtype.element_ty), mask=m)


def _run_fp8_quant(inp):
    x, = inp
    o = torch.empty(x.shape, device=DEVICE, dtype=torch.float8_e4m3fn)
    n = x.numel()
    _fp8_quant_kernel[(triton.cdiv(n, 1024), )](x, o, n, BLOCK=1024)
    return o.float()


register(name="quant_fp8_e4m3", category="quant",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/per_token_group_quant_fp8.py",
         make_inputs=lambda: (randn(1 << 20, seed=115), ),
         run=_run_fp8_quant, ref=lambda inp: inp[0].to(torch.float8_e4m3fn).float(),
         rtol=1e-4, atol=1e-4, shape_note="2^20 fp8 cast")


# ######################################################################################### #
# ## GROUP 24 — SSM / linear recurrence, batchnorm stats, count                            ## #
# ## source: https://github.com/fla-org/flash-linear-attention (fla/ops)                   ## #
# ##         https://github.com/FlagOpen/FlagGems (ops/{batch_norm,count_nonzero}.py)      ## #
# ######################################################################################### #
@triton.jit
def _ssm_combine(a1, b1, a2, b2):
    return a1 * a2, a2 * b1 + b2


@triton.jit
def _ssm_scan_kernel(g_ptr, x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    g = tl.load(g_ptr + row * N + cols)
    x = tl.load(x_ptr + row * N + cols)
    _, h = tl.associative_scan((g, x), 0, _ssm_combine)  # h_t = g_t*h_{t-1} + x_t
    tl.store(o_ptr + row * N + cols, h)


def _run_ssm_scan(inp):
    g, x = inp
    M, N = g.shape
    o = torch.empty_like(g)
    _ssm_scan_kernel[(M, )](g, x, o, M, N, BLOCK_N=N)
    return o


def _ref_ssm_scan(inp):
    g, x = inp
    h = torch.zeros(g.shape[0], device=DEVICE, dtype=torch.float32)
    outs = []
    for t in range(g.shape[1]):
        h = g[:, t] * h + x[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


register(name="ssm_linear_recurrence", category="ssm",
         source="https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule",
         make_inputs=lambda: (rand(2048, 512, seed=116), randn(2048, 512, seed=117)),
         run=_run_ssm_scan, ref=_ref_ssm_scan, rtol=2e-3, atol=2e-3, shape_note="2048x512 h=g*h+x")


@triton.jit
def _bn_stats_kernel(x_ptr, mean_ptr, var_ptr, N, C, HW, BLOCK_HW: tl.constexpr):
    c = tl.program_id(0)
    hw = tl.arange(0, BLOCK_HW)
    m = hw < HW
    s = tl.zeros([], dtype=tl.float32)
    ss = tl.zeros([], dtype=tl.float32)
    for n in range(N):
        x = tl.load(x_ptr + n * C * HW + c * HW + hw, mask=m, other=0.0)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)
    cnt = N * HW
    mean = s / cnt
    tl.store(mean_ptr + c, mean)
    tl.store(var_ptr + c, ss / cnt - mean * mean)


def _run_bn_stats(inp):
    x, = inp
    N, C, HW = x.shape
    mean = torch.empty(C, device=DEVICE, dtype=torch.float32)
    var = torch.empty(C, device=DEVICE, dtype=torch.float32)
    _bn_stats_kernel[(C, )](x, mean, var, N, C, HW, BLOCK_HW=triton.next_power_of_2(HW))
    return mean, var


register(name="batchnorm_stats", category="norm",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/batch_norm.py",
         make_inputs=lambda: (randn(64, 128, 1024, seed=118), ),
         run=_run_bn_stats,
         ref=lambda inp: (inp[0].mean((0, 2)), inp[0].var((0, 2), unbiased=False)),
         rtol=2e-3, atol=2e-3, shape_note="N=64 C=128 HW=1024")


@triton.jit
def _count_nonzero_kernel(x_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    tl.store(o_ptr + row, tl.sum((x != 0).to(tl.int32), axis=0))


def _run_count_nonzero(inp):
    x, = inp
    M, N = x.shape
    o = torch.empty(M, device=DEVICE, dtype=torch.int32)
    _count_nonzero_kernel[(M, )](x, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="count_nonzero", category="reduction",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/count_nonzero.py",
         make_inputs=lambda: (randn(4096, 4096, seed=119) * (rand(4096, 4096, seed=120) > 0.5).float(), ),
         run=_run_count_nonzero, ref=lambda inp: (inp[0] != 0).sum(1), shape_note="4096x4096 (~50% zero)")


# ######################################################################################### #
# ## GROUP 25 — MoE / GROUPED GEMM (flat tokens + per-expert weights)                      ## #
# ## source: https://github.com/triton-lang/triton (tutorials/08-grouped-gemm.py)         ## #
# ######################################################################################### #
@triton.jit
def _grouped_gemm_kernel(a_ptr, b_ptr, c_ptr, G, M, N, K,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    g = tl.program_id(0)
    pid = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + g * M * K + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + g * K * N + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BK), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BK) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = c_ptr + g * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _run_grouped_gemm(inp):
    a, b = inp  # a [G*M, K], b [G, K, N]
    G, K, N = b.shape
    M = a.shape[0] // G
    c = torch.empty((G * M, N), device=DEVICE, dtype=torch.float16)
    grid = (G, triton.cdiv(M, 64) * triton.cdiv(N, 64))
    _grouped_gemm_kernel[grid](a, b, c, G, M, N, K, BM=64, BN=64, BK=32)
    return c


register(name="moe_grouped_gemm", category="moe",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/08-grouped-gemm.py",
         make_inputs=lambda: (randn(8 * 256, 512, dtype=torch.float16, seed=121),
                              randn(8, 512, 512, dtype=torch.float16, seed=122)),
         run=_run_grouped_gemm,
         ref=lambda inp: (inp[0].view(8, 256, 512) @ inp[1]).reshape(8 * 256, 512),
         rtol=1e-2, atol=3e-1, shape_note="G=8 M=256 K=512 N=512")


# ######################################################################################### #
# ## GROUP 26 — DEPTHWISE CONV2D (direct, per-channel R x S)                               ## #
# ## source: https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/conv_depthwise2d.py)  ## #
# ######################################################################################### #
@triton.jit
def _dwconv2d_kernel(x_ptr, w_ptr, o_ptr, C, H, W, OH, OW, R: tl.constexpr, S: tl.constexpr,
                     BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)  # c*OH + oh
    c = pid // OH
    oh = pid % OH
    ow = tl.arange(0, BLOCK_OW)
    m = ow < OW
    acc = tl.zeros([BLOCK_OW], dtype=tl.float32)
    for r in range(R):
        for s in range(S):
            xv = tl.load(x_ptr + c * H * W + (oh + r) * W + (ow + s), mask=m, other=0.0)
            acc += xv * tl.load(w_ptr + c * R * S + r * S + s)
    tl.store(o_ptr + c * OH * OW + oh * OW + ow, acc, mask=m)


def _run_dwconv2d(inp):
    x, w = inp  # x [C,H,W], w [C,R,S]
    C, H, W = x.shape
    R, S = w.shape[1], w.shape[2]
    oh, ow = H - R + 1, W - S + 1
    o = torch.empty((C, oh, ow), device=DEVICE, dtype=torch.float32)
    _dwconv2d_kernel[(C * oh, )](x, w, o, C, H, W, oh, ow, R=R, S=S, BLOCK_OW=triton.next_power_of_2(ow))
    return o


def _ref_dwconv2d(inp):
    x, w = inp
    C = x.shape[0]
    return torch.nn.functional.conv2d(x.unsqueeze(0), w.unsqueeze(1), groups=C).squeeze(0)


register(name="depthwise_conv2d", category="convolution",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/conv_depthwise2d.py",
         make_inputs=lambda: (randn(64, 66, 66, seed=123), randn(64, 3, 3, seed=124)),
         run=_run_dwconv2d, ref=_ref_dwconv2d, rtol=2e-3, atol=2e-3, shape_note="C=64 64x64 3x3")


# ######################################################################################### #
# ## GROUP 27 — MORE LOSSES (nll, jsd)                                                     ## #
# ## source: https://github.com/pytorch-labs/tritonbench (operators/jsd) + FlagGems       ## #
# ######################################################################################### #
@triton.jit
def _nll_kernel(logp_ptr, tgt_ptr, o_ptr, N):
    row = tl.program_id(0)
    t = tl.load(tgt_ptr + row)
    tl.store(o_ptr + row, -tl.load(logp_ptr + row * N + t))


def _run_nll(inp):
    logp, tgt = inp
    M, N = logp.shape
    o = torch.empty(M, device=DEVICE, dtype=torch.float32)
    _nll_kernel[(M, )](logp, tgt, o, N)
    return o


register(name="nll_loss", category="loss",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/nllloss.py",
         make_inputs=lambda: (torch.log_softmax(randn(8192, 4096, seed=125), 1), randint(4096, (8192, ), seed=126)),
         run=_run_nll,
         ref=lambda inp: torch.nn.functional.nll_loss(inp[0], inp[1].long(), reduction="none"),
         rtol=2e-3, atol=2e-3, shape_note="8192x4096")


@triton.jit
def _jsd_kernel(logp_ptr, logq_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    lp = tl.load(logp_ptr + row * N + cols)
    lq = tl.load(logq_ptr + row * N + cols)
    p = tl.exp(lp)
    q = tl.exp(lq)
    lm = tl.log(0.5 * (p + q))
    klp = tl.sum(p * (lp - lm), axis=0)
    klq = tl.sum(q * (lq - lm), axis=0)
    tl.store(o_ptr + row, 0.5 * klp + 0.5 * klq)


def _run_jsd(inp):
    logp, logq = inp
    M, N = logp.shape
    o = torch.empty(M, device=DEVICE, dtype=torch.float32)
    _jsd_kernel[(M, )](logp, logq, o, M, N, BLOCK_N=N)
    return o


def _ref_jsd(inp):
    logp, logq = inp
    p, q = logp.exp(), logq.exp()
    lm = (0.5 * (p + q)).log()
    return 0.5 * (p * (logp - lm)).sum(1) + 0.5 * (q * (logq - lm)).sum(1)


register(name="jsd_loss", category="loss",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/jsd",
         make_inputs=lambda: (torch.log_softmax(randn(4096, 2048, seed=127), 1),
                              torch.log_softmax(randn(4096, 2048, seed=128), 1)),
         run=_run_jsd, ref=_ref_jsd, rtol=2e-3, atol=2e-3, shape_note="4096x2048")


# ######################################################################################### #
# ## GROUP 28 — ATTENTION VARIANTS (grouped-query, single-token decode)                    ## #
# ## source: https://github.com/pytorch-labs/tritonbench (operators/decoding_attention)    ## #
# ######################################################################################### #
@triton.jit
def _gqa_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, scale, S, Hq, Hkv, D: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)  # over B*Hq
    b = bh // Hq
    hq = bh % Hq
    hkv = hq // (Hq // Hkv)
    q_base = bh * S * D
    kv_base = (b * Hkv + hkv) * S * D
    offs_m = m_block * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + q_base + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < S, other=0.0)
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for start_n in range(0, S, BN):
        offs_n = start_n + tl.arange(0, BN)
        k = tl.load(k_ptr + kv_base + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < S, other=0.0)
        qk = tl.where(offs_n[None, :] < S, tl.dot(q, tl.trans(k)) * scale, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_base + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < S, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(o_ptr + q_base + offs_m[:, None] * D + offs_d[None, :], (acc / l_i[:, None]).to(o_ptr.dtype.element_ty),
             mask=offs_m[:, None] < S)


def _run_gqa(inp):
    q, k, v = inp
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    o = torch.empty_like(q)
    qf = q.reshape(B * Hq, S, D)
    kf = k.reshape(B * Hkv, S, D)
    vf = v.reshape(B * Hkv, S, D)
    of = o.reshape(B * Hq, S, D)
    _gqa_fwd_kernel[(triton.cdiv(S, 64), B * Hq)](qf, kf, vf, of, 1.0 / (D ** 0.5), S, Hq, Hkv, D=D, BM=64, BN=64)
    return o


def _ref_gqa(inp):
    q, k, v = inp
    g = q.shape[1] // k.shape[1]
    return torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1))


register(name="gqa_attention_fwd", category="attention",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/flash_attention",
         make_inputs=lambda: (randn(4, 16, 1024, 64, dtype=torch.float16, seed=129),
                              randn(4, 4, 1024, 64, dtype=torch.float16, seed=130),
                              randn(4, 4, 1024, 64, dtype=torch.float16, seed=131)),
         run=_run_gqa, ref=_ref_gqa, rtol=1e-2, atol=2e-2, shape_note="4x(16:4)x1024x64 GQA")


@triton.jit
def _attn_decode_kernel(q_ptr, k_ptr, v_ptr, o_ptr, scale, Skv, D: tl.constexpr, BN: tl.constexpr):
    bh = tl.program_id(0)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + bh * D + d)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, Skv, BN):
        offs_n = start + tl.arange(0, BN)
        nmask = offs_n < Skv
        k = tl.load(k_ptr + bh * Skv * D + offs_n[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.where(nmask, tl.sum(q[None, :] * k, axis=1) * scale, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(v_ptr + bh * Skv * D + offs_n[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    tl.store(o_ptr + bh * D + d, (acc / l_i).to(o_ptr.dtype.element_ty))


def _run_decode(inp):
    q, k, v = inp  # q [B,H,1,D], k/v [B,H,Skv,D]
    B, H, _, D = q.shape
    Skv = k.shape[2]
    o = torch.empty_like(q)
    _attn_decode_kernel[(B * H, )](q.reshape(B * H, D), k.reshape(B * H, Skv, D), v.reshape(B * H, Skv, D),
                                   o.reshape(B * H, D), 1.0 / (D ** 0.5), Skv, D=D, BN=128)
    return o


register(name="attention_decode", category="attention",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/decoding_attention",
         make_inputs=lambda: (randn(8, 16, 1, 64, dtype=torch.float16, seed=132),
                              randn(8, 16, 2048, 64, dtype=torch.float16, seed=133),
                              randn(8, 16, 2048, 64, dtype=torch.float16, seed=134)),
         run=_run_decode,
         ref=lambda inp: torch.nn.functional.scaled_dot_product_attention(inp[0], inp[1], inp[2]),
         rtol=1e-2, atol=2e-2, shape_note="8x16x(1<-2048)x64 decode")


# ######################################################################################### #
# ## GROUP 29 — WEIGHT-QUANTIZED GEMM (w8a16: int8 weight + per-N scale, fp16 activation)  ## #
# ## source: https://github.com/pytorch/ao (torchao/kernel/intmm_triton.py)               ## #
# ######################################################################################### #
@triton.jit
def _w8a16_gemm_kernel(a_ptr, b_ptr, scale_ptr, c_ptr, M, N, K,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BK), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BK) & (offs_n[None, :] < N), other=0)
        acc += tl.dot(a, b.to(tl.float16))
        a_ptrs += BK
        b_ptrs += BK * N
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * scale[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _run_w8a16(inp):
    a, b, scale = inp  # a fp16 [M,K], b int8 [K,N], scale fp16 [N]
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    _w8a16_gemm_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](a, b, scale, c, M, N, K, BM=64, BN=64, BK=32)
    return c


def _make_w8a16_inputs():
    a = randn(1024, 1024, dtype=torch.float16, seed=135)
    b = randint(16, (1024, 1024), seed=136, dtype=torch.int8) - 8
    scale = (rand(1024, seed=137) * 0.01 + 0.01).to(torch.float16)
    return a, b, scale


register(name="gemm_w8a16", category="quant",
         source="https://github.com/pytorch/ao/blob/main/torchao/kernel/intmm_triton.py",
         make_inputs=_make_w8a16_inputs,
         run=_run_w8a16,
         ref=lambda inp: (inp[0].float() @ inp[1].float()) * inp[2].float(),
         rtol=2e-2, atol=5e-1, shape_note="1024^3 int8-weight")


# ######################################################################################### #
# ## GROUP 30 — SORT / TOP-K (iterative argmax select)                                     ## #
# ## source: https://github.com/FlagOpen/FlagGems (src/flag_gems/ops/topk.py)             ## #
# ######################################################################################### #
@triton.jit
def _topk_kernel(x_ptr, o_ptr, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + cols, mask=cols < N, other=-float("inf"))
    for i in range(K):
        mx = tl.max(x, axis=0)
        am = tl.argmax(x, axis=0)
        tl.store(o_ptr + row * K + i, mx)
        x = tl.where(cols == am, -float("inf"), x)  # remove the selected max, repeat


def _run_topk(inp):
    x, = inp
    M, N = x.shape
    K = 8
    o = torch.empty((M, K), device=DEVICE, dtype=torch.float32)
    _topk_kernel[(M, )](x, o, M, N, K=K, BLOCK_N=triton.next_power_of_2(N))
    return o


register(name="topk_values", category="sort",
         source="https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/topk.py",
         make_inputs=lambda: (randn(4096, 2048, seed=138), ),
         run=_run_topk, ref=lambda inp: torch.topk(inp[0], 8, dim=1).values, rtol=2e-3, atol=2e-3,
         shape_note="4096x2048 top-8")


# ######################################################################################### #
# ## GROUP 31 — INT4 GEMM (2 int4 packed per byte along K; nibble unpack + tl.split)       ## #
# ## source: https://github.com/pytorch-labs/tritonbench (operators/int4_gemm)            ## #
# ######################################################################################### #
@triton.jit
def _int4_gemm_kernel(a_ptr, bpacked_ptr, scale_ptr, c_ptr, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BKP: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    Kp = K // 2
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kp in range(0, Kp, BKP):
        offs_kp = kp + tl.arange(0, BKP)
        offs_ka = 2 * kp + tl.arange(0, 2 * BKP)
        a_tile = tl.load(a_ptr + offs_m[:, None] * K + offs_ka[None, :], mask=offs_m[:, None] < M, other=0.0)
        a_lo, a_hi = tl.split(tl.reshape(a_tile, (BM, BKP, 2)))  # de-interleave consecutive K pairs
        byte = tl.load(bpacked_ptr + offs_kp[:, None] * N + offs_n[None, :], mask=offs_n[None, :] < N, other=0)
        b_lo = (byte & 0xF).to(tl.float16) - 8.0
        b_hi = ((byte >> 4) & 0xF).to(tl.float16) - 8.0
        acc += tl.dot(a_lo, b_lo) + tl.dot(a_hi, b_hi)
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * scale[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _make_int4_inputs():
    M, K, N = 512, 512, 512
    a = randn(M, K, dtype=torch.float16, seed=139)
    gg = torch.Generator(device="cpu").manual_seed(140)
    b_int4 = torch.randint(-8, 8, (K, N), generator=gg)  # values in [-8, 7]
    lo = (b_int4[0::2] + 8)
    hi = (b_int4[1::2] + 8)
    packed = (lo | (hi << 4)).to(torch.uint8).to(DEVICE)
    scale = (rand(N, seed=141) * 0.01 + 0.01).to(torch.float16)
    return a, packed, scale, b_int4.to(DEVICE)


def _run_int4_gemm(inp):
    a, packed, scale, _b = inp
    M, K = a.shape
    N = packed.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    _int4_gemm_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](a, packed, scale, c, M, N, K, BM=64, BN=64, BKP=16)
    return c


register(name="gemm_int4", category="quant",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/int4_gemm",
         make_inputs=_make_int4_inputs,
         run=_run_int4_gemm,
         ref=lambda inp: (inp[0].float() @ inp[3].float()) * inp[2].float(),
         rtol=2e-2, atol=5e-1, shape_note="512^3 int4-weight (2/byte)")


# ######################################################################################### #
# ## GROUP 32 — CHUNKED SSM / linear scan with cross-chunk state carry (mamba2-style)      ## #
# ## source: https://github.com/fla-org/flash-linear-attention (fla/ops/*/chunk*.py)      ## #
# ######################################################################################### #
@triton.jit
def _chunked_ssm_kernel(a_ptr, b_ptr, o_ptr, L, C: tl.constexpr):
    row = tl.program_id(0)
    h = tl.zeros([], dtype=tl.float32)
    idx = tl.arange(0, C)
    for c0 in range(0, L, C):
        cols = c0 + idx
        a = tl.load(a_ptr + row * L + cols)
        b = tl.load(b_ptr + row * L + cols)
        a_cum, b_cum = tl.associative_scan((a, b), 0, _ssm_combine)  # within-chunk affine scan
        out = a_cum * h + b_cum  # graft the carried state from the previous chunk
        tl.store(o_ptr + row * L + cols, out)
        h = tl.sum(tl.where(idx == C - 1, out, 0.0), axis=0)  # last element -> next chunk's state


def _run_chunked_ssm(inp):
    a, b = inp
    M, L = a.shape
    o = torch.empty_like(a)
    _chunked_ssm_kernel[(M, )](a, b, o, L, C=128)
    return o


def _ref_chunked_ssm(inp):
    a, b = inp
    h = torch.zeros(a.shape[0], device=DEVICE, dtype=torch.float32)
    outs = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


register(name="chunked_ssm_scan", category="ssm",
         source="https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/cumsum.py",
         make_inputs=lambda: (rand(2048, 1024, seed=142) * 0.5 + 0.25, randn(2048, 1024, seed=143)),
         run=_run_chunked_ssm, ref=_ref_chunked_ssm, rtol=2e-3, atol=2e-3, shape_note="2048x1024 C=128 carry")


# ######################################################################################### #
# ## GROUP 33 — BLOCK-SPARSE GEMM (skip all-zero A blocks via a block mask)                ## #
# ## source: https://github.com/pytorch/ao (torchao/kernel/bsr_triton_ops.py)             ## #
# ######################################################################################### #
@triton.jit
def _block_sparse_gemm_kernel(a_ptr, b_ptr, mask_ptr, c_ptr, M, N, K, num_k,
                              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(num_k):
        if tl.load(mask_ptr + pid_m * num_k + k) != 0:  # skip empty K-blocks (data-dependent)
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
            acc += tl.dot(a, b)
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _make_block_sparse_inputs():
    M = N = K = 1024
    bm, bk = 64, 32
    a = randn(M, K, dtype=torch.float16, seed=144)
    b = randn(K, N, dtype=torch.float16, seed=145)
    num_m, num_k = M // bm, K // bk
    mask = (rand(num_m, num_k, seed=146) > 0.5).to(torch.int32)
    for i in range(num_m):
        for j in range(num_k):
            if mask[i, j] == 0:
                a[i * bm:(i + 1) * bm, j * bk:(j + 1) * bk] = 0
    return a, b, mask.contiguous()


def _run_block_sparse(inp):
    a, b, mask = inp
    M, K = a.shape
    N = b.shape[1]
    num_k = mask.shape[1]
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    _block_sparse_gemm_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        a, b, mask, c, M, N, K, num_k, BM=64, BN=64, BK=32)
    return c


register(name="block_sparse_gemm", category="sparse",
         source="https://github.com/pytorch/ao/blob/main/torchao/kernel/bsr_triton_ops.py",
         make_inputs=_make_block_sparse_inputs,
         run=_run_block_sparse, ref=lambda inp: inp[0] @ inp[1], rtol=1e-2, atol=3e-1,
         shape_note="1024^3 ~50% block-sparse")


# ######################################################################################### #
# ## GROUP 34 — VARLEN ATTENTION (per-batch key length / padding mask)                     ## #
# ## source: https://github.com/pytorch-labs/tritonbench (operators/ragged_attention)     ## #
# ######################################################################################### #
@triton.jit
def _varlen_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, kvlen_ptr, scale, S, H, D: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    kvlen = tl.load(kvlen_ptr + b)
    base = bh * S * D
    offs_m = m_block * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + base + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < S, other=0.0)
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for start_n in range(0, S, BN):
        offs_n = start_n + tl.arange(0, BN)
        k = tl.load(k_ptr + base + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < kvlen, other=0.0)
        qk = tl.where(offs_n[None, :] < kvlen, tl.dot(q, tl.trans(k)) * scale, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + base + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < kvlen, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(o_ptr + base + offs_m[:, None] * D + offs_d[None, :], (acc / l_i[:, None]).to(o_ptr.dtype.element_ty),
             mask=offs_m[:, None] < S)


def _run_varlen_attn(inp):
    q, k, v, kvlen = inp
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    qf, kf, vf, of = (t.reshape(B * H, S, D) for t in (q, k, v, o))
    _varlen_attn_kernel[(triton.cdiv(S, 64), B * H)](qf, kf, vf, of, kvlen, 1.0 / (D ** 0.5), S, H, D=D, BM=64, BN=64)
    return o


def _make_varlen_inputs():
    B, H, S, D = 4, 8, 1024, 64
    q = randn(B, H, S, D, dtype=torch.float16, seed=147)
    k = randn(B, H, S, D, dtype=torch.float16, seed=148)
    v = randn(B, H, S, D, dtype=torch.float16, seed=149)
    kvlen = randint(S // 2, (B, ), seed=150, dtype=torch.int32) + S // 2  # each in [S/2, S]
    return q, k, v, kvlen


def _ref_varlen_attn(inp):
    q, k, v, kvlen = inp
    B, H, S, D = q.shape
    n = torch.arange(S, device=DEVICE)
    keep = n[None, :] < kvlen[:, None]  # [B, S]
    attn_mask = keep[:, None, None, :]  # [B,1,1,S] broadcast over heads + queries
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


register(name="varlen_attention", category="attention",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/ragged_attention",
         make_inputs=_make_varlen_inputs,
         run=_run_varlen_attn, ref=_ref_varlen_attn, rtol=1e-2, atol=2e-2, shape_note="4x8x1024x64 varlen kv")


# ######################################################################################### #
# ## GROUP 35 — SEQUENCE-MODEL RECURRENCES (RWKV time-mix, retention)                      ## #
# ## source: https://github.com/fla-org/flash-linear-attention (fla/ops/{rwkv6,retention}) ## #
# ######################################################################################### #
@triton.jit
def _rwkv_kernel(k_ptr, v_ptr, w_ptr, o_ptr, T, BLOCK_T: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_T)
    k = tl.load(k_ptr + row * T + cols)
    v = tl.load(v_ptr + row * T + cols)
    d = tl.exp(-tl.load(w_ptr + row))  # per-channel decay in (0,1)
    ek = tl.exp(k)
    g = tl.zeros([BLOCK_T], dtype=tl.float32) + d
    _, num = tl.associative_scan((g, ek * v), 0, _ssm_combine)  # sum_i d^(t-i) exp(k_i) v_i
    _, den = tl.associative_scan((g, ek), 0, _ssm_combine)      # sum_i d^(t-i) exp(k_i)
    tl.store(o_ptr + row * T + cols, num / den)


def _run_rwkv(inp):
    k, v, w = inp
    M, T = k.shape
    o = torch.empty_like(k)
    _rwkv_kernel[(M, )](k, v, w, o, T, BLOCK_T=T)
    return o


def _ref_rwkv(inp):
    k, v, w = inp
    d = torch.exp(-w)
    ek = torch.exp(k)
    a = torch.zeros(k.shape[0], device=DEVICE)
    b = torch.zeros(k.shape[0], device=DEVICE)
    outs = []
    for t in range(k.shape[1]):
        a = d * a + ek[:, t] * v[:, t]
        b = d * b + ek[:, t]
        outs.append(a / b)
    return torch.stack(outs, dim=1)


register(name="rwkv_time_mix", category="ssm",
         source="https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv6",
         make_inputs=lambda: (randn(2048, 512, seed=151) * 0.5, randn(2048, 512, seed=152),
                              rand(2048, seed=153) * 1.0 + 0.5),
         run=_run_rwkv, ref=_ref_rwkv, rtol=3e-3, atol=3e-3, shape_note="2048x512 RWKV recurrence")


@triton.jit
def _retention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, decay, scale, S, D: tl.constexpr, BS: tl.constexpr):
    bh = tl.program_id(0)
    base = bh * S * D
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + base + s[:, None] * D + d[None, :])
    k = tl.load(k_ptr + base + s[:, None] * D + d[None, :])
    v = tl.load(v_ptr + base + s[:, None] * D + d[None, :])
    qk = tl.dot(q, tl.trans(k)) * scale
    diff = (s[:, None] - s[None, :]).to(tl.float32)
    dmask = tl.where(s[:, None] >= s[None, :], tl.exp(diff * tl.log(decay)), 0.0)  # decay^(t-i), causal
    ret = (qk * dmask).to(v.dtype)
    tl.store(o_ptr + base + s[:, None] * D + d[None, :], tl.dot(ret, v).to(o_ptr.dtype.element_ty))


def _run_retention(inp):
    q, k, v = inp
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    qf, kf, vf, of = (t.reshape(B * H, S, D) for t in (q, k, v, o))
    _retention_kernel[(B * H, )](qf, kf, vf, of, 0.95, 1.0 / (D ** 0.5), S, D=D, BS=S)
    return o


def _ref_retention(inp):
    q, k, v = inp
    S = q.shape[2]
    qk = (q.float() @ k.float().transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    t = torch.arange(S, device=DEVICE)
    diff = (t[:, None] - t[None, :]).float()
    dm = torch.where(diff >= 0, 0.95 ** diff, torch.zeros_like(diff))
    return (qk * dm) @ v.float()


register(name="retention_parallel", category="attention",
         source="https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/retention",
         make_inputs=lambda: (randn(8, 8, 128, 64, dtype=torch.float16, seed=154),
                              randn(8, 8, 128, 64, dtype=torch.float16, seed=155),
                              randn(8, 8, 128, 64, dtype=torch.float16, seed=156)),
         run=_run_retention, ref=_ref_retention, rtol=2e-2, atol=1e-1, shape_note="8x8x128x64 decay=0.95")


# ######################################################################################### #
# ## GROUP 36 — BLACKWELL / SPARSE-TENSOR-CORE (reference only; flagged NOT runnable here)  ## #
# ## These need Nvidia Blackwell (sm_100) 5th-gen tensor core (tcgen05) block-scaled MMA,   ## #
# ## or the 2:4 sparse tensor core — neither runs on this H100 (sm_90), and 2:4 has no      ## #
# ## portable Triton intrinsic. Bodies are kept as REFERENCE and marked runnable=False so   ## #
# ## the runner SKIPs them with a note (run on a Blackwell host / via cusparselt to use).   ## #
# ## source: https://github.com/triton-lang/triton (tutorials/10-block-scaled-matmul.py)   ## #
# ######################################################################################### #
@triton.jit
def _mxfp4_gemm_kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, M, N, K,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # a/b are packed OCP-MX fp4 (e2m1, 2 values/byte); a_scale/b_scale are e8m0 per-32 block
    # scales. tl.dot_scaled lowers to the sm_100 tcgen05 block-scaled MMA (Hopper has no fp4).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK // 2)  # 2 fp4 per byte
    a_ptrs = a_ptr + offs_m[:, None] * (K // 2) + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.load(a_scale_ptr + offs_m[:, None] * (K // 32) + _k)
        b_s = tl.load(b_scale_ptr + offs_n[None, :] * (K // 32) + _k)
        acc = tl.dot_scaled(a, a_s, "e2m1", b, b_s, "e2m1", acc)  # sm_100 tcgen05 block-scaled MMA
        a_ptrs += BK // 2
        b_ptrs += BK * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


register(name="mxfp4_gemm_blockscaled", category="blackwell",
         source="https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py",
         make_inputs=lambda: None, run=lambda inp: None, runnable=False,
         note="needs Blackwell sm_100 tcgen05 block-scaled fp4 MMA (tl.dot_scaled e2m1); not on H100",
         shape_note="mxfp4 (e2m1 + e8m0/32)")


@triton.jit
def _nvfp4_gemm_kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, M, N, K,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # NVFP4 = fp4 (e2m1) data with a per-16 block fp8 (e4m3) scale — the Nvidia variant of the
    # OCP-MX format. Same tcgen05 block-scaled MMA path as mxfp4, block size 16 + e4m3 scales.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK // 2)
    a_ptrs = a_ptr + offs_m[:, None] * (K // 2) + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.load(a_scale_ptr + offs_m[:, None] * (K // 16) + _k)
        b_s = tl.load(b_scale_ptr + offs_n[None, :] * (K // 16) + _k)
        acc = tl.dot_scaled(a, a_s, "e2m1", b, b_s, "e2m1", acc)
        a_ptrs += BK // 2
        b_ptrs += BK * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


register(name="nvfp4_gemm_blockscaled", category="blackwell",
         source="https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/operators/nvfp4_gemm",
         make_inputs=lambda: None, run=lambda inp: None, runnable=False,
         note="needs Blackwell sm_100 tcgen05 block-scaled fp4 MMA (nvfp4: e2m1 + e4m3/16); not on H100",
         shape_note="nvfp4 (e2m1 + e4m3/16)")


@triton.jit
def _sparse_2to4_kernel(a_val_ptr, a_meta_ptr, b_ptr, c_ptr, M, N, K,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # 2:4 structured sparsity: A is stored compressed (a_val = the 2 nonzeros of every 4 along K,
    # a_meta = their 2-bit positions). The real matmul uses the sparse tensor core (mma.sp), which
    # Triton has no portable intrinsic for (it goes through cusparselt / CUTLASS). A dense-emulated
    # decompress+dot would work but is not the sparse-TC path; kept here only to mark the shape.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # acc = sparse_mma(a_val[BM, BK//2], a_meta[BM, BK//2], b[BK, BN])  # <- no portable Triton op
    _ = a_val_ptr + a_meta_ptr + b_ptr  # (reference stub)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


register(name="sparse_2to4_mma", category="sparse",
         source="https://github.com/pytorch/ao/blob/main/torchao/sparsity",
         make_inputs=lambda: None, run=lambda inp: None, runnable=False,
         note="2:4 structured sparse tensor core (mma.sp) has no portable Triton intrinsic (cusparselt/CUTLASS)",
         shape_note="2:4 sparse MMA")


if __name__ == "__main__":
    main(sys.argv[1:])
