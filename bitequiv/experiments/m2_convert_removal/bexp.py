"""3-way experiment: base vs A(convert) vs B(ideal-on-load, no convert).

B is emulated faithfully: take the baseline TTGIR and swap the LOAD layout's
definition body for the ideal (reduce-friendly) layout that A converts to. That
makes the load produce the reduce-friendly layout directly with ZERO convert --
exactly what approach B wants in its best case. We inject that TTGIR by patching
CUDABackend.make_ttgir to return our parsed module (same mechanism evaluate_opt
uses to inject A's pass), so the rest of the pipeline (ttgir->llir->ptx->cubin,
incl. shared-memory allocation) is recomputed from the B TTGIR.

For each (kernel, num_warps) we record base_ms / A_ms / B_ms and check base==A==B
bitwise on plain data (must hold: inner_tree makes layout a free knob) -- both a
validity check on the B edit and the bit-safety sanity check. Results append to a
TSV per config (resumable across a killgpu reap from the sibling session).
"""

import ast
import os
import re
import sys

os.environ["TRITON_ALWAYS_COMPILE"] = "1"

# repo root: this file is <root>/bitequiv/experiments/m2_convert_removal/bexp.py
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)

import contextlib  # noqa: E402

import torch  # noqa: E402

from bitequiv.evaluation import evaluate_opt  # noqa: E402
from bitequiv.evaluation.eval_kernels import resolve_kernels  # noqa: E402
from triton.compiler.compiler import parse  # noqa: E402

PASS = "tritongpu-optimize-reduction-layout"
BDIR = "/tmp/m2_probe/b_ttgir"
TSV = "/tmp/m2_probe/bexp_results.tsv"
os.makedirs(BDIR, exist_ok=True)

# The pass takes 3 Options (strategy, min-underparallel, max-elems-per-thread);
# pass the defaults so the binding resolves. Empty args would be classified as
# needs-args and the pass would be treated as baseline (a silent no-op).
resolved, _na, _un = evaluate_opt._classify([(PASS, ["ideal", 8, 256])])
assert resolved, "M2 pass binding not found"


def _cfg(spec, num_warps):
    cs = [c for c in spec.config_space("light")
          if c.reduction_ordering == "inner_tree" and c.num_warps == num_warps]
    assert cs, f"no inner_tree/nw={num_warps} config"
    return cs[0]


def _def_of(ttgir, name):
    m = re.search(rf"^#{name}\s*=\s*(#ttg\.blocked<.*>)\s*$", ttgir, re.M)
    assert m, f"no def for #{name}"
    return m.group(1)


def _load_layout_name(ttgir):
    m = re.search(r"tt\.load.*,\s*#(\w+)>", ttgir)
    assert m, "no tt.load layout"
    return m.group(1)


def _reduce_operand_name(ttgir):
    m = re.search(r"\}\)\s*:\s*\(tensor<[^)]*?,\s*#(\w+)>\)\s*->", ttgir)
    assert m, "no tt.reduce operand layout"
    return m.group(1)


def _compile(spec, cfg, size, ctx):
    evaluate_opt._clear_jit_caches()
    with ctx:
        return spec.compile(cfg, size)


@contextlib.contextmanager
def optimization_B(path):
    """Inject the B TTGIR: patch make_ttgir to return the parsed B module."""
    from triton.backends.nvidia.compiler import CUDABackend
    original = CUDABackend.make_ttgir

    def patched(mod, metadata, opt, capability):
        m = original(mod, metadata, opt, capability)  # keep metadata side effects
        return parse(path, "ttgir", m.context)

    CUDABackend.make_ttgir = staticmethod(patched)
    try:
        yield
    finally:
        CUDABackend.make_ttgir = staticmethod(original)


def build_B(spec, cfg):
    """Return (b_path, load_body, ideal_body) or (None, reason, None)."""
    size = spec.precision_size
    off = _compile(spec, cfg, size, evaluate_opt.optimization([])).asm["ttgir"]
    on = _compile(spec, cfg, size, evaluate_opt.optimization(resolved)).asm["ttgir"]

    load_name = _load_layout_name(off)
    ideal_name = _reduce_operand_name(on)
    ideal_body = _def_of(on, ideal_name)
    load_body = _def_of(off, load_name)
    if load_body == ideal_body:
        return None, "A did not change the reduce layout (pass skipped this config)", None

    b_ttgir = re.sub(rf"^#{load_name}\s*=\s*#ttg\.blocked<.*>\s*$",
                     f"#{load_name} = {ideal_body}", off, count=1, flags=re.M)
    assert b_ttgir != off, "B edit did not apply"
    b_path = f"{BDIR}/{spec.name}_nw{cfg.num_warps}.ttgir"
    with open(b_path, "w") as f:
        f.write(b_ttgir)
    return b_path, load_body, ideal_body


def _bench(spec, cfg, ctx):
    evaluate_opt._clear_jit_caches()
    with ctx:
        return spec.benchmark(cfg, spec.perf_size)  # (ms, bytes, asm)


def _already_done(kernel, num_warps):
    if not os.path.exists(TSV):
        return False
    with open(TSV) as f:
        return any(line.startswith(f"{kernel}\t{num_warps}\t") for line in f)


def run(kernel, num_warps):
    if _already_done(kernel, num_warps):
        print(f"  {kernel} nw{num_warps}: already in TSV, skip")
        return
    spec = resolve_kernels(kernel)[0]
    cfg = _cfg(spec, num_warps)

    b_path, load_body, ideal_body = build_B(spec, cfg)
    if b_path is None:
        print(f"  {kernel} nw{num_warps}: SKIP -- {load_body}")
        return

    base_ms, base_bytes, _ = _bench(spec, cfg, evaluate_opt.optimization([]))
    a_ms, a_bytes, _ = _bench(spec, cfg, evaluate_opt.optimization(resolved))

    ck_b = _compile(spec, cfg, spec.perf_size, optimization_B(b_path))
    b_ttgir = ck_b.asm["ttgir"]
    b_is_ideal = (_def_of(b_ttgir, _load_layout_name(b_ttgir)) == ideal_body)
    b_pre_conv = b_ttgir[:b_ttgir.index("tt.reduce")].count("convert_layout")
    b_ms, b_bytes, _ = _bench(spec, cfg, optimization_B(b_path))

    bit_ok = (base_bytes == a_bytes == b_bytes)
    a_sp, b_sp = base_ms / a_ms, base_ms / b_ms
    print(f"\n  {kernel} nw{num_warps}  size={spec.perf_size}")
    print(f"    load(off)= {load_body}")
    print(f"    ideal(A) = {ideal_body}")
    print(f"    B load-is-ideal={b_is_ideal}  B pre-reduce converts={b_pre_conv}  bit_ok(base==A==B)={bit_ok}")
    print(f"    base {base_ms:.4f}ms | A {a_ms:.4f}ms ({a_sp:.2f}x) | B {b_ms:.4f}ms ({b_sp:.2f}x)   B/A = {b_ms/a_ms:.2f}x slower")

    new = not os.path.exists(TSV)
    with open(TSV, "a") as f:
        if new:
            f.write("kernel\tnum_warps\tbase_ms\tA_ms\tB_ms\tA_speedup\tB_speedup\tB_over_A\tbit_ok\tB_is_ideal\tB_pre_converts\n")
        f.write(f"{kernel}\t{num_warps}\t{base_ms:.5f}\t{a_ms:.5f}\t{b_ms:.5f}\t{a_sp:.3f}\t{b_sp:.3f}\t{b_ms/a_ms:.3f}\t{bit_ok}\t{b_is_ideal}\t{b_pre_conv}\n")


TARGETS = ast.literal_eval(os.environ.get("BEXP_TARGETS", "[('sum_2d_col',[4])]"))

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("no GPU"); sys.exit(1)
    print(f"device: {torch.cuda.get_device_name()}")
    for kernel, nws in TARGETS:
        for nw in nws:
            try:
                run(kernel, nw)
            except Exception as exc:
                import traceback
                print(f"  {kernel} nw{nw}: ERROR {type(exc).__name__}: {exc}")
                traceback.print_exc()
