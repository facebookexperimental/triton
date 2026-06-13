"""IR-capture helpers for the layout/numerics examples (project: bitwise equivalence).

A trimmed, self-contained copy of the compilation-pipeline harness (no
``bitequiv.*`` imports, so these run on a bare checkout). The numerics examples
need the same primitive — compile a kernel WITHOUT launching and read its
per-stage IR — plus a couple of small text helpers for spotting the ops that
decide FP bits.

    ck = compile_only(kernel, *args, grid=(...,), CONSTEXPR=...)  # no launch
    ck.asm["ttir" | "ttgir" | "llir" | "ptx"]                    # per-stage IR

The general "what does each stage do" walkthrough lives in
``bitequiv/knowledge-base/compilation-pipeline/``; here the focus is *which passes
change the bits* (see ``bitequiv/knowledge-base/numerics-modifying-passes.md`` and
``tree-reduction-in-ptx-and-triton.md``).
"""
import difflib

import torch


def is_cuda():
    return torch.cuda.is_available()


def is_blackwell():
    """Datacenter Blackwell (cc major 10/11): MMAv5 / tcgen05 + TMEM."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major in (10, 11)


def compile_only(kernel, *args, grid=(1, ), **meta):
    """Compile a ``@triton.jit`` kernel WITHOUT launching; return the CompiledKernel."""
    return kernel.warmup(*args, grid=grid, **meta)


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _filtered(text, grep):
    lines = text.splitlines()
    if grep:
        subs = [grep] if isinstance(grep, str) else list(grep)
        lines = [ln for ln in lines if any(s in ln for s in subs)]
    return lines


def show(ck, name, grep=None, limit=40, label=None):
    """Print one IR stage, optionally only lines containing any substring in ``grep``."""
    lines = _filtered(ck.asm[name], grep)
    print(label or (f"--- {name.upper()}" + (f"  (grep={grep})" if grep else "") + " ---"))
    for ln in lines[:limit]:
        print("    " + ln.rstrip())
    if len(lines) > limit:
        print(f"    ... ({len(lines) - limit} more lines)")
    if not lines:
        print("    (no matching lines)")


def count(ck, name, needle):
    """How many times ``needle`` appears in an IR stage."""
    return ck.asm[name].count(needle)


def first(ck, name, needle):
    """The first line of an IR stage containing ``needle`` (stripped), or ''."""
    for ln in ck.asm[name].splitlines():
        if needle in ln:
            return ln.strip()
    return ""


def diff(ck_a, ck_b, name, grep=None, label_a="A", label_b="B", limit=80):
    """Unified diff of one IR stage between two compiled kernels."""
    d = list(
        difflib.unified_diff(_filtered(ck_a.asm[name], grep), _filtered(ck_b.asm[name], grep), fromfile=label_a,
                             tofile=label_b, lineterm=""))
    print(f"--- diff {name.upper()} ({label_a} vs {label_b})" + (f"  grep={grep}" if grep else "") + " ---")
    for ln in d[:limit]:
        print("    " + ln)
    if len(d) > limit:
        print(f"    ... ({len(d) - limit} more diff lines)")
    if not d:
        print("    (identical)")


def hexbits(t):
    """Raw little-endian hex of a 1-element float tensor (to compare exact bits)."""
    return t.detach().cpu().view(torch.int32 if t.dtype == torch.float32 else torch.int16).item()
