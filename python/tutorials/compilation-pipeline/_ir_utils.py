"""Tiny IR-capture helpers shared by the compilation-pipeline tutorials.

Self-contained on purpose (no project-specific imports) so each tutorial runs on
a bare checkout. The one primitive every example needs is:

    ck = compile_only(kernel, *args, grid=(...,), CONSTEXPR=...)   # NO launch
    ck.asm["ttir" | "ttgir" | "llir" | "ptx"]                     # per-stage IR
    ck.metadata.num_warps, ck.metadata.shared, ...                # compile metadata

``compile_only`` uses Triton's ``warmup`` path, which compiles a ``@triton.jit``
kernel **without launching** it, so the IR can be read on a machine whose GPUs
can't run kernels. Launching (for the runtime correctness check) is done
separately with the normal ``kernel[grid](...)`` call.
"""
import difflib

import torch


def is_cuda():
    return torch.cuda.is_available()


def is_blackwell():
    """Datacenter Blackwell (cc major 10/11): MMAv5 / tcgen05 + TMEM + AutoWS."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major in (10, 11)


def compile_only(kernel, *args, grid=(1, ), **meta):
    """Compile a ``@triton.jit`` kernel WITHOUT launching; return the CompiledKernel.

    The result exposes ``.asm`` (keyed ``ttir|ttgir|llir|ptx|cubin|...``) and
    ``.metadata`` (``num_warps``, ``num_stages``, ``shared``, ``target`` ...).
    """
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
    """How many times ``needle`` appears in an IR stage (handy for op-presence checks)."""
    return ck.asm[name].count(needle)


def diff(ck_a, ck_b, name, grep=None, label_a="A", label_b="B", limit=80):
    """Unified diff of one IR stage between two compiled kernels (e.g. two configs)."""
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
