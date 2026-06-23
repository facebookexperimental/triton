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

To look at *individual compiler passes* (not just the per-stage output IR), use
``dump_passes`` + ``pass_diff``: they drive Triton's own ``MLIR_ENABLE_DUMP``
machinery and show the IR a single named pass rewrote (before vs after).
"""
import contextlib
import difflib
import os
import re
import sys
import tempfile

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


# compute-capability -> human label, used by the cross-arch MMA tutorials.
_CC_NAMES = {80: "Ampere sm_80", 90: "Hopper sm_90", 100: "Blackwell sm_100", 120: "Blackwell sm_120"}


def cc_name(cc):
    """Human-readable name for a CUDA compute capability (e.g. 90 -> 'Hopper sm_90')."""
    return _CC_NAMES.get(cc, f"sm_{cc}")


def compile_for_target(kernel, signature, constexprs, cc, **options):
    """Compile a ``@triton.jit`` kernel for an ARBITRARY arch — no device required.

    ``compile_only`` always targets the *current* GPU. To contrast how a pass
    lowers the same kernel on different hardware (e.g. Hopper wgmma vs Blackwell
    tcgen05), we instead build an :class:`ASTSource` by hand and hand it an
    explicit :class:`GPUTarget`, so the whole TTIR->...->PTX pipeline runs for
    ``cc`` even on a box whose GPU is a different arch. Only *compilation* is
    arch-free; launching still needs the real hardware (the tutorials self-gate
    their runtime correctness check accordingly).

    ``signature`` maps each kernel arg name to a Triton type string
    (``"*fp16"``, ``"*fp32"``, ``"constexpr"`` ...); ``constexprs`` maps the
    constexpr arg names to their values. ``options`` (``num_warps``,
    ``num_stages`` ...) are forwarded verbatim. The result exposes the same
    ``.asm`` keys as :func:`compile_only`.
    """
    import triton  # local import: keep this helper module dependency-light
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=GPUTarget("cuda", cc, 32), options=options or None)


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


# --------------------------------------------------------------------------- #
# Per-pass IR hook
#
# ``ck.asm[...]`` only exposes the *stage* outputs (ttir/ttgir/llir/ptx). To see
# what an individual MLIR pass did, we turn on Triton's ``MLIR_ENABLE_DUMP``,
# which makes the pass manager print the whole module before and after EVERY
# pass. That output goes to ``llvm::dbgs()`` (a C-level, unbuffered stderr), so
# we capture it by redirecting fd 2 — not Python's ``sys.stderr`` — around a
# single ``warmup`` call. ``knobs.compilation.always_compile`` defeats the
# on-disk kernel cache so the passes actually re-run and dump.
# --------------------------------------------------------------------------- #

# e.g. "// -----// IR Dump Before Coalesce (tritongpu-coalesce) ... //----- //"
_PASS_HDR = re.compile(r"^// -----// IR Dump (Before|After)\s+(\S+)(?:\s+\(([^)]*)\))?")


@contextlib.contextmanager
def _redirect_fd(fd=2):
    """Redirect a C-level file descriptor (default stderr) to a temp file.

    MLIR's pass dumps go to ``llvm::dbgs()``, a C++ stream that is NOT Python's
    ``sys.stderr``, so we must redirect at the fd level to capture them.
    """
    saved = os.dup(fd)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), fd)
        yield tmp
    finally:
        os.dup2(saved, fd)
        os.close(saved)


def _parse_pass_dumps(text):
    """Split a raw ``MLIR_ENABLE_DUMP`` capture into ordered per-pass records."""
    records, cur = [], None
    for line in text.splitlines():
        m = _PASS_HDR.match(line)
        if m:
            event, cls, arg = m.group(1), m.group(2), m.group(3)
            cur = {"event": event, "pass": cls + (f" ({arg})" if arg else ""), "lines": []}
            records.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    return [{"event": r["event"], "pass": r["pass"], "ir": "\n".join(r["lines"]).strip()} for r in records]


def dump_passes(kernel, *args, kernel_name=None, grid=(1, ), **meta):
    """Compile ``kernel`` with per-pass IR dumping on; return the parsed dumps.

    Returns an ordered list of records ``{"event": "Before"|"After", "pass":
    "<Name> (<arg>)", "ir": "<module text>"}``. Triton enables IR printing with
    ``printAfterOnlyOnFailure``, so on a successful compile MLIR emits one
    **Before** snapshot per pass (the state *after* a pass is simply the *Before*
    snapshot of the next pass — that is how :func:`pass_diff` reconstructs it).
    Use :func:`pass_names` to discover what ran.
    """
    import triton  # local import: keep this helper module dependency-light

    name = kernel_name or kernel.__name__
    prev_dump = os.environ.get("MLIR_ENABLE_DUMP")
    prev_always = triton.knobs.compilation.always_compile
    os.environ["MLIR_ENABLE_DUMP"] = name  # filter dumps to just this kernel
    triton.knobs.compilation.always_compile = True  # bypass the on-disk cache
    # ...and evict the in-memory cache, which is consulted *before* always_compile
    # (a prior compile_only of the same signature would otherwise be reused with
    # no recompile and so no dump).
    for entry in getattr(kernel, "device_caches", {}).values():
        entry[0].clear()  # entry[0] is the per-device kernel_cache dict
    try:
        with _redirect_fd(2) as cap:
            kernel.warmup(*args, grid=grid, **meta)
            sys.stderr.flush()
        cap.seek(0)
        text = cap.read().decode("utf-8", "replace")
    finally:
        triton.knobs.compilation.always_compile = prev_always
        if prev_dump is None:
            os.environ.pop("MLIR_ENABLE_DUMP", None)
        else:
            os.environ["MLIR_ENABLE_DUMP"] = prev_dump
    return _parse_pass_dumps(text)


def pass_names(dumps):
    """Ordered, de-duplicated list of pass names seen in ``dumps`` (for discovery)."""
    seen = []
    for r in dumps:
        if r["pass"] not in seen:
            seen.append(r["pass"])
    return seen


def pass_diff(dumps, pass_substr, grep=None, limit=80):
    """Print the IR diff a single named pass produced (its before vs after).

    ``pass_substr`` is matched case-insensitively as a substring of the pass
    name, so ``pass_diff(dumps, "coalesce")`` finds ``Coalesce
    (tritongpu-coalesce)`` regardless of exact registration spelling. Since MLIR
    only emits Before snapshots, "after" is the Before snapshot of the next pass.
    Returns ``(before_ir, after_ir)``.
    """
    key = pass_substr.lower()
    befores = [r for r in dumps if r["event"] == "Before"]
    idx = next((i for i, r in enumerate(befores) if key in r["pass"].lower()), None)
    if idx is None:
        avail = ", ".join(pass_names(dumps)) or "(none)"
        raise ValueError(f"no pass matching {pass_substr!r} in dumps; available: {avail}")
    matched = befores[idx]["pass"]
    before = befores[idx]["ir"]
    # State after this pass == the next pass's Before snapshot (nothing runs in
    # between). The very last pass has no successor, so fall back to itself.
    after = befores[idx + 1]["ir"] if idx + 1 < len(befores) else before
    # ">>> pass-diff:" is the marker the smoke test greps for to prove the
    # per-pass loop actually ran (not just the final stage IR).
    print(f">>> pass-diff: {matched}" + (f"  (grep={grep})" if grep else ""))
    d = list(
        difflib.unified_diff(_filtered(before, grep), _filtered(after, grep), fromfile="before", tofile="after",
                             lineterm=""))
    for ln in d[:limit]:
        print("    " + ln)
    if len(d) > limit:
        print(f"    ... ({len(d) - limit} more diff lines)")
    if not d:
        print("    (no change under this grep filter)")
    return before, after
