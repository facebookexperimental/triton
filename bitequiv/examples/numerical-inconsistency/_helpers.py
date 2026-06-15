"""Self-contained helpers for the numerical-inconsistency examples.

This folder is a STANDALONE repro: it must run on a bare checkout at the pinned
upstream commit (see ``UPSTREAM_COMMIT.txt``), where ``bitequiv/equivalence_ttgir.py``
and the rest of ``bitequiv/`` do NOT exist yet. So we carry our own copy of the
IR-capture primitives here and import nothing from ``bitequiv.*``.

What these provide:

    ck = compile_only(kernel, *args, grid=(...,), CONSTEXPR=...)  # compile, no launch
    ck.asm["ttir" | "ttgir" | "llir" | "ptx"]                    # per-stage IR

plus the load-bearing additions for THIS folder:

    bitclass_key(t)        # exact bits of a WHOLE tensor, as a hashable key
    group_by_bits(pairs)   # [(label, tensor)] -> {bitkey: [labels]}
    short(bitkey)          # compact printable id for a bit-class
    adversarial_1d/2d(...) # inputs engineered to maximize FP non-associativity

The mechanism these examples demonstrate (layout -> reduction tree -> bits) is
written up in ``bitequiv/knowledge-base/tree-reduction-in-ptx-and-triton.md``.
"""
import hashlib

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


def first(ck, name, needle):
    """The first line of an IR stage containing ``needle`` (stripped), or ''."""
    for ln in ck.asm[name].splitlines():
        if needle in ln:
            return ln.strip()
    return ""


def count(ck, name, needle):
    """How many times ``needle`` appears in an IR stage."""
    return ck.asm[name].count(needle)


# --- exact-bit comparison over whole tensors -------------------------------

def _int_view_dtype(t):
    if t.dtype == torch.float32:
        return torch.int32
    if t.dtype in (torch.float16, torch.bfloat16):
        return torch.int16
    if t.dtype == torch.float64:
        return torch.int64
    raise TypeError(f"unsupported dtype for bit comparison: {t.dtype}")


def hexbits(t):
    """Raw little-endian int of a 1-element float tensor (to compare exact bits)."""
    return t.detach().cpu().view(_int_view_dtype(t)).item()


def bitclass_key(t):
    """Stable, hashable key for a tensor's EXACT bits (whole tensor, not 1 elem).

    Two tensors return the same key iff they are bit-for-bit identical. Returned
    as ``bytes`` so it works as a dict key and feeds ``hashlib`` in ``short()``.

    Implemented via a raw uint8 reinterpret + ``tolist`` so it needs NO numpy
    (the project's GPU venv ships torch without numpy).
    """
    raw = t.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
    return bytes(raw)


def group_by_bits(named_outputs):
    """``[(label, tensor), ...]`` -> ``{bitkey: [labels...]}`` (insertion order)."""
    classes = {}
    for label, out in named_outputs:
        classes.setdefault(bitclass_key(out), []).append(label)
    return classes


def short(bitkey):
    """First 8 hex chars of a bit-class key, for compact tables."""
    return hashlib.sha1(bitkey).hexdigest()[:8]


# --- adversarial inputs (maximize non-associativity divergence) ------------

def adversarial_1d(n, seed=0, device="cuda", dtype=torch.float32):
    """A 1-D vector engineered so reduction ORDER strongly affects the result.

    Recipe: random mantissas * a huge dynamic range (logspace 1e-6 .. 1e6) *
    alternating signs. The wide magnitude spread means partial sums depend
    heavily on pairing order, and alternating signs add catastrophic
    cancellation -- both maximally order-sensitive, so different reduction trees
    (different num_warps / BLOCK) tend to land in different bit-classes.

    Seeded via an explicit CPU generator so the result is reproducible
    regardless of global RNG state.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(n, generator=g, dtype=torch.float32)
    scale = torch.logspace(-6, 6, n, dtype=torch.float32)
    signs = torch.where(torch.arange(n) % 2 == 0,
                        torch.tensor(1.0), torch.tensor(-1.0))
    return (base * scale * signs).to(device=device, dtype=dtype)


def adversarial_2d(rows, cols, seed=0, device="cuda", dtype=torch.float32):
    """``rows`` independently-adversarial rows of width ``cols`` (per-row seed)."""
    out = torch.empty(rows, cols, dtype=torch.float32)
    for r in range(rows):
        out[r] = adversarial_1d(cols, seed=seed + r, device="cpu", dtype=torch.float32)
    return out.to(device=device, dtype=dtype)
