from __future__ import annotations

from typing import NamedTuple

from ..._shape_suites import FocusRegistry, FocusSuite

PAGE = 16
EPS = 1e-5
Q_SCALE = 5.4730064863875
ROPE_BASE = 500000.0
MAX_POSITION = 32768


class QKNormRopeShape(NamedTuple):
    """``tokens`` rows of a fused QKV projection with ``q_heads`` and ``kv_heads`` heads of ``head_dim``. ``rope``:
    RoPE on (else none); ``remap``: slots go through a second-level slot table; ``q_fp8``: q returned in FP8."""
    tokens: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rope: bool
    remap: bool
    q_fp8: bool


SYNTHETIC: tuple[QKNormRopeShape, ...] = (
    QKNormRopeShape(1, 8, 1, 128, True, False, False),
    QKNormRopeShape(33, 32, 8, 64, True, True, True),
    QKNormRopeShape(300, 4, 4, 256, False, False, True),
    QKNormRopeShape(129, 64, 2, 16, True, False, False),
)

# One rank of a GQA attention layer with 16 query and 2 KV heads of 64, page 16: sliding-window layers with RoPE whose
# cache slots go through the window pool's slot table, and full-attention layers without RoPE.
_LAYERS = ((True, True), (False, False))
# Decode batches return an FP8 q for FP8 attention.
GFX950_1 = FocusSuite(
    name="gfx950_1", op="qk_norm_rope_kv_cache",
    shapes=tuple(QKNormRopeShape(t, 16, 2, 64, r, m, True) for t in (1, 4, 8, 16, 32, 64) for r, m in _LAYERS))
# Prefill chunks return a BF16 q.
GFX950_2 = FocusSuite(
    name="gfx950_2", op="qk_norm_rope_kv_cache",
    shapes=tuple(QKNormRopeShape(t, 16, 2, 64, r, m, False) for t in (2048, 8192, 16384) for r, m in _LAYERS))
GFX950_ALL = FocusSuite(name="gfx950_all", op="qk_norm_rope_kv_cache", includes=("gfx950_1", "gfx950_2"))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("qk_norm_rope_kv_cache", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(shape: QKNormRopeShape) -> str:
    flags = ",".join(name for name, on in (("rope", shape.rope), ("remap", shape.remap), ("q_fp8", shape.q_fp8)) if on)
    return (f"(({shape.tokens}, {shape.q_heads}, {shape.kv_heads}, {shape.head_dim}), "
            f"{{'flags': '{flags or 'none'}'}})")


def cos_sin_table(head_dim: int, max_position: int = MAX_POSITION, base: float = ROPE_BASE, device="cuda"):
    """FP32 ``(max_position, head_dim)`` table, ``cos`` in the first half of a row and ``sin`` in the second."""
    import torch

    inv_freq = 1.0 / (base**(torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float32, device=device), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def inputs(shape: QKNormRopeShape, seed: int = 0, device="cuda", k_scale=1.0, v_scale=1.0):
    """Keyword arguments of the op. ``q``, ``k`` and ``v`` are column slices of one projection output that also holds a
    gate of ``q``'s width; decode batches (q in FP8) take one token of each of ``tokens`` sequences at any position and
    slot, prefill takes consecutive positions and consecutive slots in scattered pages."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    T, HQ, HK, D = shape.tokens, shape.q_heads, shape.kv_heads, shape.head_dim
    qkv = (torch.randn(T, 2 * HQ * D + 2 * HK * D, device=device, generator=gen) * 2).to(torch.bfloat16)
    q, k, v = qkv[:, :HQ * D], qkv[:, HQ * D:HQ * D + HK * D], qkv[:, HQ * D + HK * D:HQ * D + 2 * HK * D]
    # Page 0 is the padding page, as in paged pools: no token is written there.
    pages = 2 * (T + PAGE - 1) // PAGE + 4
    page_ids = torch.randperm(pages - 1, device=device, generator=gen) + 1
    if shape.q_fp8:
        positions = torch.randint(0, MAX_POSITION, (T, ), device=device, generator=gen)
        slot_mapping = torch.randperm((pages - 1) * PAGE, device=device, generator=gen)[:T] + PAGE
    else:
        positions = torch.arange(T, device=device)
        slot_mapping = (page_ids[:, None] * PAGE + torch.arange(PAGE, device=device)).flatten()[:T].contiguous()
    if T > 2:
        slot_mapping[T // 2] = -1  # a padding token
    slot_remap = None
    if shape.remap:
        # A page-granular table into a pool as large as the cache; the slots of one page map to -1.
        remap_pages = torch.randperm(pages - 1, device=device, generator=gen) + 1
        slot_remap = (remap_pages[:, None] * PAGE + torch.arange(PAGE, device=device)).flatten()
        slot_remap = torch.cat((slot_remap[:PAGE], slot_remap))  # full slots of page 0 map like page 1
        slot_remap[slot_remap >= (pages - 1) * PAGE] = -1
    k_cache = torch.zeros(pages, PAGE, HK, D, device=device, dtype=torch.float8_e4m3fn)
    scale = lambda value: torch.tensor([value], device=device)  # noqa: E731
    return dict(q=q, k=k, v=v, k_cache=k_cache, v_cache=torch.zeros_like(k_cache), slot_mapping=slot_mapping, eps=EPS,
                q_scale=Q_SCALE, positions=positions if shape.rope else None,
                cos_sin_cache=cos_sin_table(D, device=device) if shape.rope else None, slot_remap=slot_remap,
                k_scale=scale(k_scale), v_scale=scale(v_scale), q_out_scale=scale(k_scale) if shape.q_fp8 else None)


def _fma(a, b, c):
    """FP32 ``a * b + c`` with one rounding (exact for the BF16 products here)."""
    return (a.double() * b.double() + c.double()).float()


def head_sum_squares(x, lanes_per_row: int):
    """Sum of squares of the last dim of BF16 ``x`` in the op's order at 16 query and 2 KV heads of 64."""
    x = x.float()
    if lanes_per_row == 16:  # q: four elements per lane, 16 lanes per head
        x = x.unflatten(-1, (16, 4))
        p = _fma(x[..., 0], x[..., 0], x[..., 2] * x[..., 2]) + _fma(x[..., 1], x[..., 1], x[..., 3] * x[..., 3])
        for half in (8, 4, 2, 1):  # butterfly over the 16 lanes
            p = p[..., :half] + p[..., half:2 * half]
        return p[..., 0]
    # k: one element per lane, a head per wave. Each 16-lane row adds lane j - 8 into lane j with an FMA, then shifts
    # by 4, 2 and 1; the four row sums add pairwise.
    x = x.unflatten(-1, (4, 16))
    a = _fma(x[..., 8:], x[..., 8:], x[..., :8] * x[..., :8])
    b = a[..., 4:] + a[..., :4]
    c = b[..., 2:] + b[..., :2]
    r = c[..., 1] + c[..., 0]
    return (r[..., 3] + r[..., 2]) + (r[..., 1] + r[..., 0])


def _rope(x, cos, sin, rope_bf16):
    import torch

    a, b = x[..., 0::2], x[..., 1::2]
    if rope_bf16:
        c, s = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        out = torch.stack((a * c - b * s, b * c + a * s), -1)
    else:
        a, b = a.float(), b.float()
        out = torch.stack((a * cos - b * sin, b * cos + a * sin), -1).to(torch.bfloat16)
    return out.flatten(-2)


def _fp8(x, nan_to_min=False):
    import torch

    if nan_to_min:  # max(NaN, -448) is -448
        x = torch.where(x.isnan(), -448.0, x)
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def reference(q, k, v, k_cache, v_cache, slot_mapping, eps, q_scale, positions=None, cos_sin_cache=None, rope_bf16=True,
              slot_remap=None, k_scale=None, v_scale=None, q_out_scale=None, rsqrt=None, sum_squares=None,
              inplace=False):
    """Eager torch version of the op, on clones of the caches unless ``inplace``; returns ``(q_out, k_cache,
    v_cache)``.

    ``sum_squares(x, lanes_per_row)`` and ``rsqrt`` default to ``head_sum_squares`` and ``torch.rsqrt``; with the
    op's own order and the hardware reciprocal square root, the result is the op's bit for bit at 16 query and 2 KV
    heads of 64. ``q_out_scale`` must be a power of two for the FP8 q to be exact (the op uses the hardware
    reciprocal)."""
    import torch

    rsqrt = rsqrt or torch.rsqrt
    T, D = q.shape[0], k_cache.shape[3]
    HQ, HK = q.shape[1] // D, k_cache.shape[2]
    qh, kh = q.view(T, HQ, D), k.view(T, HK, D)
    if sum_squares is None:
        sum_squares = lambda x, lanes: (x.float() * x.float()).sum(-1)  # noqa: E731
    rq = rsqrt(sum_squares(qh, 16) / D + eps)[..., None]
    rk = rsqrt(sum_squares(kh, 64) / D + eps)[..., None]
    qn = ((qh.float() * rq).to(torch.bfloat16).float() * q_scale).to(torch.bfloat16)
    kn = (kh.float() * rk).to(torch.bfloat16)
    if positions is not None:
        cs = cos_sin_cache[positions.long()][:, None, :]
        qn = _rope(qn, cs[..., :D // 2], cs[..., D // 2:], rope_bf16)
        kn = _rope(kn, cs[..., :D // 2], cs[..., D // 2:], rope_bf16)
    qn = qn.reshape(T, HQ * D)
    if q_out_scale is not None:
        qn = _fp8(qn.float() * (1.0 / q_out_scale.float()), nan_to_min=True)
    kf, vf = kn.float(), v.view(T, HK, D).float()
    if k_scale is not None:
        kf, vf = kf / k_scale, vf / v_scale
    if not inplace:
        k_cache, v_cache = k_cache.clone(), v_cache.clone()
    slots = slot_mapping.long()
    if slot_remap is not None:
        slots = torch.where(slots >= 0, slot_remap.long()[slots.clamp_min(0)], slots)
    # Skipped tokens write back the old content of slot 0 (the inputs' padding page), so the scatter needs no
    # host sync and can be captured in a graph.
    keep = (slots >= 0)[:, None, None]
    page = k_cache.shape[1]
    s = torch.where(slots >= 0, slots, 0)
    for cache, x in ((k_cache, kf), (v_cache, vf)):
        old = cache[s // page, s % page].view(torch.uint8)
        cache[s // page, s % page] = torch.where(keep, _fp8(x).view(torch.uint8), old).view(torch.float8_e4m3fn)
    return qn, k_cache, v_cache
