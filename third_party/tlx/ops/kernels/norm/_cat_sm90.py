"""RMSNorm over a concatenation of (optionally RMS-normalized) row parts, for sm90 (H100). FP16/BF16, inference only.

For every row ``b``::

    cat[b]   = concat_i(norm_i(parts[i][b].flatten()))      norm_i(x) = x / sqrt(mean(x^2) + eps) * part_weights[i]
                                                            (identity when part_weights[i] is None)
    out_k[b] = round(cat[b]) / sqrt(mean(cat[b]^2) + eps) * weights[k]

All arithmetic is fp32. ``round`` is the cast to the I/O dtype that sits between the part norms and the final norm;
the final mean is taken over the unrounded ``cat``, as Inductor's fused kernel for the same expression does.

Two kernels, both reading every input element from HBM once and writing every output once:

- ``_rmsnorm_cat_fwd`` (plain Triton, any layout): one program per row holds the whole row in registers, and the
  part norms need no separate pass. The sum of squares of a normalized part comes from the same registers as
  ``sum((x * w)^2) / (mean(x^2) + eps)``. Parts may be any strided ``[B, R, C]`` view and are flattened row-major
  (``tlx.ops.rmsnorm_cat`` passes a 2-D part as ``[B, 1, n]``, or as ``[B, n, 1]`` when ``n`` is not a power of
  two). ``C`` must be a power of two. ``R`` is split into power-of-two segments, at most six in all. A part whose
  ``R`` axis is the contiguous one (``stride(1) == 1``, e.g. the transpose of a GEMM output) is loaded along that
  axis and transposed in registers. A row may hold at most 32768 elements.
- ``_rmsnorm_cat_tlx_fwd`` (TLX, the over-arch layout only, see ``tlx_layout``): persistent and
  warp-specialized, it streams rows through a shared-memory ring while earlier rows are reduced, with the weights
  staged in shared memory once per CTA. It runs every batch of that layout: on H100 it is 1.01-1.9x faster than the
  register kernel at every batch size measured, from 1 to 768 rows.
"""

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._C import _get_dispatch_mode, _TorchDispatchModeKey
from torch._library.triton import wrap_triton
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import statically_known_true
from triton.tools.tensor_descriptor import TensorDescriptor

MAX_SEGMENTS = 6
MAX_NORMALIZED_PARTS = 3
MAX_OUTPUTS = 3
MAX_ROW = 32768


@triton.jit
def _load(ptr, row, s_b, s_r, s_c, R: tl.constexpr, C: tl.constexpr, TRANS: tl.constexpr):
    rr = tl.arange(0, R)
    if TRANS and C == 2:
        x = tl.join(tl.load(ptr + row * s_b + rr * s_r), tl.load(ptr + row * s_b + s_c + rr * s_r))
    elif TRANS:
        cc = tl.arange(0, C)
        x = tl.trans(tl.load(ptr + row * s_b + cc[:, None] * s_c + rr[None, :] * s_r))
    else:
        cc = tl.arange(0, C)
        x = tl.load(ptr + row * s_b + rr[:, None] * s_r + cc[None, :] * s_c)
    return x.to(tl.float32)


@triton.jit
def _pos(R: tl.constexpr, C: tl.constexpr):
    return tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]


@triton.jit
def _pick(v0, v1, v2, G: tl.constexpr):
    if G == 0:
        return v0
    elif G == 1:
        return v1
    else:
        return v2


@triton.jit
def _sums(x, pw, G: tl.constexpr, F: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    """``(sum(x^2), sum((x * pw)^2))`` from one joint reduction; ``(sum(x^2), 0)`` for a part without a norm."""
    if G < 0:
        return tl.sum(tl.sum(x * x, 1), 0), 0.0
    else:
        y = x * tl.load(pw + F + _pos(R, C)).to(tl.float32)
        q, e = tl.split(tl.sum(tl.sum(tl.join(x * x, y * y), 1), 0))
        return q, e


@triton.jit
def _in_group(v, G: tl.constexpr, K: tl.constexpr):
    if G == K:
        return v
    else:
        return 0.0


@triton.jit
def _emit(x, row, rpre, r, pw, G: tl.constexpr, F: tl.constexpr, O: tl.constexpr, w0, w1, w2, o0, o1, o2, s_o,
          R: tl.constexpr, C: tl.constexpr, NOUT: tl.constexpr):
    pos = _pos(R, C)
    if G >= 0:
        x = x / rpre * tl.load(pw + F + pos).to(tl.float32)
    v = x.to(o0.dtype.element_ty).to(tl.float32) / r
    col = O + pos
    tl.store(o0 + row * s_o + col, (v * tl.load(w0 + col).to(tl.float32)).to(o0.dtype.element_ty))
    if NOUT > 1:
        tl.store(o1 + row * s_o + col, (v * tl.load(w1 + col).to(tl.float32)).to(o0.dtype.element_ty))
    if NOUT > 2:
        tl.store(o2 + row * s_o + col, (v * tl.load(w2 + col).to(tl.float32)).to(o0.dtype.element_ty))


# Per segment i: pointer p, row/R/C strides b/r/c, and constexpr rows R, cols C, transposed load T, norm group G
# (-1: none), offset F into the group's weight, output column O.
@triton.jit
def _rmsnorm_cat_fwd(p0, b0, r0, c0, p1, b1, r1, c1, p2, b2, r2, c2, p3, b3, r3, c3, p4, b4, r4, c4, p5, b5, r5, c5,
                     pw0, pw1, pw2, w0, w1, w2, o0, o1, o2, s_o, R0: tl.constexpr, C0: tl.constexpr, T0: tl.constexpr,
                     G0: tl.constexpr, F0: tl.constexpr, O0: tl.constexpr, R1: tl.constexpr, C1: tl.constexpr,
                     T1: tl.constexpr, G1: tl.constexpr, F1: tl.constexpr, O1: tl.constexpr, R2: tl.constexpr,
                     C2: tl.constexpr, T2: tl.constexpr, G2: tl.constexpr, F2: tl.constexpr, O2: tl.constexpr,
                     R3: tl.constexpr, C3: tl.constexpr, T3: tl.constexpr, G3: tl.constexpr, F3: tl.constexpr,
                     O3: tl.constexpr, R4: tl.constexpr, C4: tl.constexpr, T4: tl.constexpr, G4: tl.constexpr,
                     F4: tl.constexpr, O4: tl.constexpr, R5: tl.constexpr, C5: tl.constexpr, T5: tl.constexpr,
                     G5: tl.constexpr, F5: tl.constexpr, O5: tl.constexpr, NG0: tl.constexpr, NG1: tl.constexpr,
                     NG2: tl.constexpr, NSEG: tl.constexpr, NOUT: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x0 = _load(p0, row, b0, r0, c0, R0, C0, T0)
    q0, e0 = _sums(x0, _pick(pw0, pw1, pw2, G0), G0, F0, R0, C0)
    x1 = x0
    x2 = x0
    x3 = x0
    x4 = x0
    x5 = x0
    q1, e1, q2, e2, q3, e3, q4, e4, q5, e5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if NSEG > 1:
        x1 = _load(p1, row, b1, r1, c1, R1, C1, T1)
        q1, e1 = _sums(x1, _pick(pw0, pw1, pw2, G1), G1, F1, R1, C1)
    if NSEG > 2:
        x2 = _load(p2, row, b2, r2, c2, R2, C2, T2)
        q2, e2 = _sums(x2, _pick(pw0, pw1, pw2, G2), G2, F2, R2, C2)
    if NSEG > 3:
        x3 = _load(p3, row, b3, r3, c3, R3, C3, T3)
        q3, e3 = _sums(x3, _pick(pw0, pw1, pw2, G3), G3, F3, R3, C3)
    if NSEG > 4:
        x4 = _load(p4, row, b4, r4, c4, R4, C4, T4)
        q4, e4 = _sums(x4, _pick(pw0, pw1, pw2, G4), G4, F4, R4, C4)
    if NSEG > 5:
        x5 = _load(p5, row, b5, r5, c5, R5, C5, T5)
        q5, e5 = _sums(x5, _pick(pw0, pw1, pw2, G5), G5, F5, R5, C5)
    # Segments past NSEG carry G = -1 and zero sums, so they drop out of every group.
    plain = (_in_group(q0, G0, -1) + _in_group(q1, G1, -1) + _in_group(q2, G2, -1) + _in_group(q3, G3, -1) +
             _in_group(q4, G4, -1) + _in_group(q5, G5, -1))
    sq0 = (_in_group(q0, G0, 0) + _in_group(q1, G1, 0) + _in_group(q2, G2, 0) + _in_group(q3, G3, 0) +
           _in_group(q4, G4, 0) + _in_group(q5, G5, 0))
    sq1 = (_in_group(q0, G0, 1) + _in_group(q1, G1, 1) + _in_group(q2, G2, 1) + _in_group(q3, G3, 1) +
           _in_group(q4, G4, 1) + _in_group(q5, G5, 1))
    sq2 = (_in_group(q0, G0, 2) + _in_group(q1, G1, 2) + _in_group(q2, G2, 2) + _in_group(q3, G3, 2) +
           _in_group(q4, G4, 2) + _in_group(q5, G5, 2))
    se0 = (_in_group(e0, G0, 0) + _in_group(e1, G1, 0) + _in_group(e2, G2, 0) + _in_group(e3, G3, 0) +
           _in_group(e4, G4, 0) + _in_group(e5, G5, 0))
    se1 = (_in_group(e0, G0, 1) + _in_group(e1, G1, 1) + _in_group(e2, G2, 1) + _in_group(e3, G3, 1) +
           _in_group(e4, G4, 1) + _in_group(e5, G5, 1))
    se2 = (_in_group(e0, G0, 2) + _in_group(e1, G1, 2) + _in_group(e2, G2, 2) + _in_group(e3, G3, 2) +
           _in_group(e4, G4, 2) + _in_group(e5, G5, 2))
    rp0 = tl.sqrt_rn(sq0 / NG0 + EPS)
    rp1 = tl.sqrt_rn(sq1 / NG1 + EPS)
    rp2 = tl.sqrt_rn(sq2 / NG2 + EPS)
    r = tl.sqrt_rn((plain + se0 / (rp0 * rp0) + se1 / (rp1 * rp1) + se2 / (rp2 * rp2)) / N + EPS)
    _emit(x0, row, _pick(rp0, rp1, rp2, G0), r, _pick(pw0, pw1, pw2, G0), G0, F0, O0, w0, w1, w2, o0, o1, o2, s_o, R0,
          C0, NOUT)
    if NSEG > 1:
        _emit(x1, row, _pick(rp0, rp1, rp2, G1), r, _pick(pw0, pw1, pw2, G1), G1, F1, O1, w0, w1, w2, o0, o1, o2, s_o,
              R1, C1, NOUT)
    if NSEG > 2:
        _emit(x2, row, _pick(rp0, rp1, rp2, G2), r, _pick(pw0, pw1, pw2, G2), G2, F2, O2, w0, w1, w2, o0, o1, o2, s_o,
              R2, C2, NOUT)
    if NSEG > 3:
        _emit(x3, row, _pick(rp0, rp1, rp2, G3), r, _pick(pw0, pw1, pw2, G3), G3, F3, O3, w0, w1, w2, o0, o1, o2, s_o,
              R3, C3, NOUT)
    if NSEG > 4:
        _emit(x4, row, _pick(rp0, rp1, rp2, G4), r, _pick(pw0, pw1, pw2, G4), G4, F4, O4, w0, w1, w2, o0, o1, o2, s_o,
              R4, C4, NOUT)
    if NSEG > 5:
        _emit(x5, row, _pick(rp0, rp1, rp2, G5), r, _pick(pw0, pw1, pw2, G5), G5, F5, O5, w0, w1, w2, o0, o1, o2, s_o,
              R5, C5, NOUT)


# The over-arch layout of a production ranking model, which the TLX kernel below specializes: float_proj [B, 8192]
# (rows contiguous), fm [B, 32, 128] (rows of 128 contiguous), dots [B, 4352, 2] read transposed (each of the 2 columns
# contiguous).
_DA, _KB, _D, _JC, _JLO = 8192, 32, 128, 4352, 4096
_TLX_CHUNK = 1024
_TLX_WARPS = 8


@triton.jit
def _tlx_ldw(buf, idx, off, n: tl.constexpr):
    return tlx.local_load(tlx.local_slice(buf[idx], [off], [n])).to(tl.float32)


@triton.jit
def _tlx_issue_row(a, c, s_a, s_c, s_ch, row, s, bufA, bufB, bufCl, bufCh, bar, b_desc, DA: tl.constexpr,
                   KB: tl.constexpr, D: tl.constexpr, JC: tl.constexpr, JLO: tl.constexpr):
    tlx.barrier_expect_bytes(bar, 2 * (DA + KB * D + 2 * JC))
    tlx.async_load(a + row * s_a, bufA[s], bulk=True, barrier=bar)
    tlx.async_descriptor_load(b_desc, bufB[s], [0, row.to(tl.int32), 0], bar)
    for h in tl.static_range(2):
        tlx.async_load(c + row * s_c + h * s_ch, bufCl[2 * s + h], bulk=True, barrier=bar)
        tlx.async_load(c + row * s_c + h * s_ch + JLO, bufCh[2 * s + h], bulk=True, barrier=bar)


@triton.jit
def _tlx_issue_weights(bar, w, o: tl.constexpr, sWa, sWo, sWcl, sWch, DA: tl.constexpr, DB: tl.constexpr,
                       JLO: tl.constexpr):
    tlx.async_load(w, sWa[o], bulk=True, barrier=bar)
    tlx.async_load(w + DA, sWo[o], bulk=True, barrier=bar)
    tlx.async_load(w + DA + DB, sWcl[o], bulk=True, barrier=bar)
    tlx.async_load(w + DA + DB + 2 * JLO, sWch[o], bulk=True, barrier=bar)


@triton.jit
def _tlx_consume(cid: tl.constexpr, pid, G, n_rows, bufA, bufB, bufCl, bufCh, sPb, sPcl, sPch, sWa, sWo, sWcl, sWch,
                 full, empty, wbar, o0, o1, s_o, DA: tl.constexpr, KB: tl.constexpr, D: tl.constexpr, JC: tl.constexpr,
                 JLO: tl.constexpr, CH: tl.constexpr, NS: tl.constexpr, NOUT: tl.constexpr, EPS: tl.constexpr):
    DB: tl.constexpr = KB * D
    DC: tl.constexpr = 2 * JC
    NT: tl.constexpr = DA + DB + DC
    JHI: tl.constexpr = JC - JLO
    RB: tl.constexpr = CH // D
    dt = o0.dtype.element_ty
    jc = tl.arange(0, CH)
    ic = tl.arange(0, 2 * CH)
    ih = tl.arange(0, 2 * JHI)
    tlx.barrier_wait(wbar[0], 0)
    for k in range(cid, n_rows, 2):
        s = k % NS
        tlx.barrier_wait(full[s], (k // NS) & 1)
        row = (pid + k * G).to(tl.int64)
        # pass 1: sums of squares (and of the part-normalized squares) from SMEM
        acc_a = tl.zeros([CH], tl.float32)
        for t in tl.static_range(DA // CH):
            x = tlx.local_load(tlx.local_slice(bufA[s], [t * CH], [CH])).to(tl.float32)
            acc_a += x * x
        acc_b1 = tl.zeros([CH], tl.float32)
        acc_b2 = tl.zeros([CH], tl.float32)
        for t in tl.static_range(DB // CH):
            x = tl.reshape(tlx.local_load(tlx.local_slice(bufB[s], [t * RB, 0, 0], [RB, 1, D])), [CH]).to(tl.float32)
            xw = x * _tlx_ldw(sPb, 0, t * CH, CH)
            acc_b1 += x * x
            acc_b2 += xw * xw
        acc_c1 = tl.zeros([2 * CH], tl.float32)
        acc_c2 = tl.zeros([2 * CH], tl.float32)
        for t in tl.static_range(JLO // CH):
            c0 = tlx.local_load(tlx.local_slice(bufCl[2 * s], [t * CH], [CH])).to(tl.float32)
            c1 = tlx.local_load(tlx.local_slice(bufCl[2 * s + 1], [t * CH], [CH])).to(tl.float32)
            x = tl.reshape(tl.join(c0, c1), [2 * CH])
            xw = x * _tlx_ldw(sPcl, 0, 2 * t * CH, 2 * CH)
            acc_c1 += x * x
            acc_c2 += xw * xw
        x = tl.reshape(tl.join(tlx.local_load(bufCh[2 * s]), tlx.local_load(bufCh[2 * s + 1])), [2 * JHI])
        x = x.to(tl.float32)
        xw = x * _tlx_ldw(sPch, 0, 0, 2 * JHI)
        rb = tl.sqrt_rn(tl.sum(acc_b1, 0) / DB + EPS)
        rc = tl.sqrt_rn((tl.sum(acc_c1, 0) + tl.sum(x * x, 0)) / DC + EPS)
        r = tl.sqrt_rn((tl.sum(acc_a, 0) + tl.sum(acc_b2, 0) / (rb * rb) + (tl.sum(acc_c2, 0) + tl.sum(xw * xw, 0)) /
                        (rc * rc)) / NT + EPS)
        if k < 2:
            tlx.barrier_wait(wbar[1], 0)
        # pass 2: outputs, the normalized parts rounded to the I/O dtype first
        base0 = o0 + row * s_o
        base1 = o1 + row * s_o
        for t in tl.static_range(DA // CH):
            col = t * CH + jc
            v = tlx.local_load(tlx.local_slice(bufA[s], [t * CH], [CH])).to(tl.float32) / r
            tl.store(base0 + col, (v * _tlx_ldw(sWa, 0, t * CH, CH)).to(dt))
            if NOUT > 1:
                tl.store(base1 + col, (v * _tlx_ldw(sWa, 1, t * CH, CH)).to(dt))
        for t in tl.static_range(DB // CH):
            x = tl.reshape(tlx.local_load(tlx.local_slice(bufB[s], [t * RB, 0, 0], [RB, 1, D])), [CH]).to(tl.float32)
            v = (x / rb * _tlx_ldw(sPb, 0, t * CH, CH)).to(dt).to(tl.float32) / r
            col = DA + t * CH + jc
            tl.store(base0 + col, (v * _tlx_ldw(sWo, 0, t * CH, CH)).to(dt))
            if NOUT > 1:
                tl.store(base1 + col, (v * _tlx_ldw(sWo, 1, t * CH, CH)).to(dt))
        for t in tl.static_range(JLO // CH):
            c0 = tlx.local_load(tlx.local_slice(bufCl[2 * s], [t * CH], [CH])).to(tl.float32)
            c1 = tlx.local_load(tlx.local_slice(bufCl[2 * s + 1], [t * CH], [CH])).to(tl.float32)
            x = tl.reshape(tl.join(c0, c1), [2 * CH]) / rc
            v = (x * _tlx_ldw(sPcl, 0, 2 * t * CH, 2 * CH)).to(dt).to(tl.float32) / r
            col = DA + DB + 2 * t * CH + ic
            tl.store(base0 + col, (v * _tlx_ldw(sWcl, 0, 2 * t * CH, 2 * CH)).to(dt))
            if NOUT > 1:
                tl.store(base1 + col, (v * _tlx_ldw(sWcl, 1, 2 * t * CH, 2 * CH)).to(dt))
        x = tl.reshape(tl.join(tlx.local_load(bufCh[2 * s]), tlx.local_load(bufCh[2 * s + 1])), [2 * JHI])
        x = x.to(tl.float32) / rc
        v = (x * _tlx_ldw(sPch, 0, 0, 2 * JHI)).to(dt).to(tl.float32) / r
        col = DA + DB + 2 * JLO + ih
        tl.store(base0 + col, (v * _tlx_ldw(sWch, 0, 0, 2 * JHI)).to(dt))
        if NOUT > 1:
            tl.store(base1 + col, (v * _tlx_ldw(sWch, 1, 0, 2 * JHI)).to(dt))
        tlx.barrier_arrive(empty[s])


@triton.jit
def _rmsnorm_cat_tlx_fwd(a, s_a, b_desc, c, s_c, s_ch, pwb, pwc, w0, w1, o0, o1, s_o, B, DA: tl.constexpr,
                         KB: tl.constexpr, D: tl.constexpr, JC: tl.constexpr, JLO: tl.constexpr, CH: tl.constexpr,
                         NOUT: tl.constexpr, EPS: tl.constexpr):
    """Persistent, warp-specialized: one producer warp streams whole rows into a two-slot SMEM ring (cp.async.bulk
    for the contiguous parts, TMA for fm; a full and an empty mbarrier per slot) and two consumer warp groups reduce
    alternate rows from SMEM, with every weight staged in SMEM once per CTA."""
    NS: tl.constexpr = 2
    DB: tl.constexpr = KB * D
    DC: tl.constexpr = 2 * JC
    JHI: tl.constexpr = JC - JLO
    dt = o0.dtype.element_ty
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    bufA = tlx.local_alloc((DA, ), dt, NS)
    bufB = tlx.local_alloc((KB, 1, D), dt, NS)
    bufCl = tlx.local_alloc((JLO, ), dt, 2 * NS)
    bufCh = tlx.local_alloc((JHI, ), dt, 2 * NS)
    sPb = tlx.local_alloc((DB, ), dt, 1)
    sPcl = tlx.local_alloc((2 * JLO, ), dt, 1)
    sPch = tlx.local_alloc((2 * JHI, ), dt, 1)
    sWa = tlx.local_alloc((DA, ), dt, NOUT)
    sWo = tlx.local_alloc((DB, ), dt, NOUT)
    sWcl = tlx.local_alloc((2 * JLO, ), dt, NOUT)
    sWch = tlx.local_alloc((2 * JHI, ), dt, NOUT)
    full = tlx.alloc_barriers(NS, arrive_count=1)
    empty = tlx.alloc_barriers(NS, arrive_count=1)
    wbar = tlx.alloc_barriers(2, arrive_count=1)
    n_rows = (B - pid + G - 1) // G
    with tlx.async_tasks():
        with tlx.async_task("default", replicate=2):
            _tlx_consume(tlx.async_task_replica_id(), pid, G, n_rows, bufA, bufB, bufCl, bufCh, sPb, sPcl, sPch, sWa,
                         sWo, sWcl, sWch, full, empty, wbar, o0, o1, s_o, DA, KB, D, JC, JLO, CH, NS, NOUT, EPS)
        with tlx.async_task(num_warps=1):
            # part-norm weights first; the output weights after the first row
            tlx.barrier_expect_bytes(wbar[0], 2 * (DB + DC))
            tlx.async_load(pwb, sPb[0], bulk=True, barrier=wbar[0])
            tlx.async_load(pwc, sPcl[0], bulk=True, barrier=wbar[0])
            tlx.async_load(pwc + 2 * JLO, sPch[0], bulk=True, barrier=wbar[0])
            for k in range(0, n_rows):
                s = k % NS
                tlx.barrier_wait(empty[s], ((k // NS) & 1) ^ 1)
                _tlx_issue_row(a, c, s_a, s_c, s_ch, (pid + k * G).to(tl.int64), s, bufA, bufB, bufCl, bufCh, full[s],
                               b_desc, DA, KB, D, JC, JLO)
                if k == 0:
                    tlx.barrier_expect_bytes(wbar[1], 2 * NOUT * (DA + DB + DC))
                    _tlx_issue_weights(wbar[1], w0, 0, sWa, sWo, sWcl, sWch, DA, DB, JLO)
                    if NOUT > 1:
                        _tlx_issue_weights(wbar[1], w1, 1, sWa, sWo, sWcl, sWch, DA, DB, JLO)


def _known(cond):
    return statically_known_true(cond)


def _aligned(x, dims):
    """16-byte aligned storage offset and strides ``dims`` (what cp.async.bulk and TMA need), known without guards."""
    return all(_known(v % 8 == 0) for v in (x.storage_offset(), *(x.stride(d) for d in dims)))


def tlx_layout(parts, part_weights, weights):
    """Whether the TLX kernel takes these inputs: the over-arch layout, 16-bit types, one or two outputs.

    Decided from metadata alone (no guards on the batch size), so a traced graph takes the same decision for every
    batch.
    """
    if len(parts) != 3 or not 1 <= len(weights) <= 2:
        return False
    if part_weights[0] is not None or part_weights[1] is None or part_weights[2] is None:
        return False
    a, b, c = parts
    dtype = a.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    shapes = [(1, _DA), (_KB, _D), (_JC, 2)]
    if any(not (_known(p.shape[1] == r) and _known(p.shape[2] == k)) for p, (r, k) in zip(parts, shapes)):
        return False
    pc = c.permute(0, 2, 1)
    if not (_known(a.stride(2) == 1) and _known(b.stride(2) == 1) and _known(pc.stride(2) == 1)):
        return False
    vecs = [part_weights[1], part_weights[2], *weights]
    if any(v.dtype != dtype or not _known(v.stride(0) == 1) or not _known(v.storage_offset() % 8 == 0) for v in vecs):
        return False
    if not (_aligned(a, [0]) and _aligned(b, [0, 1]) and _aligned(pc, [0, 1])):
        return False
    if not _tracing() and any(t.data_ptr() % 16 for t in (a, b, c, *vecs)):
        return False
    return True


def _launch_tlx(parts, part_weights, weights, outs, eps, grid, traced):
    a, b, c = parts
    b_desc = TensorDescriptor.from_tensor(b.permute(1, 0, 2), [_KB, 1, _D])
    ws = list(weights) + [weights[0]] * (2 - len(weights))
    os = outs + [outs[0]] * (2 - len(outs))
    kernel = wrap_triton(_rmsnorm_cat_tlx_fwd) if traced else _rmsnorm_cat_tlx_fwd
    kernel[(grid, )](a, a.stride(0), b_desc, c, c.stride(0), c.stride(2), part_weights[1], part_weights[2], *ws, *os,
                     outs[0].stride(0), a.shape[0], DA=_DA, KB=_KB, D=_D, JC=_JC, JLO=_JLO, CH=_TLX_CHUNK,
                     NOUT=len(weights), EPS=eps, num_warps=_TLX_WARPS)


def _pow2_pieces(n):
    """``n`` as consecutive power-of-two pieces, largest first: 4352 -> (0, 4096), (4096, 256)."""
    pieces, start = [], 0
    while n:
        size = 1 << (n.bit_length() - 1)
        pieces.append((start, size))
        start, n = start + size, n - size
    return pieces


def segments(parts, part_weights):
    """The launch's segment table; raises ``ValueError`` when the parts need more than the kernel holds."""
    segs, group_sizes, col = [], [], 0
    for x, pw in zip(parts, part_weights):
        _, rows, cols = x.shape
        if cols & (cols - 1):
            raise ValueError(f"the last dim of every part must be a power of two; got {cols}")
        group = -1
        if pw is not None:
            group = len(group_sizes)
            group_sizes.append(rows * cols)
        trans = x.stride(1) == 1 and cols > 1 and x.stride(2) != 1
        for start, size in _pow2_pieces(rows):
            segs.append(
                dict(x=x[:, start:start + size], R=size, C=cols, T=trans, G=group, F=start * cols,
                     O=col + start * cols))
        col += rows * cols
    if len(segs) > MAX_SEGMENTS:
        raise ValueError(f"the parts split into {len(segs)} power-of-two segments; at most {MAX_SEGMENTS} fit")
    if len(group_sizes) > MAX_NORMALIZED_PARTS:
        raise ValueError(f"at most {MAX_NORMALIZED_PARTS} parts may be normalized; got {len(group_sizes)}")
    if col > MAX_ROW:
        raise ValueError(f"a row of {col} elements does not fit in registers; at most {MAX_ROW}")
    return segs, group_sizes, col


def _num_warps(n):
    # At most 128 row elements per thread up to 16K-element rows (more spills); four warps keep two 20K-element rows in
    # flight per SM, and the largest rows need eight.
    return 8 if n > 24576 else max(1, min(4, triton.next_power_of_2(n) // 4096))


def _tracing():
    # Only the tracing modes need the launch as a graph node; other modes (profilers, flop counters) see eager tensors.
    return torch.compiler.is_compiling() or any(
        _get_dispatch_mode(key) is not None
        for key in (_TorchDispatchModeKey.PROXY, _TorchDispatchModeKey.FUNCTIONAL, _TorchDispatchModeKey.FAKE))


@functools.cache
def _num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def rmsnorm_cat(parts, weights, part_weights, eps):
    """Launch the forward; inputs are validated by ``tlx.ops.rmsnorm_cat``. Returns one ``[B, N]`` per weight.

    ``parts`` are ``[B, R, C]`` views. Under tracing the launch goes through ``wrap_triton`` so ``torch.compile`` and
    AOTInductor can capture it; eager calls launch directly, because the higher-order-op path costs about half a
    millisecond of host time for this many arguments.

    The over-arch layout (``tlx_layout``) runs the TLX kernel with at most one CTA per SM; every other layout runs the
    register kernel. Each row is computed by one program or consumer group in a fixed order, so a row's outputs do not
    depend on the batch it is in.
    """
    segs, group_sizes, n = segments(parts, part_weights)
    batch = parts[0].shape[0]
    outs = [torch.empty(batch, n, device=parts[0].device, dtype=parts[0].dtype) for _ in weights]
    if batch == 0 or isinstance(parts[0], FakeTensor):
        return outs
    traced = _tracing()
    if tlx_layout(parts, part_weights, weights):
        grid = torch.sym_min(batch, _num_sms(parts[0].device.index))
        _launch_tlx(parts, part_weights, weights, outs, eps, grid, traced)
        return outs
    args, meta = [], {}
    for i in range(MAX_SEGMENTS):
        seg = segs[i] if i < len(segs) else dict(segs[0], G=-1)
        x = seg["x"]
        args += [x, x.stride(0), x.stride(1), x.stride(2)]
        meta.update({f"{k}{i}": seg[k] for k in "RCTGFO"})
    pws = [pw for pw in part_weights if pw is not None]
    pws += [weights[0]] * (MAX_NORMALIZED_PARTS - len(pws))
    sizes = group_sizes + [1] * (MAX_NORMALIZED_PARTS - len(group_sizes))
    ws = list(weights) + [weights[0]] * (MAX_OUTPUTS - len(weights))
    os = outs + [outs[0]] * (MAX_OUTPUTS - len(outs))
    kernel = wrap_triton(_rmsnorm_cat_fwd) if traced else _rmsnorm_cat_fwd
    kernel[(batch, )](*args, *pws, *ws, *os, outs[0].stride(0), NG0=sizes[0], NG1=sizes[1], NG2=sizes[2],
                      NSEG=len(segs), NOUT=len(weights), N=n, EPS=eps, num_warps=_num_warps(n), **meta)
    return outs
