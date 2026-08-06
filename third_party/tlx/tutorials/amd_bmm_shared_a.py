"""Shared-A batched GEMM (BMM) for gfx950 / CDNA4 — ROW-major B (shared-LHS).

Companion to ``amd_bmm.py``. Both files are shared-A; what differs is B's memory
layout: ``amd_bmm.py`` takes COLUMN-major B (``stride_bk == 1``), this file takes
ROW-major B (``stride_bn == 1``), the standard torch.bmm / inductor layout.

LAYOUT:
  * A: shared-A — one (M, K) matrix reused across the whole batch,
    ``a.stride(0) == 0`` (mat1 batch-stride 0). Benchmark against shared-A, not
    distinct-A: rocBLAS reads shared-A once and keeps it L2-resident, so a
    distinct-A comparison flatters TLX.
  * B: (B, K, N) ROW-major (N-contiguous, ``stride_bn == 1``).
  * C: (B, M, N) row-major.

CONFIG: num_warps=8, matrix_instr_nonkdim=32 (32x32 MFMA). nw=8 does not compile
with the column-major kernel's swizzle + 4-warp split store, which is why the two
layouts are separate files rather than one parameterised kernel.

Two load paths, selected by K alignment (``K % BLOCK_K == 0`` -> aligned A rows,
no K-tail):
  * aligned -> direct-to-LDS (``buffer_load_to_local``) + swizzled LDS.
  * odd K   -> register path (``tl.load`` -> ``tlx.local_store``), masked K-tail.
    Required because odd K gives 2-byte-aligned rows, where direct-to-LDS is
    illegal on CDNA4.
"""
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.testing import do_bench

BLOCK_N = 256
BLOCK_K = 32
NUM_XCDS = 8
NB = 3


def _swz(shape, cd):
    def basis(d, i):
        return [1 << i, 0] if d == 0 else [0, 1 << i]

    fd = 1 - cd
    cb = int(shape[cd]).bit_length() - 1
    fb = int(shape[fd]).bit_length() - 1
    return ([basis(cd, i) for i in range(cb)] + [basis(fd, i)
                                                 for i in range(4, fb)] + [basis(fd, i) for i in range(min(4, fb))])


@triton.jit
def _chip(pid, nt, nx: tl.constexpr, cs: tl.constexpr):
    """L2 XCD-chunk remap: keep a batch's MN-tiles on one XCD (B stays hot in L2)."""
    al = (nt // (nx * cs)) * (nx * cs)
    if pid >= al:
        return pid
    x = pid % nx
    lp = pid // nx
    return (lp // cs) * nx * cs + x * cs + (lp % cs)


@triton.jit
def _bmm_direct(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, AB: tl.constexpr, BB: tl.constexpr,
                NUM_XCDS: tl.constexpr, GMN: tl.constexpr, NT: tl.constexpr, NB: tl.constexpr):
    """Aligned rows, no K-tail (K % BLOCK_K == 0): direct-to-LDS + swizzled LDS."""
    npn = tl.cdiv(N, BN)
    pidf = _chip(tl.program_id(0), NT, NUM_XCDS, GMN)
    bid = pidf // GMN; pid = pidf % GMN; pm = pid // npn; pn = pid % npn
    ash: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], AB, [BM, BK])
    bsh: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], BB, [BK, BN])
    sA = tlx.local_alloc((BM, BK), tl.float16, NB, layout=ash)
    sB = tlx.local_alloc((BK, BN), tl.float16, NB, layout=bsh)
    om = (pm * BM + tl.arange(0, BM)) % M; on = (pn * BN + tl.arange(0, BN)) % N; ok = tl.arange(0, BK)
    a_ptr = a_ptr + bid.to(tl.int64) * sab; b_ptr = b_ptr + bid.to(tl.int64) * sbb
    ao = om[:, None] * sam; bo = on[None, :] * sbn; KI = tl.cdiv(K, BK)
    for i in tl.range(0, NB, loop_unroll_factor=NB):
        kk = i * BK
        tlx.buffer_load_to_local(tlx.local_view(sA, i), a_ptr, ao + (kk + ok[None, :]) * sak)
        tlx.buffer_load_to_local(tlx.local_view(sB, i), b_ptr, (kk + ok[:, None]) * sbk + bo)
        tlx.async_load_commit_group()
    tlx.async_load_wait_group(NB - 2)
    a = tlx.local_load(tlx.local_view(sA, 0)); b = tlx.local_load(tlx.local_view(sB, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, KI - NB):
        cur = (k + 1) % NB; pf = k % NB; kp = (k + NB) * BK
        acc = tl.dot(a, b, acc)
        tlx.buffer_load_to_local(tlx.local_view(sA, pf), a_ptr, ao + (kp + ok[None, :]) * sak)
        tlx.buffer_load_to_local(tlx.local_view(sB, pf), b_ptr, (kp + ok[:, None]) * sbk + bo)
        tlx.async_load_commit_group(); tlx.async_load_wait_group(NB - 2)
        a = tlx.local_load(tlx.local_view(sA, cur)); b = tlx.local_load(tlx.local_view(sB, cur))
    acc = tl.dot(a, b, acc)
    tlx.async_load_wait_group(0)
    for i in tl.range(0, NB - 1, loop_unroll_factor=NB - 1):
        bf = (KI - (NB - 1) + i) % NB
        acc = tl.dot(tlx.local_load(tlx.local_view(sA, bf)), tlx.local_load(tlx.local_view(sB, bf)), acc)
    et = c_ptr.dtype.element_ty; cb = c_ptr + bid.to(tl.int64) * scb
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
    tl.store(cb + scm * rm[:, None] + scn * rn[None, :], acc.to(et), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _bmm_register(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  NUM_XCDS: tl.constexpr, GMN: tl.constexpr, NT: tl.constexpr, NB: tl.constexpr):
    """Odd / unaligned K: register path (tl.load -> local_store), masked K-tail."""
    npn = tl.cdiv(N, BN)
    pidf = _chip(tl.program_id(0), NT, NUM_XCDS, GMN)
    bid = pidf // GMN; pid = pidf % GMN; pm = pid // npn; pn = pid % npn
    sA = tlx.local_alloc((BM, BK), tl.float16, NB); sB = tlx.local_alloc((BK, BN), tl.float16, NB)
    om = (pm * BM + tl.arange(0, BM)) % M; on = (pn * BN + tl.arange(0, BN)) % N; ok = tl.arange(0, BK)
    a_ptr = a_ptr + bid.to(tl.int64) * sab; b_ptr = b_ptr + bid.to(tl.int64) * sbb
    ao = om[:, None] * sam; bo = on[None, :] * sbn; KI = tl.cdiv(K, BK)
    for i in tl.range(0, NB, loop_unroll_factor=NB):
        kk = i * BK; km = (kk + ok) < K
        ar = tl.load(a_ptr + ao + (kk + ok[None, :]) * sak, mask=km[None, :], other=0.0)
        br = tl.load(b_ptr + (kk + ok[:, None]) * sbk + bo, mask=km[:, None], other=0.0)
        tlx.local_store(tlx.local_view(sA, i), ar); tlx.local_store(tlx.local_view(sB, i), br)
    tl.debug_barrier()
    a = tlx.local_load(tlx.local_view(sA, 0)); b = tlx.local_load(tlx.local_view(sB, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, KI - NB):
        cur = (k + 1) % NB; pf = k % NB; kp = (k + NB) * BK
        acc = tl.dot(a, b, acc)
        km = (kp + ok) < K
        ar = tl.load(a_ptr + ao + (kp + ok[None, :]) * sak, mask=km[None, :], other=0.0)
        br = tl.load(b_ptr + (kp + ok[:, None]) * sbk + bo, mask=km[:, None], other=0.0)
        tlx.local_store(tlx.local_view(sA, pf), ar); tlx.local_store(tlx.local_view(sB, pf), br)
        tl.debug_barrier()
        a = tlx.local_load(tlx.local_view(sA, cur)); b = tlx.local_load(tlx.local_view(sB, cur))
    acc = tl.dot(a, b, acc)
    for i in tl.range(0, NB - 1, loop_unroll_factor=NB - 1):
        bf = (KI - (NB - 1) + i) % NB
        acc = tl.dot(tlx.local_load(tlx.local_view(sA, bf)), tlx.local_load(tlx.local_view(sB, bf)), acc)
    et = c_ptr.dtype.element_ty; cb = c_ptr + bid.to(tl.int64) * scb
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
    tl.store(cb + scm * rm[:, None] + scn * rn[None, :], acc.to(et), mask=(rm[:, None] < M) & (rn[None, :] < N))


def bmm(a, b):
    """C = A @ B, shared-A, ROW-major B (stride_bn == 1). nw=8, mfma=32."""
    Bs, M, K = a.shape
    N = b.shape[-1]
    bm = 64 if M <= 64 else 128
    KI = triton.cdiv(K, BLOCK_K)
    assert KI >= 2, f"K must span >= 2 BLOCK_K={BLOCK_K} tiles for the pipeline, got K={K}"
    # The 2-stage pipeline needs 2 <= nb <= KI: the prologue unconditionally issues
    # nb loads and the drain indexes (KI - (nb - 1) + i) % nb, so nb > KI would
    # over-read past K and re-accumulate tile 0. Requiring KI >= 2 makes
    # min(NB, KI) >= 2 on its own -- a max(2, ...) here would defeat the clamp.
    nb = min(NB, KI)
    GMN = triton.cdiv(M, bm) * triton.cdiv(N, BLOCK_N)
    NT = Bs * GMN
    c = torch.empty((Bs, M, N), device=a.device, dtype=a.dtype)
    attrs = (("amdgpu-agpr-alloc", "0,0"), )
    common = dict(num_warps=8, num_stages=1, matrix_instr_nonkdim=32, llvm_fn_attrs=attrs)
    st = (a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2), c.stride(0), c.stride(1),
          c.stride(2))
    # K % BLOCK_K, not K % 8: the direct path does no K-tail masking on
    # buffer_load_to_local, so a K that is 8-aligned but not BLOCK_K-aligned
    # (e.g. 264) reads past the K extent. Matches the guard in amd_bmm.py.
    if K % BLOCK_K == 0:  # 16-byte-aligned A rows, no K-tail -> direct-to-LDS (wins)
        AB = tuple(tuple(x) for x in _swz([bm, BLOCK_K], 1))
        BB = tuple(tuple(x) for x in _swz([BLOCK_K, BLOCK_N], 1))
        _bmm_direct[(NT, )](a, b, c, M, N, K, *st, BM=bm, BN=BLOCK_N, BK=BLOCK_K, AB=AB, BB=BB, NUM_XCDS=NUM_XCDS,
                            GMN=GMN, NT=NT, NB=nb, **common)
    else:  # odd / unaligned K -> register path
        _bmm_register[(NT, )](a, b, c, M, N, K, *st, BM=bm, BN=BLOCK_N, BK=BLOCK_K, NUM_XCDS=NUM_XCDS, GMN=GMN, NT=NT,
                              NB=nb, **common)
    return c


def make_bmm_inputs(B, M, N, K, device, dtype=torch.float16, seed=0):
    """SHARED-A: one (M,K) reused across the batch -> a.stride(0)==0. B row-major.

    We always use shared-A: it is the shared-LHS layout, and distinct-A flatters TLX
    (rocBLAS reads shared-A once from HBM, so it is much faster on distinct-A).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((M, K), device=device, dtype=dtype, generator=g).unsqueeze(0).expand(B, M, K)
    b = torch.randn((B, K, N), device=device, dtype=dtype, generator=g)  # row-major (stride_bn == 1)
    return a, b


def _warm_ms(fn, iters=60, warmup=20):
    """Warm device time (L2 hot, back-to-back) — matches rocprofv3 kernel-trace, no launch tax."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    dev = triton.runtime.driver.active.get_active_torch_device()
    # (B, M, N, K): representative shared-LHS shapes.  Always shared-A.
    shapes = [(320, 1024, 256, 256), (1024, 395, 256, 320), (1024, 40, 256, 1956), (1024, 262, 256, 294),
              (1024, 1195, 256, 2309)]
    print("mode: shared-A (shared-LHS)   (B row-major)")
    print(f"{'M x N x K (B)':<22}{'path':<8}{'TLX':>9}{'rocBLAS':>10}{'ratio':>8}  {'ok'}")
    for B, M, N, K in shapes:
        a, b = make_bmm_inputs(B, M, N, K, dev)
        ref = torch.bmm(a, b)
        out = bmm(a, b)
        ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
        t = _warm_ms(lambda: bmm(a, b)) * 1e3
        rb = _warm_ms(lambda: torch.bmm(a, b)) * 1e3
        path = "direct" if K % BLOCK_K == 0 else "reg"
        print(f"{f'{M}x{N}x{K} ({B})':<22}{path:<8}{t:8.0f}u{rb:9.0f}u{rb / t:7.2f}x  {'OK' if ok else 'WRONG'}")
