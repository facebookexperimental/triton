"""AMD a16w16 batched GEMM (BMM) for gfx950 / CDNA4 — dual load-path, single kernel.

C[b] = A[b] @ B[b], fp16/bf16, batched. A is (B, M, K); B is (B, K, N) COLUMN-major
(K-contiguous, ``stride_bk == 1``), i.e. built as (B, N, K) then transposed — the
layout the direct-to-LDS B loads expect. C is (B, M, N) row-major.
``make_bmm_inputs`` builds SHARED-A (one (M, K) reused across the batch,
``a.stride(0) == 0``); benchmark against shared-A, since rocBLAS reads it once and
a distinct-A comparison flatters TLX.

For the standard torch.bmm / inductor layout (ROW-major B) see the companion
``amd_bmm_shared_a.py``, which needs num_warps=8 + matrix_instr_nonkdim=32; nw=8
does not compile with this kernel's swizzle + 4-warp split store.

ONE kernel, two load paths selected by a ``USE_DIRECT`` constexpr the launcher sets
from K's alignment (``K % BLOCK_K == 0``):
  * USE_DIRECT=1 (aligned rows): async direct-to-LDS (``buffer_load_to_local`` —
    global -> LDS with no register round-trip).
  * USE_DIRECT=0 (K % BLOCK_K != 0): register path (``tl.load`` -> registers ->
    ``tlx.local_store``, masked K-tail), required wherever direct-to-LDS is illegal
    on CDNA4 (odd K -> 2-byte-aligned rows).

Design notes shared by both paths:
  * num_warps=4 -> 2 workgroups co-resident per CU; they drift out of phase so the
    hardware overlaps one WG's MFMA with the other's loads (inter-WG ping-pong).
  * Swizzled + padded LDS (``padded_shared_layout_encoding``) -> bank-conflict-free
    ``ds_read_b128``.
  * L2 XCD-chunk remap (``_chip``) -> a batch's M-tiles stay on one XCD, keeping B
    hot in L2.
  * int64 per-batch base advance (batch*M*K can exceed 2**31); within-tile offsets
    stay int32.
  * ``% M`` / ``% N`` wrap on load indices -> OOB rows re-read valid data (L2),
    never HBM garbage.
  * NB-deep software pipeline (prologue prime / steady overlap / epilogue drain).
  * 4-warp coalesced split-store ([128, 256] -> two [128, 128] dwordx4 stores).

Exposes ``bmm(a, b)``.
"""
import os

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BLOCK_N = 256
BLOCK_K = 32
NUM_XCDS = 8
NB = 3
KERNEL_NAME = "amd_bmm"


def _swz(shape, cd):
    """Swizzle-offset bases for a [d0, d1] tile whose contiguous dim is ``cd``."""

    def basis(d, i):
        return [1 << i, 0] if d == 0 else [0, 1 << i]

    fd = 1 - cd
    cb = int(shape[cd]).bit_length() - 1
    fb = int(shape[fd]).bit_length() - 1
    return ([basis(cd, i) for i in range(cb)] + [basis(fd, i)
                                                 for i in range(4, fb)] + [basis(fd, i) for i in range(min(4, fb))])


_B_BASES = tl.constexpr(_swz([BLOCK_K, BLOCK_N], 0))
# 4-warp (256-thread) coalesced [128, 128] store layout: 8 contiguous N per thread (dwordx4).
_C4 = tlx.layout(shape=((16, 16), (8, 8)), stride=((8, 128), (1, 2048)))


@triton.jit
def _chip(pid, nwg, nx: tl.constexpr, cs: tl.constexpr):
    """L2-aware XCD remap: keep a batch's M-tiles (chunk of size ``cs``) on one XCD."""
    al = (nwg // (nx * cs)) * (nx * cs)
    if pid >= al:
        return pid
    x = pid % nx
    lp = pid // nx
    return (lp // cs) * nx * cs + x * cs + (lp % cs)


@triton.jit
def amd_bmm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, A_BASES: tl.constexpr, NUM_XCDS: tl.constexpr,
                   GRID_MN: tl.constexpr, NUM_TILES: tl.constexpr, NB: tl.constexpr, USE_DIRECT: tl.constexpr):
    tl.assume(sam > 0)
    tl.assume(sak > 0)
    tl.assume(sbn > 0)
    tl.assume(sbk > 0)
    npn = tl.cdiv(N, BLOCK_N)
    pidf = _chip(tl.program_id(0), NUM_TILES, NUM_XCDS, GRID_MN)
    bid = pidf // GRID_MN
    pid = pidf % GRID_MN
    pm = pid // npn
    pn = pid % npn
    a_sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], A_BASES, [BLOCK_M, BLOCK_K])
    b_sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], _B_BASES, [BLOCK_K, BLOCK_N])
    sA = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NB, layout=a_sh)
    sB = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NB, layout=b_sh)
    om = (pm * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    on = (pn * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    ok = tl.arange(0, BLOCK_K)
    a_ptr = a_ptr + bid.to(tl.int64) * sab
    b_ptr = b_ptr + bid.to(tl.int64) * sbb
    ao = om[:, None] * sam
    bo = on[None, :] * sbn
    KI = tl.cdiv(K, BLOCK_K)

    # Prime NB LDS buffers before the steady state so tile 0 is already resident.
    for i in tl.range(0, NB, loop_unroll_factor=NB):
        kk = i * BLOCK_K
        if USE_DIRECT:
            tlx.buffer_load_to_local(tlx.local_view(sA, i), a_ptr, ao + (kk + ok[None, :]) * sak)
            tlx.buffer_load_to_local(tlx.local_view(sB, i), b_ptr, (kk + ok[:, None]) * sbk + bo)
            tlx.async_load_commit_group()
        else:
            km = (kk + ok) < K
            ar = tl.load(a_ptr + ao + (kk + ok[None, :]) * sak, mask=km[None, :], other=0.0)
            br = tl.load(b_ptr + (kk + ok[:, None]) * sbk + bo, mask=km[:, None], other=0.0)
            tlx.local_store(tlx.local_view(sA, i), ar)
            tlx.local_store(tlx.local_view(sB, i), br)
    if USE_DIRECT:
        tlx.async_load_wait_group(NB - 2)
    else:
        tl.debug_barrier()
    a = tlx.local_load(tlx.local_view(sA, 0))
    b = tlx.local_load(tlx.local_view(sB, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Steady state: tile k computes while tile k+NB is still in flight.
    for k in tl.range(0, KI - NB):
        cur = (k + 1) % NB
        pf = k % NB
        kp = (k + NB) * BLOCK_K
        acc = tl.dot(a, b, acc)
        if USE_DIRECT:
            tlx.buffer_load_to_local(tlx.local_view(sA, pf), a_ptr, ao + (kp + ok[None, :]) * sak)
            tlx.buffer_load_to_local(tlx.local_view(sB, pf), b_ptr, (kp + ok[:, None]) * sbk + bo)
            tlx.async_load_commit_group()
            tlx.async_load_wait_group(NB - 2)
        else:
            km = (kp + ok) < K
            ar = tl.load(a_ptr + ao + (kp + ok[None, :]) * sak, mask=km[None, :], other=0.0)
            br = tl.load(b_ptr + (kp + ok[:, None]) * sbk + bo, mask=km[:, None], other=0.0)
            tlx.local_store(tlx.local_view(sA, pf), ar)
            tlx.local_store(tlx.local_view(sB, pf), br)
            tl.debug_barrier()
        a = tlx.local_load(tlx.local_view(sA, cur))
        b = tlx.local_load(tlx.local_view(sB, cur))

    # Drain the NB tiles the prologue put in flight but the loop never consumed.
    acc = tl.dot(a, b, acc)
    if USE_DIRECT:
        tlx.async_load_wait_group(0)
    for i in tl.range(0, NB - 1, loop_unroll_factor=NB - 1):
        bf = (KI - (NB - 1) + i) % NB
        acc = tl.dot(tlx.local_load(tlx.local_view(sA, bf)), tlx.local_load(tlx.local_view(sB, bf)), acc)

    # Split the [BLOCK_M, BLOCK_N] accumulator into two N-halves so each of the 4
    # warps stores 8 contiguous N per thread (dwordx4).
    et = c_ptr.dtype.element_ty
    cb = c_ptr + bid.to(tl.int64) * scb
    HN: tl.constexpr = BLOCK_N // 2
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rnl = pn * BLOCK_N + tl.arange(0, HN)
    rnr = rnl + HN
    mm = rm[:, None] < M
    al, ar2 = tl.split(tl.trans(tl.reshape(acc, BLOCK_M, 2, HN), 0, 2, 1))
    tl.store(cb + scm * rm[:, None] + scn * rnl[None, :], tlx.require_layout(al.to(et), _C4),
             mask=mm & (rnl[None, :] < N))
    tl.store(cb + scm * rm[:, None] + scn * rnr[None, :], tlx.require_layout(ar2.to(et), _C4),
             mask=mm & (rnr[None, :] < N))


def bmm(a, b, block_m=None, nw=4, nb=None):
    """C = A @ B batched. a (B, M, K) row-major; b (B, K, N) COLUMN-major (stride_bk == 1)."""
    assert a.ndim == 3 and b.ndim == 3 and a.shape[0] == b.shape[0]
    # sA / sB take their element type from a_ptr / b_ptr independently, so a dtype
    # mismatch would silently give the two LDS buffers different types.
    assert a.dtype == b.dtype, f"A and B must have the same dtype, got {a.dtype} and {b.dtype}"
    Bs, M, K = a.shape
    _, _, N = b.shape
    bm = block_m or int(os.environ.get("BM_BMM", "128"))
    # _C4 is a pinned [128,128] store layout (see its definition), so the epilogue
    # require_layout only matches at bm == 128. Fail here rather than deep inside
    # the compiler if BM_BMM / block_m is overridden.
    assert bm == 128, f"block_m must be 128 to match the pinned _C4 store layout, got {bm}"
    A_BASES = tuple(tuple(x) for x in _swz([bm, BLOCK_K], 1))
    nb = nb or int(os.environ.get("NB_BMM", NB))
    KI = triton.cdiv(K, BLOCK_K)
    assert KI >= 2, f"K must span >= 2 BLOCK_K={BLOCK_K} tiles for the pipeline, got K={K}"
    # The 2-stage pipeline needs 2 <= nb <= KI: the prologue unconditionally issues
    # nb loads and the drain indexes (KI - (nb - 1) + i) % nb, so nb > KI would
    # over-read past K and re-accumulate tile 0. Requiring KI >= 2 makes
    # min(NB, KI) >= 2 on its own -- a max(2, ...) here would defeat the clamp.
    nb = min(nb, KI)
    use_direct = (K % BLOCK_K == 0)  # aligned, no K-tail -> fast direct-to-LDS
    GM = triton.cdiv(M, bm) * triton.cdiv(N, BLOCK_N)
    NT = Bs * GM
    c = torch.empty((Bs, M, N), device=a.device, dtype=a.dtype)
    amd_bmm_kernel[(NT, )](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        BLOCK_M=bm,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        A_BASES=A_BASES,
        NUM_XCDS=NUM_XCDS,
        GRID_MN=GM,
        NUM_TILES=NT,
        NB=nb,
        USE_DIRECT=use_direct,
        num_warps=nw,
        num_stages=1,
        matrix_instr_nonkdim=16,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
    )
    return c


def make_bmm_inputs(B, M, N, K, device, dtype=torch.float16, seed=0):
    """SHARED-A (the shared-LHS layout): a single (M, K) matrix reused
    across the whole batch via ``expand`` -> ``a.stride(0) == 0`` (mat1 batch-stride 0).
    rocBLAS/hipBLASLt exploits this to read A once from HBM (L2-resident), so it is MUCH
    faster than distinct-A -> we ALWAYS benchmark shared-A; distinct-A flatters TLX.
    B is (B, K, N) COLUMN-major (built as (B, N, K) then transposed); C is (B, M, N) row-major.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((M, K), device=device, dtype=dtype, generator=g).unsqueeze(0).expand(B, M, K)
    b = torch.randn((B, N, K), device=device, dtype=dtype, generator=g).transpose(1, 2)  # (B, K, N) col-major
    return a, b


if __name__ == "__main__":
    dev = triton.runtime.driver.active.get_active_torch_device()
    # (M, N, K, B): mix of aligned (direct-to-LDS) and odd/unaligned (register) K.
    shapes = [(1024, 256, 256, 320), (395, 256, 320, 1024), (140, 256, 1888, 1024), (262, 256, 294, 1024),
              (40, 256, 1956, 1024), (176, 256, 257, 1024), (1195, 256, 2309, 1024), (433, 256, 2352, 1024)]
    print(f"{'M x N x K (B)':<24}{'path':<9}{'max_err':<10}{'result'}")
    for M, N, K, B in shapes:
        a, b = make_bmm_inputs(B, M, N, K, dev)
        ref = torch.bmm(a.float(), b.float())
        out = bmm(a, b).float()
        err = (out - ref).abs().max().item()
        path = "direct" if K % BLOCK_K == 0 else "register"
        print(f"{f'{M}x{N}x{K} ({B})':<24}{path:<9}{err:<10.4f}{'OK' if err < 0.15 else 'FAIL'}")
