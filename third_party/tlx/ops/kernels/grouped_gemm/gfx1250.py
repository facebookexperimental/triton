"""AMD TLX grouped GEMM for gfx1250 -- persistent TDM pipeline over pointer tables.

The tutorial ``tutorials/amd_grouped_gemm_gfx1250`` imports this module for
its packed-tensor entry point, tuning sweeps and benchmarks. Each program walks the
output tiles of the whole group in a persistent loop. Tensor data mover (TDM)
loads fill a two-stage LDS ring per 256x256x128 tile; with an even number of
K blocks the next tile's first stages are prefetched during the current
tile's tail (cross-tile prefetch). Groups whose shapes allow it run in
clusters of four workgroups that share their A/B loads through TDM
multicast; other shapes use single workgroups with the same schedule.

Storage:
  A[i]: row-major FP16 ``[M_i, K]``.
  B[i]: column-major FP16 ``[K, N]`` (a ``[N, K]`` weight's ``.T``).
  C[i]: row-major FP16 ``[M_i, N]``.

Only M may vary across groups; N and K are common. K must be a multiple of
128 with at least two K blocks. M and N tails are masked; zero-row groups are
allowed. Base pointers must be 16-byte aligned. Inputs are used in place:
setup builds pointer, size and stride tables and allocates the outputs, with
no repacking.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import InvalidInput


def _active_num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _grouped_gemm_enable_wmma_queuing(block_m, block_n, num_buffers, c_staging_mode, cross_tile_prefetch):
    # The square single-wave-per-SIMD schedules benefit from SCHED_MODE[2].
    return block_m == block_n == 256 and (cross_tile_prefetch or (c_staging_mode == 0 and num_buffers == 2))


_XCD_REMAP_MODES = {
    "none": 0,
    "balanced": 1,
    "chunked": 2,
}


def _cdiv_int(a: int, b: int) -> int:
    return -(-a // b)


@triton.jit
def _tdm_load_subtile(
    a_buf,
    b_buf,
    consumer,
    start: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_LEN: tl.constexpr,
):
    slot = consumer % NUM_BUFFERS
    a_view = tlx.local_slice(a_buf[slot], [0, start], [BLOCK_M, SUBTILE_LEN])
    b_view = tlx.local_slice(b_buf[slot], [0, start], [BLOCK_N, SUBTILE_LEN])
    return tlx.local_load(a_view), tlx.local_load(tlx.local_trans(b_view))


@triton.jit
def _tdm_load_fused(a_desc, b_desc, a_view, b_view, CLUSTER_SIZE: tl.constexpr, CLUSTER_MULTICAST: tl.constexpr):
    if CLUSTER_SIZE > 1:
        # Every wave must finish reading a slot before any workgroup can
        # refill that slot through multicast into a neighbor's LDS.
        tlx.cluster_barrier()
    if CLUSTER_SIZE > 1 and CLUSTER_MULTICAST:
        rank = tlx.cluster_cta_rank()
        a_mask = 0 if CLUSTER_SIZE == 2 else 5 << (rank & 1)
        b_mask = 3 if CLUSTER_SIZE == 2 else 3 << (rank & 2)
        masks = [a_mask, b_mask]
    else:
        masks = None
    tlx.async_amd_descriptor_load_fused([(a_desc, a_view, 3), (b_desc, b_view, 12)], multicast_masks=masks)


@triton.jit
def _tdm_issue_loads(
    a_desc,
    b_desc,
    a_buf,
    b_buf,
    producer,
    off_m,
    off_n,
    pred,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr = 1,
    CLUSTER_MULTICAST: tl.constexpr = True,
):
    slot = producer % NUM_BUFFERS
    a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[off_m, producer * BLOCK_K], clamp_bounds=False,
                                          pred=pred)
    b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[off_n, producer * BLOCK_K], clamp_bounds=False,
                                          pred=pred)
    _tdm_load_fused(a_desc, b_desc, a_buf[slot], b_buf[slot], CLUSTER_SIZE, CLUSTER_MULTICAST)
    return producer + 1


@triton.jit
def _tdm_issue_loads_unpredicated(
    a_desc,
    b_desc,
    a_buf,
    b_buf,
    producer,
    off_m,
    off_n,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr = 1,
    CLUSTER_MULTICAST: tl.constexpr = True,
):
    slot = producer % NUM_BUFFERS
    a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[off_m, producer * BLOCK_K], clamp_bounds=False)
    b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[off_n, producer * BLOCK_K], clamp_bounds=False)
    _tdm_load_fused(a_desc, b_desc, a_buf[slot], b_buf[slot], CLUSTER_SIZE, CLUSTER_MULTICAST)
    return producer + 1


@triton.jit
def _tdm_prefetch_unpredicated(a_desc, b_desc, prefetch_iter, off_m, off_n, BLOCK_K: tl.constexpr):
    tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, prefetch_iter * BLOCK_K])
    tlx.amd_descriptor_prefetch_tensor(b_desc, [off_n, prefetch_iter * BLOCK_K])


@triton.jit
def _tdm_accumulate_subtiles(
    acc,
    a0,
    b0,
    a_desc,
    b_desc,
    a_buf,
    b_buf,
    consumer,
    producer,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_LEN: tl.constexpr,
    LIMIT_DOT_LIFETIME: tl.constexpr = False,
):
    a1, b1 = _tdm_load_subtile(a_buf, b_buf, consumer, SUBTILE_LEN, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)
    acc = tl.dot(a0, b0, acc)
    if LIMIT_DOT_LIFETIME:
        # Bound live operand tiles while allowing loads to interleave with
        # this dot. The following dot starts a new scheduling region.
        tlx.amd_sched_barrier()

    if L2_PREFETCH_DISTANCE > 0:
        prefetch_iter = producer + L2_PREFETCH_DISTANCE - 1
        _tdm_prefetch_unpredicated(a_desc, b_desc, prefetch_iter, 0, 0, BLOCK_K)

    a2, b2 = _tdm_load_subtile(a_buf, b_buf, consumer, 2 * SUBTILE_LEN, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)
    acc = tl.dot(a1, b1, acc)
    if LIMIT_DOT_LIFETIME:
        tlx.amd_sched_barrier()

    a3, b3 = _tdm_load_subtile(a_buf, b_buf, consumer, 3 * SUBTILE_LEN, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)
    acc = tl.dot(a2, b2, acc)
    if LIMIT_DOT_LIFETIME:
        tlx.amd_sched_barrier()
    return acc, a3, b3


@triton.jit
def _tdm_wait_and_finish_k_block(
    acc,
    a3,
    b3,
    a_buf,
    b_buf,
    consumer,
    NUM_BUFFERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_LEN: tl.constexpr,
    LIMIT_DOT_LIFETIME: tl.constexpr = False,
):
    tlx.async_amd_descriptor_wait(NUM_BUFFERS - 1)
    a0, b0 = _tdm_load_subtile(a_buf, b_buf, consumer, 0, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)
    acc = tl.dot(a3, b3, acc)
    if LIMIT_DOT_LIFETIME:
        tlx.amd_sched_barrier()
    return acc, a0, b0


@triton.jit
def _tdm_c_store_chunk(acc, part: tl.constexpr):
    """Select 32 rows without keeping a second full output tile in registers."""
    width: tl.constexpr = acc.shape[1]
    lo, hi = tl.split(tl.reshape(acc, (2, 128, width)).permute(1, 2, 0))
    half = lo if part < 4 else hi
    lo2, hi2 = tl.split(tl.reshape(half, (2, 64, width)).permute(1, 2, 0))
    quarter = lo2 if part % 4 < 2 else hi2
    lo3, hi3 = tl.split(tl.reshape(quarter, (2, 32, width)).permute(1, 2, 0))
    return lo3 if part % 2 == 0 else hi3


@triton.jit
def _grouped_tile_offsets(
    tile_idx,
    last_problem_end,
    num_m_tiles,
    num_n_tiles,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_idx_in_gemm = tile_idx - last_problem_end
    num_pid_in_group = GROUP_M * num_n_tiles
    group_id = tile_idx_in_gemm // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_m_tiles - first_pid_m, GROUP_M)
    tile_m_idx = first_pid_m + ((tile_idx_in_gemm % num_pid_in_group) % group_size_m)
    tile_n_idx = (tile_idx_in_gemm % num_pid_in_group) // group_size_m
    return tile_m_idx * BLOCK_M, tile_n_idx * BLOCK_N


@triton.jit
def _remap_grouped_program_id(
    pid,
    num_programs,
    XCD_REMAP_MODE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    if XCD_REMAP_MODE == 0 or NUM_XCDS == 1:
        return pid
    elif XCD_REMAP_MODE == 1:
        # Physical workgroups are assigned round-robin across XCDs. Map each
        # XCD's local sequence onto one balanced contiguous logical range.
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        min_per_xcd = num_programs // NUM_XCDS
        extra = num_programs % NUM_XCDS
        return xcd * min_per_xcd + min(xcd, extra) + local_pid
    else:
        tl.static_assert(XCD_REMAP_MODE == 2, "XCD_REMAP_MODE must be none, balanced, or chunked")
        tl.static_assert(XCD_CHUNK >= 1, "XCD_CHUNK must be positive")
        aligned = (num_programs // (NUM_XCDS * XCD_CHUNK)) * (NUM_XCDS * XCD_CHUNK)
        if pid >= aligned:
            return pid
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        return ((local_pid // XCD_CHUNK) * NUM_XCDS * XCD_CHUNK + xcd * XCD_CHUNK + (local_pid % XCD_CHUNK))


@triton.jit
def _grouped_gemm_matrix_pointer(base, group, packed_offset, POINTER_TABLES: tl.constexpr):
    if POINTER_TABLES:
        ptr = tl.load(base + group).to(tl.pointer_type(tl.float16))
        # The list wrapper validates every matrix's base alignment. A scalar
        # group index selects the same pointer for all lanes in the workgroup.
        return tl.multiple_of(tlx.assume_uniform(ptr), 16)
    else:
        return base + packed_offset


@triton.jit
def _grouped_gemm_tdm_impl(
    a_packed,
    b_t,
    c_packed,
    group_offsets,
    group_size,
    N,
    stride_am,
    stride_bg,
    stride_bn,
    stride_cm,
    K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    C_STAGING_MODE: tl.constexpr,
    CROSS_TILE_PREFETCH: tl.constexpr,
    XCD_REMAP_MODE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr = 1,
    CLUSTER_MULTICAST: tl.constexpr = True,
    POINTER_TABLES: tl.constexpr = False,
    group_lds=None,
    ALLOW_MN_TAILS: tl.constexpr = False,
):
    """Shared packed/list ragged-M scheduler and TDM compute pipeline.

    With POINTER_TABLES, the first three arguments hold per-group FP16 matrix
    addresses, group_offsets holds gfx950-style [M, N, K] rows, and group_lds
    holds [lda, ldb, ldc] rows. Otherwise inputs are packed tensors and
    group_offsets holds cumulative M offsets. Both paths load B as
    [BLOCK_N, BLOCK_K] and transpose locally before ``tl.dot``.
    """
    tl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    tl.static_assert(C_STAGING_MODE == 0 or C_STAGING_MODE == 1, "C_STAGING_MODE must be 0 or 1")
    tl.static_assert(not CROSS_TILE_PREFETCH or C_STAGING_MODE == 1 or (BLOCK_M == 256 and BLOCK_N == 256),
                     "alias-C cross-tile prefetch requires a 256x256 tile")
    tl.static_assert(not CROSS_TILE_PREFETCH or NUM_BUFFERS == 2, "cross-tile prefetch currently requires depth 2")
    tl.static_assert(NUM_XCDS >= 1, "NUM_XCDS must be positive")
    tl.static_assert(not ALLOW_MN_TAILS or (POINTER_TABLES and CLUSTER_SIZE == 1),
                     "M/N tails require pointer tables and cluster size 1")
    # The square tile uses one third fewer LDS reads per WMMA than 128x256.
    # Limit operand lifetimes and K-loop scope to keep it free of scratch spills.
    OPTIMIZE_CROSS_TILE: tl.constexpr = CROSS_TILE_PREFETCH and BLOCK_M == 256 and BLOCK_N == 256
    HYBRID_TILE_PREFETCH: tl.constexpr = OPTIMIZE_CROSS_TILE and C_STAGING_MODE == 0
    tl.static_assert(CLUSTER_SIZE == 1 or CLUSTER_SIZE == 2 or CLUSTER_SIZE == 4, "cluster size must be 1, 2, or 4")
    tl.static_assert(CLUSTER_SIZE == 1 or HYBRID_TILE_PREFETCH, "clustering requires the square within-group hybrid")
    CROSS_GROUP_PREFETCH: tl.constexpr = OPTIMIZE_CROSS_TILE and not HYBRID_TILE_PREFETCH
    OPTIMIZE_ALIAS_C: tl.constexpr = C_STAGING_MODE == 0 and BLOCK_M == 256 and BLOCK_N == 256 and NUM_BUFFERS == 2
    # Cross-tile prefetch with dedicated C staging also bounds operand
    # lifetimes. Otherwise the scheduler may hoist each K block's TDM refill and
    # its workgroup barrier ahead of most of the block's dots, then read LDS
    # operands far ahead of their dots, which raises register pressure and
    # slows the 128x256 tile.
    LIMIT_DOT_LIFETIME: tl.constexpr = (OPTIMIZE_CROSS_TILE or OPTIMIZE_ALIAS_C
                                        or (CROSS_TILE_PREFETCH and C_STAGING_MODE == 1))
    DIRECT_LOAD_OFFSETS: tl.constexpr = OPTIMIZE_CROSS_TILE and L2_PREFETCH_DISTANCE == 0
    K_BLOCKS_PER_LOOP: tl.constexpr = 1 if CROSS_GROUP_PREFETCH else NUM_BUFFERS
    NUM_SUBTILES: tl.constexpr = 4
    SUBTILE_LEN: tl.constexpr = BLOCK_K // NUM_SUBTILES
    K_ITERS: tl.constexpr = K // BLOCK_K
    USE_FULL_C_TILE: tl.constexpr = (NUM_BUFFERS * BLOCK_K) % BLOCK_N == 0
    tl.static_assert(K % BLOCK_K == 0, "K must be divisible by BLOCK_K")
    tl.static_assert(K_ITERS >= NUM_BUFFERS, "K must cover the TDM pipeline")
    tl.static_assert(SUBTILE_LEN == 32, "BLOCK_K must be 128 for the first TDM schedule")

    pid = _remap_grouped_program_id(tl.program_id(0), NUM_PROGRAMS, XCD_REMAP_MODE, NUM_XCDS, XCD_CHUNK)
    tile_idx = pid
    last_problem_end = 0
    num_n_tiles = tl.cdiv(N, BLOCK_N)

    a_dtype: tl.constexpr = tl.float16 if POINTER_TABLES else tlx.dtype_of(a_packed)
    b_dtype: tl.constexpr = tl.float16 if POINTER_TABLES else tlx.dtype_of(b_t)
    c_dtype: tl.constexpr = tl.float16 if POINTER_TABLES else tlx.dtype_of(c_packed)
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), a_dtype, NUM_BUFFERS)
    b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K), b_dtype, NUM_BUFFERS)
    # TLXRewriteLocalAlias only accepts size-mismatched shared aliases when
    # the backing allocation is an integer multiple of the alias.  Reuse A only
    # when that ratio is legal.  Otherwise store C in BLOCK_K-wide chunks that
    # are each small enough to alias A.  Reusing B can satisfy the size check
    # for asymmetric tiles, but B's TDM-load/dot-transpose layout constraints
    # conflict with C's TDM-store layout constraints.
    if HYBRID_TILE_PREFETCH:
        # Two 16-KiB output slots fit alongside the input rings. TDM drains
        # one chunk while the other is filled, leaving prefetched A/B intact.
        c_buf = tlx.local_alloc((32, BLOCK_N), c_dtype, 2)
    elif C_STAGING_MODE == 1:
        tl.static_assert(USE_FULL_C_TILE, "dedicated C staging currently requires a full C buffer")
        c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), c_dtype, 1)
    elif USE_FULL_C_TILE:
        c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), c_dtype, 1, reuse=a_buf)
    else:
        tl.static_assert(BLOCK_N == 2 * BLOCK_K, "split C store currently supports BLOCK_N == 2 * BLOCK_K")
        c_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), c_dtype, 1, reuse=a_buf)

    group_primed = False
    if CROSS_GROUP_PREFETCH:
        # Keep four upcoming boundaries available across output stores. Refill
        # the window in a tile's tail so group entry needs no metadata load.
        if POINTER_TABLES:
            # Pointer-table matrices start at row zero. Carry their M sizes
            # instead of packed boundaries; no prefix-sum buffer is needed.
            carried_m_start = tl.full((), 0, tl.int32)
            carried_m_end = tl.load(group_offsets, mask=group_size > 0, other=0)
            carried_following_m_end = tl.load(group_offsets + 3, mask=group_size > 1, other=0)
            carried_third_m_end = tl.load(group_offsets + 6, mask=group_size > 2, other=0)
            carried_fourth_m_end = tl.load(group_offsets + 9, mask=group_size > 3, other=0)
        else:
            carried_m_start = tl.load(group_offsets)
            carried_m_end = tl.load(group_offsets + 1, mask=group_size > 0, other=0)
            carried_following_m_end = tl.load(group_offsets + tl.minimum(2, group_size))
            carried_third_m_end = tl.load(group_offsets + tl.minimum(3, group_size))
            carried_fourth_m_end = tl.load(group_offsets + tl.minimum(4, group_size))
    for g in range(group_size):
        if CROSS_GROUP_PREFETCH:
            m_start, m_end = carried_m_start, carried_m_end
            following_m_end = carried_following_m_end
            third_m_end = carried_third_m_end
            fourth_m_end = carried_fourth_m_end
            prefetched_boundary = g + 4 >= group_size
        elif POINTER_TABLES:
            m_start = tl.full((), 0, tl.int32)
            m_end = tl.load(group_offsets + 3 * g)
        else:
            m_start = tl.load(group_offsets + g)
            m_end = tl.load(group_offsets + g + 1)
        gm = m_end - m_start
        if POINTER_TABLES:
            group_stride_am = tl.multiple_of(tl.load(group_lds + 3 * g), 8)
            group_stride_bn = tl.multiple_of(tl.load(group_lds + 3 * g + 1), 8)
            if ALLOW_MN_TAILS:
                group_stride_cm = tl.load(group_lds + 3 * g + 2)
            else:
                group_stride_cm = tl.multiple_of(tl.load(group_lds + 3 * g + 2), 8)
        else:
            group_stride_am, group_stride_bn, group_stride_cm = stride_am, stride_bn, stride_cm

        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        group_base_m = m_start.to(tl.int64) * group_stride_am
        group_base_c = m_start.to(tl.int64) * group_stride_cm
        group_base_b = g.to(tl.int64) * stride_bg
        a_group = _grouped_gemm_matrix_pointer(a_packed, g, group_base_m, POINTER_TABLES)
        b_group = _grouped_gemm_matrix_pointer(b_t, g, group_base_b, POINTER_TABLES)
        c_group = _grouped_gemm_matrix_pointer(c_packed, g, group_base_c, POINTER_TABLES)

        a_desc_base = tl.make_tensor_descriptor(
            a_group,
            # The launcher validates exact BLOCK_K coverage. Keeping the K
            # extent static avoids rebuilding dynamic bounds at every TDM issue.
            shape=[gm, K],
            strides=[group_stride_am, tl.constexpr(1)],
            block_shape=[BLOCK_M, BLOCK_K],
        )
        b_desc_base = tl.make_tensor_descriptor(
            b_group,
            shape=[N, K],
            strides=[group_stride_bn, tl.constexpr(1)],
            block_shape=[BLOCK_N, BLOCK_K],
        )
        if USE_FULL_C_TILE:
            c_desc_base = tl.make_tensor_descriptor(
                c_group,
                shape=[gm, N],
                strides=[group_stride_cm, tl.constexpr(1)],
                block_shape=[32 if HYBRID_TILE_PREFETCH else BLOCK_M, BLOCK_N],
            )
        else:
            c_desc_base = tl.make_tensor_descriptor(
                c_group,
                shape=[gm, N],
                strides=[group_stride_cm, tl.constexpr(1)],
                # Match the BLOCK_K-wide LDS alias above: when a full C tile
                # cannot legally alias the A ring, store its two halves. This
                # is an aliasing constraint, not a store-efficiency choice.
                block_shape=[BLOCK_M, BLOCK_K],
            )

        if CROSS_TILE_PREFETCH:
            # Prime only when the predecessor did not provide K0/K1. The
            # cross-group mode also carries inputs across group boundaries.
            if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles and (not CROSS_GROUP_PREFETCH
                                                                                             or not group_primed):
                if HYBRID_TILE_PREFETCH:
                    # Retire the previous group's outstanding output chunks.
                    tlx.async_amd_descriptor_wait(0)
                first_off_m, first_off_n = _grouped_tile_offsets(tile_idx, last_problem_end, num_m_tiles, num_n_tiles,
                                                                 GROUP_M, BLOCK_M, BLOCK_N)
                if ALLOW_MN_TAILS and DIRECT_LOAD_OFFSETS:
                    # Keep the group address and fold tile offsets into each
                    # load, as in the full-tile path. Only bounds vary per tile.
                    prime_a_desc = tlx.update_tensor_descriptor(a_desc_base,
                                                                set_bounds=[tl.maximum(gm - first_off_m, 0), K])
                    prime_b_desc = tlx.update_tensor_descriptor(b_desc_base,
                                                                set_bounds=[tl.maximum(N - first_off_n, 0), K])
                    prime_m, prime_n = first_off_m, first_off_n
                elif ALLOW_MN_TAILS:
                    # Clamp once at tile entry, keeping K-loop updates address-only.
                    prime_a_desc = tlx.update_tensor_descriptor(a_desc_base, add_offsets=[first_off_m, 0],
                                                                clamp_bounds=True)
                    prime_b_desc = tlx.update_tensor_descriptor(b_desc_base, add_offsets=[first_off_n, 0],
                                                                clamp_bounds=True)
                    prime_m, prime_n = 0, 0
                else:
                    prime_a_desc, prime_b_desc = a_desc_base, b_desc_base
                    prime_m, prime_n = first_off_m, first_off_n
                _tdm_issue_loads_unpredicated(
                    prime_a_desc,
                    prime_b_desc,
                    a_buf,
                    b_buf,
                    0,
                    prime_m,
                    prime_n,
                    BLOCK_K,
                    NUM_BUFFERS,
                    CLUSTER_SIZE,
                    CLUSTER_MULTICAST,
                )
                # Fold the tile position into each fused load to shorten
                # descriptor lifetimes. L2 prefetch retains the positioned path.
                if ALLOW_MN_TAILS:
                    first_a_desc, first_b_desc = prime_a_desc, prime_b_desc
                    first_load_m, first_load_n = prime_m, prime_n
                elif DIRECT_LOAD_OFFSETS:
                    first_a_desc, first_b_desc = a_desc_base, b_desc_base
                    first_load_m, first_load_n = first_off_m, first_off_n
                else:
                    first_a_desc = tlx.update_tensor_descriptor(a_desc_base, add_offsets=[first_off_m, 0])
                    first_b_desc = tlx.update_tensor_descriptor(b_desc_base, add_offsets=[first_off_n, 0])
                    first_load_m, first_load_n = 0, 0
                _tdm_issue_loads_unpredicated(
                    first_a_desc,
                    first_b_desc,
                    a_buf,
                    b_buf,
                    1,
                    first_load_m,
                    first_load_n,
                    BLOCK_K,
                    NUM_BUFFERS,
                    CLUSTER_SIZE,
                    CLUSTER_MULTICAST,
                )
                tlx.async_amd_descriptor_wait(NUM_BUFFERS - 1)

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            off_m, off_n = _grouped_tile_offsets(tile_idx, last_problem_end, num_m_tiles, num_n_tiles, GROUP_M, BLOCK_M,
                                                 BLOCK_N)
            if DIRECT_LOAD_OFFSETS:
                a_desc, b_desc = a_desc_base, b_desc_base
                if ALLOW_MN_TAILS:
                    a_desc = tlx.update_tensor_descriptor(a_desc, set_bounds=[tl.maximum(gm - off_m, 0), K])
                    b_desc = tlx.update_tensor_descriptor(b_desc, set_bounds=[tl.maximum(N - off_n, 0), K])
                load_off_m, load_off_n = off_m, off_n
            else:
                a_desc = tlx.update_tensor_descriptor(a_desc_base, add_offsets=[off_m, 0], clamp_bounds=ALLOW_MN_TAILS)
                b_desc = tlx.update_tensor_descriptor(b_desc_base, add_offsets=[off_n, 0], clamp_bounds=ALLOW_MN_TAILS)
                load_off_m, load_off_n = 0, 0

            producer = 0
            consumer = 0

            if C_STAGING_MODE == 0 and not HYBRID_TILE_PREFETCH:
                # The previous tile's async C store may still be reading from
                # c_buf, which aliases a_buf. Delay the drain until the next
                # actual LDS overwrite hazard instead of waiting immediately
                # after the store.
                tlx.async_amd_descriptor_wait(0)

            if not CROSS_TILE_PREFETCH:
                # Prime all ring slots up front so the first consumer load only
                # waits until the oldest TDM op is complete, leaving newer
                # slots in flight.
                for _ in tl.static_range(NUM_BUFFERS):
                    producer = _tdm_issue_loads_unpredicated(
                        a_desc,
                        b_desc,
                        a_buf,
                        b_buf,
                        producer,
                        0,
                        0,
                        BLOCK_K,
                        NUM_BUFFERS,
                        CLUSTER_SIZE,
                        CLUSTER_MULTICAST,
                    )
                # With dedicated C staging, the previous C store is older than
                # these input groups. This count-based wait retires that store
                # and the oldest input group before the first LDS read.
                tlx.async_amd_descriptor_wait(NUM_BUFFERS - 1)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            epilogue_lb = K_ITERS - (NUM_BUFFERS - 1)
            a0, b0 = _tdm_load_subtile(a_buf, b_buf, consumer, 0, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)

            if CROSS_TILE_PREFETCH:
                if not OPTIMIZE_CROSS_TILE:
                    next_tile_idx = tile_idx + NUM_PROGRAMS
                    group_end = last_problem_end + num_tiles
                    has_next = next_tile_idx < group_end
                    safe_next_tile_idx = min(next_tile_idx, group_end - 1)
                    next_off_m, next_off_n = _grouped_tile_offsets(safe_next_tile_idx, last_problem_end, num_m_tiles,
                                                                   num_n_tiles, GROUP_M, BLOCK_M, BLOCK_N)
                    if DIRECT_LOAD_OFFSETS:
                        next_a_desc, next_b_desc = a_desc_base, b_desc_base
                        next_load_m, next_load_n = next_off_m, next_off_n
                    else:
                        next_a_desc = tlx.update_tensor_descriptor(a_desc_base, add_offsets=[next_off_m, 0],
                                                                   clamp_bounds=ALLOW_MN_TAILS)
                        next_b_desc = tlx.update_tensor_descriptor(b_desc_base, add_offsets=[next_off_n, 0],
                                                                   clamp_bounds=ALLOW_MN_TAILS)
                        next_load_m, next_load_n = 0, 0

                steady_end = K_ITERS - NUM_BUFFERS
                # Producer and consumer are affine functions of the canonical
                # loop IV; deriving them avoids two scalar loop-carried values.
                # Cross-group lookahead keeps one K block per loop to limit
                # live state. Other paths retain ring-sized chunks; the
                # host contract makes steady_end divisible by NUM_BUFFERS.
                chunk_end = steady_end // K_BLOCKS_PER_LOOP
                for chunk in tl.range(0, chunk_end):
                    for phase in tl.static_range(K_BLOCKS_PER_LOOP):
                        i = chunk * K_BLOCKS_PER_LOOP + phase
                        producer_i = i + NUM_BUFFERS
                        acc, a3, b3 = _tdm_accumulate_subtiles(acc, a0, b0, a_desc, b_desc, a_buf, b_buf, i, producer_i,
                                                               L2_PREFETCH_DISTANCE, BLOCK_K, NUM_BUFFERS, BLOCK_M,
                                                               BLOCK_N, SUBTILE_LEN, LIMIT_DOT_LIFETIME)
                        _tdm_issue_loads_unpredicated(
                            a_desc,
                            b_desc,
                            a_buf,
                            b_buf,
                            producer_i,
                            load_off_m,
                            load_off_n,
                            BLOCK_K,
                            NUM_BUFFERS,
                            CLUSTER_SIZE,
                            CLUSTER_MULTICAST,
                        )
                        acc, a0, b0 = _tdm_wait_and_finish_k_block(acc, a3, b3, a_buf, b_buf, i + 1, NUM_BUFFERS,
                                                                   BLOCK_M, BLOCK_N, SUBTILE_LEN, LIMIT_DOT_LIFETIME)

                if CROSS_GROUP_PREFETCH:
                    if not prefetched_boundary:
                        if POINTER_TABLES:
                            carried_fourth_m_end = tl.load(group_offsets + 3 * (g + 4))
                        else:
                            carried_fourth_m_end = tl.load(group_offsets + g + 5)
                        prefetched_boundary = True
                    # Discover the next tile only after the steady K loop, so its
                    # descriptors are not live through the main matrix schedule.
                    tlx.amd_sched_barrier()
                    next_tile_idx = tile_idx + NUM_PROGRAMS
                    next_group = g
                    next_group_start = last_problem_end
                    next_group_end = last_problem_end + num_tiles
                    next_m_start = m_start
                    next_gm = gm
                    next_num_m_tiles = num_m_tiles
                    while next_tile_idx >= next_group_end and next_group + 1 < group_size:
                        next_group += 1
                        if not POINTER_TABLES:
                            next_m_start += next_gm
                        if next_group == g + 1:
                            next_m_end = following_m_end
                        elif next_group == g + 2:
                            next_m_end = third_m_end
                        elif next_group == g + 3:
                            next_m_end = fourth_m_end
                        elif POINTER_TABLES:
                            next_m_end = tl.load(group_offsets + 3 * next_group)
                        else:
                            next_m_end = tl.load(group_offsets + next_group + 1)
                        next_gm = next_m_end - next_m_start
                        next_num_m_tiles = tl.cdiv(next_gm, BLOCK_M)
                        next_group_start = next_group_end
                        next_group_end += next_num_m_tiles * num_n_tiles
                    has_next = next_tile_idx < next_group_end
                    # Keep descriptor dimensions and tile arithmetic valid even
                    # when the final groups are empty and these loads are disabled.
                    next_group = tl.where(has_next, next_group, g)
                    next_group_start = tl.where(has_next, next_group_start, last_problem_end)
                    next_m_start = tl.where(has_next, next_m_start, m_start)
                    next_gm = tl.where(has_next, next_gm, gm)
                    next_num_m_tiles = tl.where(has_next, next_num_m_tiles, num_m_tiles)
                    safe_next_tile_idx = tl.where(has_next, next_tile_idx, tile_idx)
                    next_off_m, next_off_n = _grouped_tile_offsets(safe_next_tile_idx, next_group_start,
                                                                   next_num_m_tiles, num_n_tiles, GROUP_M, BLOCK_M,
                                                                   BLOCK_N)
                    next_a_group = _grouped_gemm_matrix_pointer(a_packed, next_group,
                                                                next_m_start.to(tl.int64) * group_stride_am,
                                                                POINTER_TABLES)
                    next_b_group = _grouped_gemm_matrix_pointer(b_t, next_group,
                                                                next_group.to(tl.int64) * stride_bg, POINTER_TABLES)
                    if POINTER_TABLES:
                        next_stride_am = tl.multiple_of(tl.load(group_lds + 3 * next_group), 8)
                        next_stride_bn = tl.multiple_of(tl.load(group_lds + 3 * next_group + 1), 8)
                    else:
                        next_stride_am, next_stride_bn = group_stride_am, group_stride_bn
                    next_a_desc = tl.make_tensor_descriptor(next_a_group, shape=[next_gm, K],
                                                            strides=[next_stride_am,
                                                                     tl.constexpr(1)], block_shape=[BLOCK_M, BLOCK_K])
                    next_b_desc = tl.make_tensor_descriptor(next_b_group, shape=[N, K],
                                                            strides=[next_stride_bn,
                                                                     tl.constexpr(1)], block_shape=[BLOCK_N, BLOCK_K])
                    next_load_m, next_load_n = next_off_m, next_off_n
                    group_primed = has_next
                    tlx.amd_sched_barrier()
                elif HYBRID_TILE_PREFETCH:
                    # Keep only the current group's descriptors alive.
                    tlx.amd_sched_barrier()
                    next_tile_idx = tile_idx + NUM_PROGRAMS
                    group_end = last_problem_end + num_tiles
                    has_next = next_tile_idx < group_end
                    safe_next_tile_idx = min(next_tile_idx, group_end - 1)
                    next_off_m, next_off_n = _grouped_tile_offsets(safe_next_tile_idx, last_problem_end, num_m_tiles,
                                                                   num_n_tiles, GROUP_M, BLOCK_M, BLOCK_N)
                    next_a_desc, next_b_desc = a_desc_base, b_desc_base
                    next_load_m, next_load_n = next_off_m, next_off_n
                    tlx.amd_sched_barrier()

                if ALLOW_MN_TAILS and OPTIMIZE_CROSS_TILE:
                    # Both within-group and cross-group lookahead need the next
                    # tile's remaining extents, not the current tile's bounds.
                    if DIRECT_LOAD_OFFSETS:
                        next_rows = next_gm if CROSS_GROUP_PREFETCH else gm
                        next_a_desc = tlx.update_tensor_descriptor(
                            next_a_desc, set_bounds=[tl.maximum(next_rows - next_off_m, 0), K])
                        next_b_desc = tlx.update_tensor_descriptor(next_b_desc,
                                                                   set_bounds=[tl.maximum(N - next_off_n, 0), K])
                    else:
                        next_a_desc = tlx.update_tensor_descriptor(next_a_desc, add_offsets=[next_off_m, 0],
                                                                   clamp_bounds=True)
                        next_b_desc = tlx.update_tensor_descriptor(next_b_desc, add_offsets=[next_off_n, 0],
                                                                   clamp_bounds=True)
                        next_load_m, next_load_n = 0, 0

                # Peel one full ring rotation. Once a current-tile slot has
                # been read into registers, recycle it for the matching K
                # block of the next tile handled by this persistent program.
                # Prefer the late unroll hint over static_range. Early
                # expansion lets LICM hoist both iterations' LDS view setup
                # across the group loop, increasing C-store barrier skew.
                for j in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
                    consumer_j = steady_end + j
                    acc, a3, b3 = _tdm_accumulate_subtiles(acc, a0, b0, a_desc, b_desc, a_buf, b_buf, consumer_j, j, 0,
                                                           BLOCK_K, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN,
                                                           LIMIT_DOT_LIFETIME)
                    _tdm_issue_loads(
                        next_a_desc,
                        next_b_desc,
                        a_buf,
                        b_buf,
                        j,
                        next_load_m,
                        next_load_n,
                        has_next,
                        BLOCK_K,
                        NUM_BUFFERS,
                        CLUSTER_SIZE,
                        CLUSTER_MULTICAST,
                    )
                    acc, a0, b0 = _tdm_wait_and_finish_k_block(acc, a3, b3, a_buf, b_buf, consumer_j + 1, NUM_BUFFERS,
                                                               BLOCK_M, BLOCK_N, SUBTILE_LEN, LIMIT_DOT_LIFETIME)
            else:
                for i in tl.range(0, K_ITERS):
                    acc, a3, b3 = _tdm_accumulate_subtiles(acc, a0, b0, a_desc, b_desc, a_buf, b_buf, consumer,
                                                           producer, L2_PREFETCH_DISTANCE, BLOCK_K, NUM_BUFFERS,
                                                           BLOCK_M, BLOCK_N, SUBTILE_LEN, LIMIT_DOT_LIFETIME)
                    consumer += 1
                    pred = (i + 1) - epilogue_lb
                    pred = (pred >> 31) & 1
                    producer = _tdm_issue_loads(
                        a_desc,
                        b_desc,
                        a_buf,
                        b_buf,
                        producer,
                        0,
                        0,
                        pred,
                        BLOCK_K,
                        NUM_BUFFERS,
                        CLUSTER_SIZE,
                        CLUSTER_MULTICAST,
                    )
                    # Issue first so the next producer group can overlap the
                    # final WMMA, then wait until the older consumer group is
                    # complete.
                    acc, a0, b0 = _tdm_wait_and_finish_k_block(acc, a3, b3, a_buf, b_buf, consumer, NUM_BUFFERS,
                                                               BLOCK_M, BLOCK_N, SUBTILE_LEN, LIMIT_DOT_LIFETIME)

            if HYBRID_TILE_PREFETCH:
                tlx.amd_sched_barrier()
                for part in tl.static_range(BLOCK_M // 32):
                    if part >= 2:
                        # The older output must finish before its slot is
                        # overwritten. Keep the newer chunk in flight.
                        tlx.async_amd_descriptor_wait(1)
                    c_part = _tdm_c_store_chunk(acc, part)
                    c_part = c_part.to(c_dtype)
                    c_view = c_buf[part % 2]
                    c_desc = tlx.update_tensor_descriptor(c_desc_base, add_offsets=[off_m + part * 32, off_n],
                                                          clamp_bounds=ALLOW_MN_TAILS)
                    tlx.local_store(c_view, c_part)
                    tlx.async_amd_descriptor_store(c_desc, c_view, [0, 0], clamp_bounds=False)
                # The last two output stores overlap the next tile's entry.
                # Its input waits also retire them before C staging is reused.
            elif OPTIMIZE_CROSS_TILE:
                # Direct C stores leave both input rings intact without a
                # full C tile in LDS or serialized stores through a small one.
                tlx.amd_sched_barrier()
                cols = off_n + tl.arange(0, BLOCK_N)
                if POINTER_TABLES:
                    rows = off_m.to(tl.int64) + tl.arange(0, BLOCK_M)
                    c_ptrs = c_group + rows[:, None] * group_stride_cm + cols[None, :]
                else:
                    rows = (m_start + off_m).to(tl.int64) + tl.arange(0, BLOCK_M)
                    c_ptrs = c_packed + rows[:, None] * group_stride_cm + cols[None, :]
                c = acc.to(c_dtype)
                tlx.amd_sched_barrier()
                if ALLOW_MN_TAILS:
                    tl.store(c_ptrs, c, mask=(rows[:, None] < gm) & (cols[None, :] < N))
                else:
                    tl.store(c_ptrs, c)

            else:
                c_view = c_buf[0]
                c = acc.to(c_dtype)
                # The descriptor update is pure, so LICM otherwise hoists it into
                # the tile preheader and lengthens its live range through the K
                # loop. Keep it in the epilogue to avoid an extra gfx1250 VGPR-MSB
                # mode transition in the steady body.
                tlx.amd_sched_barrier()
                c_desc = tlx.update_tensor_descriptor(c_desc_base, add_offsets=[off_m, off_n],
                                                      clamp_bounds=ALLOW_MN_TAILS)
                tlx.amd_sched_barrier()
                if USE_FULL_C_TILE:
                    tlx.local_store(c_view, c)
                    tlx.async_amd_descriptor_store(c_desc, c_view, [0, 0], clamp_bounds=False)
                else:
                    c_pair = tl.reshape(c, (BLOCK_M, 2, BLOCK_K)).permute(0, 2, 1)
                    c0, c1 = tl.split(c_pair)
                    tlx.local_store(c_view, c0)
                    tlx.async_amd_descriptor_store(c_desc, c_view, [0, 0], clamp_bounds=False)
                    # c1 reuses the same LDS view immediately, so this wait is
                    # still required; the second-half store is drained lazily.
                    tlx.async_amd_descriptor_wait(0)
                    tlx.local_store(c_view, c1)
                    tlx.async_amd_descriptor_store(c_desc, c_view, [0, BLOCK_K], clamp_bounds=ALLOW_MN_TAILS)

            tile_idx += NUM_PROGRAMS

        last_problem_end += num_tiles
        if CROSS_GROUP_PREFETCH:
            if not prefetched_boundary:
                # Groups with no tile still advance the boundary pipeline.
                if POINTER_TABLES:
                    carried_fourth_m_end = tl.load(group_offsets + 3 * (g + 4))
                else:
                    carried_fourth_m_end = tl.load(group_offsets + g + 5)
            if POINTER_TABLES:
                carried_m_start = tl.full((), 0, tl.int32)
            else:
                carried_m_start = m_end
            carried_m_end = following_m_end
            carried_following_m_end, carried_third_m_end = third_m_end, fourth_m_end

    # At kernel exit, rely on s_endpgm to drain the final async TDM
    # local-to-global store. Keep the explicit wait here as a marker in case
    # target semantics require restoring it.
    # tlx.async_amd_descriptor_wait(0)


@triton.jit
def grouped_gemm_tdm_list_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size,
    K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    C_STAGING_MODE: tl.constexpr,
    CROSS_TILE_PREFETCH: tl.constexpr,
    XCD_REMAP_MODE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr = 1,
    CLUSTER_MULTICAST: tl.constexpr = True,
    ALLOW_MN_TAILS: tl.constexpr = True,
):
    """Ragged-M GEMM over independently allocated A/B/C matrices.

    A is row-major [M_g, K], B is column-major [K, N], and C is row-major
    [M_g, N]. The first six arguments match gfx950 grouped_gemm_kernel:
    three int64 pointer tables, int32 [G, 3] sizes (M, N, K), int32 [G, 3]
    leading strides (lda, ldb, ldc), and the group count. Flat tables work too.
    The caller must validate common N/K, K matching the constexpr argument,
    full K tiles covering NUM_BUFFERS, 16-byte-aligned matrix bases, and A/B
    leading strides divisible by 8. M/N tails and arbitrary C leading strides
    are supported with cluster size 1. Set ALLOW_MN_TAILS=False only when M/N
    cover full tiles and ldc is divisible by 8; clustering requires this mode.
    No offsets, descriptor buffers, or matrix repacking are required.
    """
    N = tl.load(group_gemm_sizes + 1)
    _grouped_gemm_tdm_impl(
        group_a_ptrs,
        group_b_ptrs,
        group_c_ptrs,
        group_gemm_sizes,
        group_size,
        N,
        K,
        0,
        K,
        N,
        K,
        NUM_PROGRAMS,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
        NUM_BUFFERS,
        L2_PREFETCH_DISTANCE,
        C_STAGING_MODE,
        CROSS_TILE_PREFETCH,
        XCD_REMAP_MODE,
        NUM_XCDS,
        XCD_CHUNK,
        CLUSTER_SIZE,
        CLUSTER_MULTICAST,
        POINTER_TABLES=True,
        group_lds=g_lds,
        ALLOW_MN_TAILS=ALLOW_MN_TAILS,
    )


def _validate_grouped_gemm_cluster_config(m_list, n, num_programs, *, block_m, block_n, block_k, group_m,
                                          tdm_pipeline_depth, l2_prefetch_distance, c_staging_mode, cross_tile_prefetch,
                                          auto_config, xcd_remap_mode, num_xcds, xcd_chunk):
    """Require identical cluster control flow and the tested operand-sharing map."""
    if (block_m, block_n, block_k, group_m, tdm_pipeline_depth, l2_prefetch_distance,
            c_staging_mode) != (256, 256, 128, 4, 2, 0, 0) or not cross_tile_prefetch or auto_config:
        raise InvalidInput("clustering requires the 256x256x128 depth-2 hybrid, group_m=4, "
                           "no L2 prefetch, and auto_config=False")
    if (xcd_remap_mode, num_xcds, xcd_chunk) != ("chunked", 8, 2):
        raise InvalidInput("clustering requires chunked XCD remapping with num_xcds=8 and xcd_chunk=2")
    if not m_list or min(m_list) <= 0 or len(set(m_list)) != 1 or m_list[0] % (4 * block_m) or n % (2 * block_n):
        raise InvalidInput("clustering requires equal nonempty groups, M divisible by 1024, and N divisible by 512")
    tiles_per_group = (m_list[0] // block_m) * (n // block_n)
    if num_programs % 16 or tiles_per_group % num_programs:
        raise InvalidInput("clustering requires num_programs divisible by 16 and each group's tile count "
                           "divisible by num_programs; set num_programs explicitly if necessary")


def _validate_grouped_gemm_tdm_list_inputs(group_a, group_b):
    """Check tensor metadata without allocating outputs or device tables."""
    if not isinstance(group_a, (list, tuple)) or not isinstance(group_b, (list, tuple)):
        raise InvalidInput("group_a and group_b must be lists or tuples")
    if not group_a or len(group_a) != len(group_b):
        raise InvalidInput("group_a and group_b must have the same nonzero length")

    m_list = []
    device = None
    n = k = None
    for index, (a, b) in enumerate(zip(group_a, group_b)):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or a.ndim != 2 or b.ndim != 2:
            raise InvalidInput(f"group {index} requires rank-2 tensor inputs")
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise InvalidInput("the TDM list kernel requires FP16 inputs")
        if device is None:
            device = a.device
            k, n = a.shape[1], b.shape[1]
        if a.device != device or b.device != device:
            raise InvalidInput("all grouped GEMM inputs must be on the same device")
        if a.shape[1] <= 0 or min(b.shape) <= 0:
            raise InvalidInput("grouped GEMM N and K dimensions must be positive")
        if a.shape[1] != b.shape[0]:
            raise InvalidInput(f"group {index} has mismatched reduction dimensions")
        if a.shape[1] != k or b.shape[1] != n:
            raise InvalidInput("the TDM list kernel requires the same N and K in every group")
        if a.stride() != (k, 1) or b.stride() != (1, k):
            raise InvalidInput("A must be row-major [M, K] and B column-major [K, N], with canonical strides")
        if a.data_ptr() % 16 or b.data_ptr() % 16:
            raise InvalidInput("TDM matrix base pointers must be 16-byte aligned")
        m_list.append(a.shape[0])

    if torch.is_grad_enabled() and any(t.requires_grad for t in (*group_a, *group_b)):
        raise InvalidInput("the TDM list kernel is forward-only; use torch.no_grad()")
    if sum(m_list) >= 1 << 31:
        raise InvalidInput("the TDM scheduler requires sum(M) to fit in int32")
    return m_list, n, k, device


def _make_grouped_gemm_tdm_list_args(group_a, group_b, group_c=None):
    """Validate the list contract and construct metadata without packing inputs."""
    m_list, n, k, device = _validate_grouped_gemm_tdm_list_inputs(group_a, group_b)
    if group_c is not None and (not isinstance(group_c, (list, tuple)) or len(group_c) != len(group_a)):
        raise InvalidInput("group_c must have one output tensor per group")
    outputs = ([torch.empty((m, n), device=device, dtype=torch.float16)
                for m in m_list] if group_c is None else list(group_c))
    for index, (c, m) in enumerate(zip(outputs, m_list)):
        if not isinstance(c, torch.Tensor) or c.device != device or c.dtype != torch.float16:
            raise InvalidInput(f"output {index} must be an FP16 tensor on the input device")
        if c.shape != (m, n) or c.stride() != (n, 1):
            raise InvalidInput(f"output {index} must be row-major [{m}, {n}] with canonical strides")
        if c.data_ptr() % 16:
            raise InvalidInput("TDM matrix base pointers must be 16-byte aligned")
    pointers = [
        torch.tensor([tensor.data_ptr()
                      for tensor in group], dtype=torch.int64, device=device)
        for group in (group_a, group_b, outputs)
    ]
    group_sizes = torch.tensor([[m, n, k] for m in m_list], dtype=torch.int32, device=device)
    group_lds = torch.tensor([[a.stride(0), b.stride(1), c.stride(0)] for a, b, c in zip(group_a, group_b, outputs)],
                             dtype=torch.int32, device=device)
    return (*pointers, group_sizes, group_lds, outputs, m_list, n, k)


_GROUPED_GEMM_INTERFACE_CONFIG = dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128, GROUP_SIZE_M=4,
                                      NUM_BUFFERS=2, C_STAGING_MODE=0, CROSS_TILE_PREFETCH=True, L2_PREFETCH_DISTANCE=0,
                                      XCD_REMAP_MODE=2, NUM_XCDS=8, XCD_CHUNK=2, CLUSTER_SIZE=4, CLUSTER_MULTICAST=True,
                                      num_warps=4, waves_per_eu=1)

# Validated (BM, BN, depth, C staging, cross-tile prefetch) schedules. In
# particular, larger rings/full dedicated C tiles can exceed the LDS budget.
_GROUPED_GEMM_INTERFACE_SCHEDULES = {
    (128, 128, 2, 0, False),
    (128, 128, 3, 0, False),
    (128, 128, 4, 0, False),
    (128, 256, 2, 1, True),
    (128, 256, 3, 0, False),
    (256, 128, 2, 1, True),
    (256, 256, 2, 0, False),
    (256, 256, 2, 0, True),
    (256, 256, 2, 1, True),
}


def _grouped_gemm_interface_config(config, k):
    """Validate gfx1250 tuning without silently replacing requested settings."""
    if config is not None and not isinstance(config, dict):
        raise InvalidInput("gfx1250 grouped GEMM config must be a dict or None")
    config = {} if config is None else config
    unknown = config.keys() - (_GROUPED_GEMM_INTERFACE_CONFIG.keys() | {"NUM_SM"})
    if unknown:
        raise InvalidInput(f"unsupported gfx1250 grouped GEMM config keys: {sorted(map(str, unknown))}")
    cfg = dict(_GROUPED_GEMM_INTERFACE_CONFIG, **config)
    for key, value in cfg.items():
        if key in ("CROSS_TILE_PREFETCH", "CLUSTER_MULTICAST"):
            if type(value) is not bool:
                raise InvalidInput(f"{key} must be a bool")
        elif type(value) is not int:
            raise InvalidInput(f"{key} must be an int")
    for key, allowed in {
            "BLOCK_SIZE_M": (128, 256),
            "BLOCK_SIZE_N": (128, 256),
            "BLOCK_SIZE_K": (128, ),
            "NUM_BUFFERS": (2, 3, 4),
            "C_STAGING_MODE": (0, 1),
            "XCD_REMAP_MODE": (0, 1, 2),
            "CLUSTER_SIZE": (1, 2, 4),
            "num_warps": (4, ),
            "waves_per_eu": (1, ),
    }.items():
        if cfg[key] not in allowed:
            raise InvalidInput(f"unsupported gfx1250 {key}={cfg[key]}; supported values: {allowed}")
    for key in ("GROUP_SIZE_M", "NUM_XCDS", "XCD_CHUNK", "NUM_SM"):
        if key in cfg and not 0 < cfg[key] < 1 << 31:
            raise InvalidInput(f"{key} must be a positive int32")
    if not 0 <= cfg["L2_PREFETCH_DISTANCE"] < 1 << 31:
        raise InvalidInput("L2_PREFETCH_DISTANCE must be a nonnegative int32")
    if cfg["NUM_XCDS"] * cfg["XCD_CHUNK"] >= 1 << 31:
        raise InvalidInput("NUM_XCDS * XCD_CHUNK must fit in int32")
    if type(k) is not int or not 0 < k < 1 << 31 or k % cfg["BLOCK_SIZE_K"]:
        raise InvalidInput("gfx1250 grouped GEMM requires full K tiles: positive int32 K divisible by 128")
    depth = cfg["NUM_BUFFERS"]
    if k // 128 < depth:
        raise InvalidInput("K must contain enough blocks to fill the TDM pipeline")
    if cfg["C_STAGING_MODE"] == 1 and (depth * 128) % cfg["BLOCK_SIZE_N"]:
        raise InvalidInput("dedicated C staging requires a full C buffer (NUM_BUFFERS * 128 divisible by BLOCK_SIZE_N)")
    if cfg["CROSS_TILE_PREFETCH"]:
        if depth != 2 or (k // 128) % 2:
            raise InvalidInput("cross-tile prefetch requires depth 2 and an even number of K blocks")
        if cfg["C_STAGING_MODE"] == 0 and (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]) != (256, 256):
            raise InvalidInput("alias-C cross-tile prefetch requires a 256x256 tile")
    schedule = tuple(cfg[key] for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "NUM_BUFFERS", "C_STAGING_MODE",
                                          "CROSS_TILE_PREFETCH"))
    if schedule not in _GROUPED_GEMM_INTERFACE_SCHEDULES:
        raise InvalidInput(f"unsupported gfx1250 TDM schedule (BM, BN, depth, C staging, cross-tile): {schedule}")
    if cfg["L2_PREFETCH_DISTANCE"] > k // 128:
        raise InvalidInput("L2_PREFETCH_DISTANCE must not exceed the number of K blocks")
    return cfg


def _require_grouped_gemm_gfx1250(device):
    if device.type != "cuda" or not torch.version.hip:
        raise InvalidInput("the TDM grouped GEMM interface requires an AMD gfx1250 device")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if arch != "gfx1250":
        raise InvalidInput(f"the TDM grouped GEMM interface requires gfx1250; got {arch}")


def _grouped_gemm_interface_launch_config(cfg, m_list, n, k, num_sms, *, auto_cluster):
    """Use four-workgroup multicast whenever the cluster contract allows it."""
    cfg = dict(cfg)
    total_tiles = sum(_cdiv_int(m, cfg["BLOCK_SIZE_M"]) * _cdiv_int(n, cfg["BLOCK_SIZE_N"]) for m in m_list)
    num_sm = min(num_sms, total_tiles)
    if total_tiles + num_sm >= 1 << 31:
        raise InvalidInput("gfx1250 grouped GEMM tile count plus NUM_SM must fit in int32")
    if cfg["CLUSTER_SIZE"] > 1:
        try:
            _validate_grouped_gemm_cluster_config(
                m_list, n, num_sm, block_m=cfg["BLOCK_SIZE_M"], block_n=cfg["BLOCK_SIZE_N"],
                block_k=cfg["BLOCK_SIZE_K"], group_m=cfg["GROUP_SIZE_M"], tdm_pipeline_depth=cfg["NUM_BUFFERS"],
                l2_prefetch_distance=cfg["L2_PREFETCH_DISTANCE"], c_staging_mode=cfg["C_STAGING_MODE"],
                cross_tile_prefetch=cfg["CROSS_TILE_PREFETCH"], auto_config=False,
                xcd_remap_mode=next(name for name, value in _XCD_REMAP_MODES.items() if value == cfg["XCD_REMAP_MODE"]),
                num_xcds=cfg["NUM_XCDS"], xcd_chunk=cfg["XCD_CHUNK"])
        except InvalidInput:
            if not auto_cluster:
                raise
            cfg["CLUSTER_SIZE"] = 1
    cfg.update(K=k, NUM_SM=num_sm, ALLOW_MN_TAILS=bool(n % cfg["BLOCK_SIZE_N"]
                                                       or any(m % cfg["BLOCK_SIZE_M"] for m in m_list)))
    return cfg


def _make_grouped_gemm_args(group_A, group_B, config=None):
    """gfx950-compatible setup: return five tables, G, cfg, and output list.

    A is canonical row-major FP16 [M_g, K], B canonical column-major FP16
    [K, N]. Only M may vary across groups. No input packing is performed.
    Config uses gfx950's common tuning names plus TDM-specific options;
    gfx950-only options are rejected. The default is the 256x256 hybrid
    with depth 2 and cluster size 4 when shapes/grid permit it, otherwise 1.
    Explicit cluster requests are validated. All validation precedes allocation.
    Keep inputs, returned tables, and outputs alive through the launch.
    """
    m_list, n, k, device = _validate_grouped_gemm_tdm_list_inputs(group_A, group_B)
    if n >= 1 << 31 or 3 * len(m_list) >= 1 << 31:
        raise InvalidInput("gfx1250 grouped GEMM requires N and 3 * group count to fit in int32")
    cfg = _grouped_gemm_interface_config(config, k)
    if cfg["GROUP_SIZE_M"] * _cdiv_int(n, cfg["BLOCK_SIZE_N"]) >= 1 << 31:
        raise InvalidInput("GROUP_SIZE_M * number of N tiles must fit in int32")
    _require_grouped_gemm_gfx1250(device)
    cfg = _grouped_gemm_interface_launch_config(cfg, m_list, n, k,
                                                cfg["NUM_SM"] if "NUM_SM" in cfg else _active_num_cus(device),
                                                auto_cluster=config is None or "CLUSTER_SIZE" not in config)
    a, b, c, sizes, lds, outputs, _, _, _ = _make_grouped_gemm_tdm_list_args(group_A, group_B)
    # gfx950 emits flat size/stride tables. Views preserve the same allocation.
    return a, b, c, sizes.view(-1), lds.view(-1), len(m_list), cfg, outputs


def _perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, G, cfg):
    """Launch prepared gfx950-style arguments without allocation or readback.

    Use the config returned by _make_grouped_gemm_args. Device table contents
    are trusted: any in-place updates must preserve common N/K, canonical
    layouts, alignment, scheduler/cluster limits, and the chosen tail specialization.
    The raw JIT entry remains available for padded leading strides.
    """
    if not isinstance(cfg, dict) or not {"K", "NUM_SM", "ALLOW_MN_TAILS", "CLUSTER_SIZE"} <= cfg.keys():
        raise InvalidInput("cfg must include K, NUM_SM, ALLOW_MN_TAILS, and CLUSTER_SIZE from _make_grouped_gemm_args")
    if type(G) is not int or G <= 0 or 3 * G >= 1 << 31:
        raise InvalidInput("G must be a positive int32 group count with 3 * G fitting in int32")
    if type(cfg["NUM_SM"]) is not int or not 0 <= cfg["NUM_SM"] < 1 << 31:
        raise InvalidInput("NUM_SM must be a nonnegative int32")
    if type(cfg["ALLOW_MN_TAILS"]) is not bool:
        raise InvalidInput("ALLOW_MN_TAILS must be a bool")
    tuning = _grouped_gemm_interface_config(
        {key: value
         for key, value in cfg.items()
         if key not in ("K", "NUM_SM", "ALLOW_MN_TAILS")}, cfg["K"])
    if tuning["CLUSTER_SIZE"] > 1:
        if cfg["ALLOW_MN_TAILS"]:
            raise InvalidInput("clustered launches require ALLOW_MN_TAILS=False")
        if cfg["NUM_SM"] == 0 or cfg["NUM_SM"] % 16:
            raise InvalidInput("clustered launches require NUM_SM to be positive and divisible by 16")
    tables = (d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds)
    if not all(isinstance(t, torch.Tensor) for t in tables):
        raise InvalidInput("grouped GEMM metadata must be tensors")
    device = d_a_ptrs.device
    for index, table in enumerate(tables):
        dtype = torch.int64 if index < 3 else torch.int32
        shapes = ((G, ), ) if index < 3 else ((3 * G, ), (G, 3))
        if table.device != device or table.dtype != dtype or tuple(
                table.shape) not in shapes or not table.is_contiguous():
            raise InvalidInput(f"metadata table {index} must be contiguous {dtype}, shape in {shapes}, on {device}")
    _require_grouped_gemm_gfx1250(device)
    if cfg["NUM_SM"] == 0:
        return
    specialize_groups = tuning["CROSS_TILE_PREFETCH"] and tuning["BLOCK_SIZE_M"] == tuning["BLOCK_SIZE_N"] == 256
    with torch.cuda.device(device):
        grouped_gemm_tdm_list_kernel[(cfg["NUM_SM"], )](
            *tables, tl.constexpr(G) if specialize_groups else G, K=cfg["K"], NUM_PROGRAMS=cfg["NUM_SM"],
            BLOCK_M=tuning["BLOCK_SIZE_M"], BLOCK_N=tuning["BLOCK_SIZE_N"], BLOCK_K=tuning["BLOCK_SIZE_K"],
            GROUP_M=tuning["GROUP_SIZE_M"], NUM_BUFFERS=tuning["NUM_BUFFERS"],
            L2_PREFETCH_DISTANCE=tuning["L2_PREFETCH_DISTANCE"], C_STAGING_MODE=tuning["C_STAGING_MODE"],
            CROSS_TILE_PREFETCH=tuning["CROSS_TILE_PREFETCH"], XCD_REMAP_MODE=tuning["XCD_REMAP_MODE"],
            NUM_XCDS=tuning["NUM_XCDS"], XCD_CHUNK=tuning["XCD_CHUNK"], CLUSTER_SIZE=tuning["CLUSTER_SIZE"],
            CLUSTER_MULTICAST=tuning["CLUSTER_MULTICAST"], ALLOW_MN_TAILS=cfg["ALLOW_MN_TAILS"],
            enable_wmma_queuing=_grouped_gemm_enable_wmma_queuing(tuning["BLOCK_SIZE_M"], tuning["BLOCK_SIZE_N"],
                                                                  tuning["NUM_BUFFERS"], tuning["C_STAGING_MODE"],
                                                                  tuning["CROSS_TILE_PREFETCH"]),
            num_warps=tuning["num_warps"], waves_per_eu=tuning["waves_per_eu"], ctas_per_cga=(tuning["CLUSTER_SIZE"], 1,
                                                                                              1))


def grouped_gemm(group_a, group_b):
    """``tlx.ops.grouped_gemm`` entry point: one FP16 ``A[i] @ B[i]`` per group.

    Validates the inputs and the gfx1250 restrictions in the module
    docstring, raising ``InvalidInput``, and returns one row-major output per
    group. K with an odd number of 128-wide blocks runs without cross-tile
    prefetch.
    """
    config = None
    _, _, k, _ = _validate_grouped_gemm_tdm_list_inputs(group_a, group_b)
    if k % 128 == 0 and (k // 128) % 2:
        # Cross-tile prefetch needs an even number of K blocks.
        config = {"CROSS_TILE_PREFETCH": False}
    *args, outputs = _make_grouped_gemm_args(group_a, group_b, config)
    _perf_fn(*args)
    return outputs
