import triton
import triton.language as tl
try:
    import triton.language.extra.cuda.tlx as tlx
except ModuleNotFoundError:
    import triton.language.extra.tlx as tlx

@triton.jit
def _attn_bwd_ws(desc_q, desc_k, desc_v, sm_scale, desc_do, desc_dq, desc_dk, desc_dv, desc_m, desc_delta, M_ptr, delta_ptr, H, N_CTX, desc_kt, desc_qt, desc_dot):
  start_n = tl.program_id(axis=0)  # blackwell_fa_ws_pipelined_persistent.py:2181
  head = tl.program_id(axis=1)  # blackwell_fa_ws_pipelined_persistent.py:2182
  batch = tl.program_id(axis=2)  # blackwell_fa_ws_pipelined_persistent.py:2183
  off_chz = batch * H  # blackwell_fa_ws_pipelined_persistent.py:2184
  off_chz_0 = off_chz + head  # blackwell_fa_ws_pipelined_persistent.py:2184
  off_chz_1 = off_chz_0 * N_CTX  # blackwell_fa_ws_pipelined_persistent.py:2184
  num_steps = N_CTX // 128  # blackwell_fa_ws_pipelined_persistent.py:2194
  start_block_n = start_n * 128  # blackwell_fa_ws_pipelined_persistent.py:2195
  k_mma_done = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2205
  k_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2206
  q_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2207
  q_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2208
  do_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2209
  do_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2210
  m_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2211
  m_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2212
  d_fulls = tlx.alloc_barriers(2)  # blackwell_fa_ws_pipelined_persistent.py:2213
  d_empties = tlx.alloc_barriers(2)  # blackwell_fa_ws_pipelined_persistent.py:2214
  ds_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2218
  dsT_tmem_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2219
  qk_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2221
  qk_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2226
  p_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2227
  dp_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2228
  dq_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2229
  dq_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2233
  dv_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2235
  dv_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2239
  dk_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2240
  dk_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2244
  dp_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2254
  k_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2260
  v_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2261
  kt_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2262
  kt_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2263
  qt_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2264
  qt_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2265
  dot_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2266
  dot_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2267
  ds_peer_fulls = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2268
  ds_empties = tlx.alloc_barriers(1)  # blackwell_fa_ws_pipelined_persistent.py:2269
  k_tiles = tlx.local_alloc((128, 128), tl.float16, 1)  # blackwell_fa_ws_pipelined_persistent.py:2276
  v_tiles = tlx.local_alloc((128, 128), tl.float16, 1)  # blackwell_fa_ws_pipelined_persistent.py:2277
  q_tiles = tlx.local_alloc((128, 64), tl.float16, 1)  # blackwell_fa_ws_pipelined_persistent.py:2278
  do_tiles = tlx.local_alloc((128, 64), tl.float16, 1)  # blackwell_fa_ws_pipelined_persistent.py:2279
  ds_tiles = tlx.local_alloc((256, 64), tl.float16, 1)  # blackwell_fa_ws_pipelined_persistent.py:2283
  dq_store_buf = tlx.local_alloc((128, 16), tl.float32, 2)  # blackwell_fa_ws_pipelined_persistent.py:2288
  v_tiles_38 = tlx.local_alloc((128, 64), tl.float16, 1, tlx.storage_kind.tmem, reuse=v_tiles)  # blackwell_fa_ws_pipelined_persistent.py:2277
  k_tiles_39 = tlx.local_alloc((128, 64), tl.float16, 1, tlx.storage_kind.tmem, reuse=k_tiles)  # blackwell_fa_ws_pipelined_persistent.py:2276
  sM_tiles = tlx.local_alloc((1, 128), tl.float32, 1)  # blackwell_fa_ws_pipelined_persistent.py:2300
  sD_tiles = tlx.local_alloc((2, 128), tl.float32, 1)  # blackwell_fa_ws_pipelined_persistent.py:2301
  qk_p_storage_alias = tlx.local_alloc((128,), tl.int32, 128, tlx.storage_kind.tmem)  # blackwell_fa_ws_pipelined_persistent.py:2306
  qk_p_storage_alias_40 = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem, reuse=qk_p_storage_alias)  # blackwell_fa_ws_pipelined_persistent.py:2306
  qk_p_storage_alias_41 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=qk_p_storage_alias)  # blackwell_fa_ws_pipelined_persistent.py:2306
  dp_dq_storage_alias = tlx.local_alloc((128,), tl.int32, 128, tlx.storage_kind.tmem)  # blackwell_fa_ws_pipelined_persistent.py:2322
  dp_dq_storage_alias_42 = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem, reuse=dp_dq_storage_alias)  # blackwell_fa_ws_pipelined_persistent.py:2322
  dp_dq_storage_alias_43 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=dp_dq_storage_alias)  # blackwell_fa_ws_pipelined_persistent.py:2322
  dv_tiles = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)  # blackwell_fa_ws_pipelined_persistent.py:2338
  dk_tiles = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)  # blackwell_fa_ws_pipelined_persistent.py:2339
  qk_p_storage_alias_44 = tlx.local_alloc((64, 128), tl.float32, 2, tlx.storage_kind.tmem, reuse=qk_p_storage_alias)  # blackwell_fa_ws_pipelined_persistent.py:2306
  qk_p_storage_alias_45 = tlx.local_alloc((128, 64), tl.float32, 2, tlx.storage_kind.tmem, reuse=qk_p_storage_alias)  # blackwell_fa_ws_pipelined_persistent.py:2306
  cluster_cta_rank = tlx.cluster_cta_rank()  # blackwell_fa_ws_pipelined_persistent.py:2413
  is_leader = cluster_cta_rank == 0  # blackwell_fa_ws_pipelined_persistent.py:2414
  kt_tiles = tlx.local_alloc((256, 64), tl.float16, 1)  # blackwell_fa_ws_pipelined_persistent.py:2416
  qt_tiles = tlx.local_alloc((64, 128), tl.float16, 1)  # blackwell_fa_ws_pipelined_persistent.py:2419
  dot_tiles = tlx.local_alloc((64, 128), tl.float16, 1)  # blackwell_fa_ws_pipelined_persistent.py:2421
  ds_xchg_tiles = tlx.local_alloc((128, 64), tl.float16, 1)  # blackwell_fa_ws_pipelined_persistent.py:2424
  with tlx.async_tasks():
    with tlx.async_task("default"):
      var_1 = tlx.remote_view(qk_empties_18, 0)  # blackwell_fa_ws_pipelined_persistent.py:2047
      var_2 = tlx.remote_view(p_fulls_19, 0)  # blackwell_fa_ws_pipelined_persistent.py:2048
      var_4 = tlx.remote_view(dsT_tmem_fulls_16, 0)  # blackwell_fa_ws_pipelined_persistent.py:2076
      peer_rank = 1 - cluster_cta_rank  # blackwell_fa_ws_pipelined_persistent.py:2080
      if is_leader:
        own_tmem = tlx.subslice(var_3)  # blackwell_fa_ws_pipelined_persistent.py:2083
        peer_tmem = tlx.subslice(var_3)  # blackwell_fa_ws_pipelined_persistent.py:2084
        own_smem_55 = tlx.local_slice(own_smem, [0, 0], [128, 64])  # blackwell_fa_ws_pipelined_persistent.py:2086
        var_5_0 = own_smem_55
        var_5_1 = own_tmem
        var_5_2 = peer_tmem
      else:
        own_tmem = tlx.subslice(var_3)  # blackwell_fa_ws_pipelined_persistent.py:2088
        peer_tmem = tlx.subslice(var_3)  # blackwell_fa_ws_pipelined_persistent.py:2090
        own_smem_55 = tlx.local_slice(own_smem, [128, 0], [128, 64])  # blackwell_fa_ws_pipelined_persistent.py:2091
        var_5_0 = own_smem_55
        var_5_1 = own_tmem
        var_5_2 = peer_tmem
      var_6 = tlx.remote_view(dp_empties_27, 0)  # blackwell_fa_ws_pipelined_persistent.py:2097
      arg102 = 0
      for arg101 in range(0, num_steps, 1):
        phase = arg102 & 1  # warp_spec.py:8
        buf_idx = arg102 % 2  # warp_spec.py:7
        phase_55 = arg102 // 2  # warp_spec.py:8
        phase_56 = phase_55 & 1  # warp_spec.py:8
        tlx.barrier_wait(qk_fulls_17, phase)  # blackwell_fa_ws_pipelined_persistent.py:2019
        tlx.barrier_wait(m_fulls_9, phase)  # blackwell_fa_ws_pipelined_persistent.py:2020
        qkT_57 = tlx.local_load(qkT)  # blackwell_fa_ws_pipelined_persistent.py:2022
        sT = tlx.local_load(m)  # blackwell_fa_ws_pipelined_persistent.py:2028
        sT_58 = tl.expand_dims(sT, axis=0)  # blackwell_fa_ws_pipelined_persistent.py:2028
        sT_60 = tl.inline_asm_elementwise(qkT_57, sT_58)  # blackwell_fa_ws_pipelined_persistent.py:2028
        pT = tl.math.exp2(sT_60)  # blackwell_fa_ws_pipelined_persistent.py:2032
        tlx.named_barrier_wait(10, 256)  # blackwell_fa_ws_pipelined_persistent.py:2040
        tlx.local_store(var_0, pT.to(tl.float16))  # blackwell_fa_ws_pipelined_persistent.py:2041
        tlx.barrier_arrive(var_1)  # blackwell_fa_ws_pipelined_persistent.py:2047
        tlx.barrier_arrive(var_2)  # blackwell_fa_ws_pipelined_persistent.py:2048
        tlx.barrier_wait(dp_fulls_20, phase)  # blackwell_fa_ws_pipelined_persistent.py:2054
        dpT_61 = tlx.local_load(dpT)  # blackwell_fa_ws_pipelined_persistent.py:2055
        tlx.barrier_wait(var_16, phase_56)  # blackwell_fa_ws_pipelined_persistent.py:2056
        dsT = tlx.local_load(Di)  # blackwell_fa_ws_pipelined_persistent.py:2060
        tlx.barrier_arrive(m_empties_10)  # blackwell_fa_ws_pipelined_persistent.py:2058
        tlx.barrier_arrive(var_17)  # blackwell_fa_ws_pipelined_persistent.py:2059
        dsT_62 = tl.expand_dims(dsT, axis=0)  # blackwell_fa_ws_pipelined_persistent.py:2060
        dsT_64 = tl.inline_asm_elementwise(dpT_61, dsT_62)  # blackwell_fa_ws_pipelined_persistent.py:2060
        dsT_65 = tl.inline_asm_elementwise(pT, dsT_64)  # blackwell_fa_ws_pipelined_persistent.py:2060
        tlx.named_barrier_wait(11, 256)  # blackwell_fa_ws_pipelined_persistent.py:2064
        tlx.local_store(var_3, dsT_65.to(tl.float16))  # blackwell_fa_ws_pipelined_persistent.py:2065
        tlx.barrier_arrive(var_4)  # blackwell_fa_ws_pipelined_persistent.py:2076
        var_18 = phase ^ 1  # blackwell_fa_ws_pipelined_persistent.py:2079
        tlx.barrier_wait(ds_empties_37, var_18)  # blackwell_fa_ws_pipelined_persistent.py:2079
        own_data = tlx.local_load(var_5_1)  # blackwell_fa_ws_pipelined_persistent.py:2092
        tlx.local_store(var_5_0, own_data)  # blackwell_fa_ws_pipelined_persistent.py:2093
        peer_data = tlx.local_load(var_5_2)  # blackwell_fa_ws_pipelined_persistent.py:2094
        tlx.barrier_arrive(var_6)  # blackwell_fa_ws_pipelined_persistent.py:2097
        tlx.local_store(var_7, peer_data)  # blackwell_fa_ws_pipelined_persistent.py:2098
        tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2099
        tlx.barrier_expect_bytes(ds_peer_fulls_36, 16384)  # blackwell_fa_ws_pipelined_persistent.py:2101
        tlx.async_remote_shmem_copy(var_7, var_5_0, peer_rank, ds_peer_fulls_36)  # blackwell_fa_ws_pipelined_persistent.py:2102
        blk_idx_67 = arg102 + 1  # blackwell_fa_ws_pipelined_persistent.py:2116
        arg102 = blk_idx_67
      tlx.barrier_wait(dv_fulls_23, 0)  # blackwell_fa_ws_pipelined_persistent.py:2591
      dv_slice_46 = tlx.subslice(dv_slice)  # blackwell_fa_ws_pipelined_persistent.py:2594
      dv = tlx.local_load(dv_slice_46)  # blackwell_fa_ws_pipelined_persistent.py:2599
      tlx.async_descriptor_store_wait(0)  # blackwell_fa_ws_pipelined_persistent.py:2600
      tlx.local_store(var_8, dv.to(tl.float16))  # blackwell_fa_ws_pipelined_persistent.py:2601
      tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2602
      tlx.async_descriptor_store(desc_dv, var_8, [batch, head, start_block_n, 0])  # blackwell_fa_ws_pipelined_persistent.py:2602
      dv_slice_47 = tlx.subslice(dv_slice)  # blackwell_fa_ws_pipelined_persistent.py:2594
      dv_48 = tlx.local_load(dv_slice_47)  # blackwell_fa_ws_pipelined_persistent.py:2599
      tlx.async_descriptor_store_wait(0)  # blackwell_fa_ws_pipelined_persistent.py:2600
      tlx.local_store(var_8, dv_48.to(tl.float16))  # blackwell_fa_ws_pipelined_persistent.py:2601
      tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2602
      tlx.async_descriptor_store(desc_dv, var_8, [batch, head, start_block_n, 64])  # blackwell_fa_ws_pipelined_persistent.py:2602
      var_11 = tlx.remote_view(dv_empties_24, 0)  # blackwell_fa_ws_pipelined_persistent.py:2608
      tlx.barrier_arrive(var_11)  # blackwell_fa_ws_pipelined_persistent.py:2608
      tlx.barrier_wait(dk_fulls_25, 0)  # blackwell_fa_ws_pipelined_persistent.py:2611
      tlx.barrier_wait(k_mma_done_3, 0)  # blackwell_fa_ws_pipelined_persistent.py:2614
      dk_slice_49 = tlx.subslice(dk_slice)  # blackwell_fa_ws_pipelined_persistent.py:2616
      dk = tlx.local_load(dk_slice_49)  # blackwell_fa_ws_pipelined_persistent.py:2621
      dk_51 = dk * sm_scale  # blackwell_fa_ws_pipelined_persistent.py:2622
      tlx.async_descriptor_store_wait(0)  # blackwell_fa_ws_pipelined_persistent.py:2623
      tlx.local_store(var_12, dk_51.to(tl.float16))  # blackwell_fa_ws_pipelined_persistent.py:2624
      tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2625
      tlx.async_descriptor_store(desc_dk, var_12, [batch, head, start_block_n, 0])  # blackwell_fa_ws_pipelined_persistent.py:2625
      dk_slice_52 = tlx.subslice(dk_slice)  # blackwell_fa_ws_pipelined_persistent.py:2616
      dk_53 = tlx.local_load(dk_slice_52)  # blackwell_fa_ws_pipelined_persistent.py:2621
      dk_54 = dk_53 * sm_scale  # blackwell_fa_ws_pipelined_persistent.py:2622
      tlx.async_descriptor_store_wait(0)  # blackwell_fa_ws_pipelined_persistent.py:2623
      tlx.local_store(var_12, dk_54.to(tl.float16))  # blackwell_fa_ws_pipelined_persistent.py:2624
      tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2625
      tlx.async_descriptor_store(desc_dk, var_12, [batch, head, start_block_n, 64])  # blackwell_fa_ws_pipelined_persistent.py:2625
      tlx.async_descriptor_store_wait(0)  # blackwell_fa_ws_pipelined_persistent.py:2630
      var_15 = tlx.remote_view(dk_empties_26, 0)  # blackwell_fa_ws_pipelined_persistent.py:2641
      tlx.barrier_arrive(var_15)  # blackwell_fa_ws_pipelined_persistent.py:2641
    with tlx.async_task(num_warps=4, registers=88):
      cst = tl.full([128, 64], 0.693147182, tl.float32)
      dq_m_offset = arg103 * 64  # blackwell_fa_ws_pipelined_persistent.py:2657
      var_2 = tlx.remote_view(var_1, 0)  # blackwell_fa_ws_pipelined_persistent.py:2664
      arg171 = 0
      arg172 = 0
      for arg170 in range(0, arg152, 1):
        phase = arg171 & 1  # warp_spec.py:8
        tlx.barrier_wait(var_0, phase)  # blackwell_fa_ws_pipelined_persistent.py:2655
        packed_row_base = arg172 + dq_m_offset  # blackwell_fa_ws_pipelined_persistent.py:2658
        packed_row_base_53 = packed_row_base * 2  # blackwell_fa_ws_pipelined_persistent.py:2658
        dq_full_54 = tlx.local_load(dq_full)  # blackwell_fa_ws_pipelined_persistent.py:2660
        tlx.barrier_arrive(var_2)  # blackwell_fa_ws_pipelined_persistent.py:2664
        dq_full_55 = dq_full_54 * cst  # blackwell_fa_ws_pipelined_persistent.py:2665
        dq_slices = tl.reshape(dq_full_55, [128, 2, 32])
        dq_slices_56 = tl.trans(dq_slices)
        (dq_slices_57, dq_slices_58) = tl.split(dq_slices_56)
        dq_slices_59 = tl.reshape(dq_slices_57, [128, 2, 16])
        dq_slices_60 = tl.trans(dq_slices_59)
        (dq_slices_61, dq_slices_62) = tl.split(dq_slices_60)
        dq_slices_63 = tl.reshape(dq_slices_58, [128, 2, 16])
        dq_slices_64 = tl.trans(dq_slices_63)
        (dq_slices_65, dq_slices_66) = tl.split(dq_slices_64)
        tlx.async_descriptor_store_wait(1)  # blackwell_fa_ws_pipelined_persistent.py:2669
        tlx.local_store(dq_smem, dq_slices_61)  # blackwell_fa_ws_pipelined_persistent.py:2670
        tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2671
        tlx.async_descriptor_store(arg109, dq_smem, [arg102, arg141, packed_row_base_53, 0], store_reduce="add")  # blackwell_fa_ws_pipelined_persistent.py:2671
        tlx.async_descriptor_store_wait(1)  # blackwell_fa_ws_pipelined_persistent.py:2669
        tlx.local_store(dq_smem_52, dq_slices_62)  # blackwell_fa_ws_pipelined_persistent.py:2670
        tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2671
        tlx.async_descriptor_store(arg109, dq_smem_52, [arg102, arg141, packed_row_base_53, 16], store_reduce="add")  # blackwell_fa_ws_pipelined_persistent.py:2671
        tlx.async_descriptor_store_wait(1)  # blackwell_fa_ws_pipelined_persistent.py:2669
        tlx.local_store(dq_smem, dq_slices_65)  # blackwell_fa_ws_pipelined_persistent.py:2670
        tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2671
        tlx.async_descriptor_store(arg109, dq_smem, [arg102, arg141, packed_row_base_53, 32], store_reduce="add")  # blackwell_fa_ws_pipelined_persistent.py:2671
        tlx.async_descriptor_store_wait(1)  # blackwell_fa_ws_pipelined_persistent.py:2669
        tlx.local_store(dq_smem_52, dq_slices_66)  # blackwell_fa_ws_pipelined_persistent.py:2670
        tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2671
        tlx.async_descriptor_store(arg109, dq_smem_52, [arg102, arg141, packed_row_base_53, 48], store_reduce="add")  # blackwell_fa_ws_pipelined_persistent.py:2671
        curr_m_67 = arg172 + 128  # blackwell_fa_ws_pipelined_persistent.py:2715
        blk_idx = arg171 + 1  # blackwell_fa_ws_pipelined_persistent.py:2716
        arg171 = blk_idx
        arg172 = curr_m_67
      tlx.async_descriptor_store_wait(0)  # blackwell_fa_ws_pipelined_persistent.py:2719
    with tlx.async_task(num_warps=1, registers=88):
      if arg142:
        tlx.barrier_wait(blk_idx, 0)  # blackwell_fa_ws_pipelined_persistent.py:1376
        tlx.barrier_wait(blk_idx_50, 0)  # blackwell_fa_ws_pipelined_persistent.py:1377
        tlx.barrier_wait(blk_idx_51, 0)  # blackwell_fa_ws_pipelined_persistent.py:1384
        tlx.barrier_wait(blk_idx_52, 1)  # blackwell_fa_ws_pipelined_persistent.py:1385
        qT_53 = tlx.local_trans(qT)  # blackwell_fa_ws_pipelined_persistent.py:1386
        tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:1387
        tlx.async_dot(blk_idx_56, qT_53, blk_idx_57, use_acc=False, pred=True, mBarriers=[blk_idx_54, blk_idx_55], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1387
        tlx.barrier_wait(blk_idx_58, 0)  # blackwell_fa_ws_pipelined_persistent.py:1397
        doT_59 = tlx.local_trans(doT)  # blackwell_fa_ws_pipelined_persistent.py:1398
        tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:1399
        tlx.async_dot(blk_idx_62, doT_59, blk_idx_63, use_acc=False, pred=True, mBarriers=[blk_idx_60, blk_idx_61], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1399
        tlx.barrier_wait(blk_idx_64, 0)  # blackwell_fa_ws_pipelined_persistent.py:1410
        tlx.barrier_wait(blk_idx_65, 0)  # blackwell_fa_ws_pipelined_persistent.py:1411
        tlx.barrier_wait(blk_idx_66, 1)  # blackwell_fa_ws_pipelined_persistent.py:1412
        tlx.async_dot(blk_idx_68, blk_idx_69, blk_idx_70, use_acc=False, pred=True, mBarriers=[blk_idx_67], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1413
        tlx.barrier_wait(blk_idx_71, 1)  # blackwell_fa_ws_pipelined_persistent.py:1427
        tlx.barrier_wait(blk_idx_72, 0)  # blackwell_fa_ws_pipelined_persistent.py:1430
        dsT_view_82 = tlx.local_trans(dsT_view)  # blackwell_fa_ws_pipelined_persistent.py:1486
        arg171 = 1
        for arg170 in range(1, arg152, 1):
          phase_98 = arg171 & 1  # warp_spec.py:8
          tlx.barrier_wait(blk_idx_51, phase_98)  # blackwell_fa_ws_pipelined_persistent.py:1435
          blk_idx_99 = phase_98 ^ 1  # blackwell_fa_ws_pipelined_persistent.py:1436
          tlx.barrier_wait(blk_idx_52, blk_idx_99)  # blackwell_fa_ws_pipelined_persistent.py:1436
          prev_blk_idx_100 = arg171 - 1  # blackwell_fa_ws_pipelined_persistent.py:1437
          phase_101 = prev_blk_idx_100 & 1  # warp_spec.py:8
          blk_idx_102 = phase_101 ^ 1  # blackwell_fa_ws_pipelined_persistent.py:1439
          tlx.barrier_wait(blk_idx_73, blk_idx_102)  # blackwell_fa_ws_pipelined_persistent.py:1439
          tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:1441
          tlx.async_dot(blk_idx_56, qT_53, blk_idx_57, use_acc=False, pred=True, mBarriers=[blk_idx_54, blk_idx_55], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1441
          tlx.barrier_wait(blk_idx_74, phase_101)  # blackwell_fa_ws_pipelined_persistent.py:1453
          tlx.barrier_wait(blk_idx_75, phase_101)  # blackwell_fa_ws_pipelined_persistent.py:1454
          blk_idx_103 = arg170 - 1  # blackwell_fa_ws_pipelined_persistent.py:1459
          blk_idx_104 = blk_idx_103 > 0  # blackwell_fa_ws_pipelined_persistent.py:1459
          tlx.async_dot(blk_idx_78, blk_idx_79, blk_idx_80, use_acc=blk_idx_104, pred=True, mBarriers=[blk_idx_76, blk_idx_77], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1455
          tlx.barrier_wait(blk_idx_58, phase_98)  # blackwell_fa_ws_pipelined_persistent.py:1465
          tlx.barrier_wait(blk_idx_77, blk_idx_99)  # blackwell_fa_ws_pipelined_persistent.py:1466
          tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:1468
          tlx.async_dot(blk_idx_62, doT_59, blk_idx_63, use_acc=False, pred=True, mBarriers=[blk_idx_60, blk_idx_61], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1468
          tlx.barrier_wait(blk_idx_81, phase_101)  # blackwell_fa_ws_pipelined_persistent.py:1480
          tlx.barrier_wait(blk_idx_52, phase_98)  # blackwell_fa_ws_pipelined_persistent.py:1485
          tlx.async_dot(dsT_view_82, blk_idx_85, blk_idx_86, use_acc=False, pred=True, mBarriers=[blk_idx_83, blk_idx_84], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1487
          tlx.barrier_wait(blk_idx_64, phase_98)  # blackwell_fa_ws_pipelined_persistent.py:1496
          tlx.barrier_wait(blk_idx_65, phase_98)  # blackwell_fa_ws_pipelined_persistent.py:1497
          tlx.async_dot(blk_idx_68, blk_idx_69, blk_idx_70, use_acc=True, pred=True, mBarriers=[blk_idx_67], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1498
          blk_idx_105 = arg171 + 1  # blackwell_fa_ws_pipelined_persistent.py:1506
          arg171 = blk_idx_105
        blk_idx_89 = tlx.cluster_cta_rank()  # blackwell_fa_ws_pipelined_persistent.py:1509
        blk_idx_90 = blk_idx_89 % 2  # blackwell_fa_ws_pipelined_persistent.py:1509
        blk_idx_91 = blk_idx_90 == 0  # blackwell_fa_ws_pipelined_persistent.py:1509
        tlx.tcgen05_commit(blk_idx_88, two_ctas=True)  # blackwell_fa_ws_pipelined_persistent.py:1509
        prev_blk_idx = arg171 - 1  # blackwell_fa_ws_pipelined_persistent.py:1516
        phase = prev_blk_idx & 1  # warp_spec.py:8
        tlx.barrier_wait(blk_idx_74, phase)  # blackwell_fa_ws_pipelined_persistent.py:1523
        tlx.barrier_wait(blk_idx_75, phase)  # blackwell_fa_ws_pipelined_persistent.py:1524
        blk_idx_92 = arg152 > 1  # blackwell_fa_ws_pipelined_persistent.py:1529
        tlx.async_dot(blk_idx_78, blk_idx_79, blk_idx_80, use_acc=blk_idx_92, pred=True, mBarriers=[blk_idx_76, blk_idx_93, blk_idx_77], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1525
        tlx.barrier_wait(blk_idx_81, phase)  # blackwell_fa_ws_pipelined_persistent.py:1535
        blk_idx_94 = phase ^ 1  # blackwell_fa_ws_pipelined_persistent.py:1536
        tlx.barrier_wait(blk_idx_73, blk_idx_94)  # blackwell_fa_ws_pipelined_persistent.py:1536
        tlx.barrier_wait(blk_idx_72, 0)  # blackwell_fa_ws_pipelined_persistent.py:1538
        tlx.async_dot(dsT_view_82, blk_idx_85, blk_idx_86, use_acc=False, pred=True, mBarriers=[blk_idx_83, blk_idx_95, blk_idx_96, blk_idx_97], two_ctas=True, force_async=True)  # blackwell_fa_ws_pipelined_persistent.py:1539
    with tlx.async_task(num_warps=1, registers=88):
      start_block_n_52 = arg167 * 128  # blackwell_fa_ws_pipelined_persistent.py:1775
      tlx.barrier_wait(blk_idx, 1)  # blackwell_fa_ws_pipelined_persistent.py:1778
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 65536)  # blackwell_fa_ws_pipelined_persistent.py:1780
      blk_idx_55 = tlx.cluster_cta_rank()  # blackwell_fa_ws_pipelined_persistent.py:1781
      blk_idx_56 = blk_idx_55 & -2  # blackwell_fa_ws_pipelined_persistent.py:1781
      blk_idx_57 = tlx.remote_view(blk_idx_54, blk_idx_56)  # blackwell_fa_ws_pipelined_persistent.py:1781
      tlx.async_descriptor_load(arg110, blk_idx_53, [arg102, arg141, start_block_n_52, 0], blk_idx_57)  # blackwell_fa_ws_pipelined_persistent.py:1781
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 65536)  # blackwell_fa_ws_pipelined_persistent.py:1791
      blk_idx_60 = tlx.remote_view(blk_idx_59, blk_idx_56)  # blackwell_fa_ws_pipelined_persistent.py:1792
      tlx.async_descriptor_load(arg114, blk_idx_58, [arg102, arg141, start_block_n_52, 0], blk_idx_60)  # blackwell_fa_ws_pipelined_persistent.py:1792
      tlx.barrier_wait(blk_idx_61, 1)  # blackwell_fa_ws_pipelined_persistent.py:1806
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 32768)  # blackwell_fa_ws_pipelined_persistent.py:1808
      blk_idx_63 = arg103 * 64  # blackwell_fa_ws_pipelined_persistent.py:1812
      blk_idx_65 = tlx.remote_view(blk_idx_64, blk_idx_56)  # blackwell_fa_ws_pipelined_persistent.py:1809
      tlx.async_descriptor_load(arg113, blk_idx_62, [arg102, arg141, blk_idx_63, 0], blk_idx_65)  # blackwell_fa_ws_pipelined_persistent.py:1809
      tlx.barrier_wait(blk_idx_66, 1)  # blackwell_fa_ws_pipelined_persistent.py:1819
      tlx.barrier_expect_bytes(blk_idx_67, 512)  # blackwell_fa_ws_pipelined_persistent.py:1820
      blk_idx_68 = arg101 + arg153  # blackwell_fa_ws_pipelined_persistent.py:1821
      blk_idx_70 = tlx.async_load(blk_idx_68, blk_idx_69, 512, blk_idx_67)  # blackwell_fa_ws_pipelined_persistent.py:1821
      tlx.barrier_wait(blk_idx_71, 1)  # blackwell_fa_ws_pipelined_persistent.py:1825
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 32768)  # blackwell_fa_ws_pipelined_persistent.py:1827
      blk_idx_74 = tlx.remote_view(blk_idx_73, blk_idx_56)  # blackwell_fa_ws_pipelined_persistent.py:1828
      tlx.async_descriptor_load(arg107, blk_idx_72, [arg102, arg141, 0, blk_idx_63], blk_idx_74)  # blackwell_fa_ws_pipelined_persistent.py:1828
      tlx.barrier_wait(blk_idx_75, 1)  # blackwell_fa_ws_pipelined_persistent.py:1836
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 32768)  # blackwell_fa_ws_pipelined_persistent.py:1838
      blk_idx_78 = tlx.remote_view(blk_idx_77, blk_idx_56)  # blackwell_fa_ws_pipelined_persistent.py:1839
      tlx.async_descriptor_load(arg108, blk_idx_76, [arg102, arg141, blk_idx_63, 0], blk_idx_78)  # blackwell_fa_ws_pipelined_persistent.py:1839
      tlx.barrier_wait(blk_idx_79, 1)  # blackwell_fa_ws_pipelined_persistent.py:1849
      tlx.barrier_expect_bytes(blk_idx_80, 512)  # blackwell_fa_ws_pipelined_persistent.py:1850
      blk_idx_81 = arg106 + arg153  # blackwell_fa_ws_pipelined_persistent.py:1851
      blk_idx_83 = tlx.async_load(blk_idx_81, blk_idx_82, 512, blk_idx_80)  # blackwell_fa_ws_pipelined_persistent.py:1851
      tlx.barrier_wait(blk_idx_84, 1)  # blackwell_fa_ws_pipelined_persistent.py:1854
      lower_start_block_n = arg103 * 128  # blackwell_fa_ws_pipelined_persistent.py:1855
      lower_start_block_n_85 = start_block_n_52 - lower_start_block_n  # blackwell_fa_ws_pipelined_persistent.py:1855
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 65536)  # blackwell_fa_ws_pipelined_persistent.py:1857
      blk_idx_88 = tlx.remote_view(blk_idx_87, blk_idx_56)  # blackwell_fa_ws_pipelined_persistent.py:1858
      tlx.async_descriptor_load(arg111, blk_idx_86, [arg102, arg141, lower_start_block_n_85, blk_idx_63], blk_idx_88)  # blackwell_fa_ws_pipelined_persistent.py:1858
      blk_idx_92 = tlx.remote_view(blk_idx_91, blk_idx_56)  # blackwell_fa_ws_pipelined_persistent.py:1899
      arg171 = 1
      arg172 = 128
      for arg170 in range(1, arg152, 1):
        phase_96 = arg171 & 1  # warp_spec.py:8
        blk_idx_97 = phase_96 ^ 1  # blackwell_fa_ws_pipelined_persistent.py:1873
        tlx.barrier_wait(blk_idx_61, blk_idx_97)  # blackwell_fa_ws_pipelined_persistent.py:1873
        if arg142:
          tlx.barrier_expect_bytes(blk_idx_64, 32768)  # blackwell_fa_ws_pipelined_persistent.py:1875
        blk_idx_98 = arg172 + blk_idx_63  # blackwell_fa_ws_pipelined_persistent.py:1879
        tlx.async_descriptor_load(arg113, blk_idx_62, [arg102, arg141, blk_idx_98, 0], blk_idx_65)  # blackwell_fa_ws_pipelined_persistent.py:1876
        tlx.barrier_wait(blk_idx_75, blk_idx_97)  # blackwell_fa_ws_pipelined_persistent.py:1884
        if arg142:
          tlx.barrier_expect_bytes(blk_idx_77, 32768)  # blackwell_fa_ws_pipelined_persistent.py:1886
        tlx.async_descriptor_load(arg108, blk_idx_76, [arg102, arg141, blk_idx_98, 0], blk_idx_78)  # blackwell_fa_ws_pipelined_persistent.py:1887
        blk_idx_99 = arg171 - 1  # blackwell_fa_ws_pipelined_persistent.py:1895
        phase_100 = blk_idx_99 & 1  # warp_spec.py:8
        blk_idx_101 = phase_100 ^ 1  # blackwell_fa_ws_pipelined_persistent.py:1896
        tlx.barrier_wait(blk_idx_89, blk_idx_101)  # blackwell_fa_ws_pipelined_persistent.py:1896
        if arg142:
          tlx.barrier_expect_bytes(blk_idx_91, 32768)  # blackwell_fa_ws_pipelined_persistent.py:1898
        blk_idx_102 = arg172 - 128  # blackwell_fa_ws_pipelined_persistent.py:1902
        tlx.async_descriptor_load(arg112, blk_idx_90, [arg102, arg141, blk_idx_102, blk_idx_63], blk_idx_92)  # blackwell_fa_ws_pipelined_persistent.py:1899
        tlx.barrier_wait(blk_idx_66, blk_idx_97)  # blackwell_fa_ws_pipelined_persistent.py:1909
        tlx.barrier_expect_bytes(blk_idx_67, 512)  # blackwell_fa_ws_pipelined_persistent.py:1910
        blk_idx_103 = blk_idx_68 + arg172  # blackwell_fa_ws_pipelined_persistent.py:1911
        blk_idx_104 = tlx.async_load(blk_idx_103, blk_idx_69, 512, blk_idx_67)  # blackwell_fa_ws_pipelined_persistent.py:1911
        tlx.barrier_wait(blk_idx_71, blk_idx_97)  # blackwell_fa_ws_pipelined_persistent.py:1914
        if arg142:
          tlx.barrier_expect_bytes(blk_idx_73, 32768)  # blackwell_fa_ws_pipelined_persistent.py:1916
        tlx.async_descriptor_load(arg107, blk_idx_72, [arg102, arg141, arg172, blk_idx_63], blk_idx_74)  # blackwell_fa_ws_pipelined_persistent.py:1917
        buf_idx = arg171 % 2  # warp_spec.py:7
        phase_105 = arg171 // 2  # warp_spec.py:8
        phase_106 = phase_105 & 1  # warp_spec.py:8
        blk_idx_108 = phase_106 ^ 1  # blackwell_fa_ws_pipelined_persistent.py:1927
        tlx.barrier_wait(blk_idx_107, blk_idx_108)  # blackwell_fa_ws_pipelined_persistent.py:1927
        tlx.barrier_expect_bytes(blk_idx_109, 512)  # blackwell_fa_ws_pipelined_persistent.py:1928
        blk_idx_110 = blk_idx_81 + arg172  # blackwell_fa_ws_pipelined_persistent.py:1929
        blk_idx_112 = tlx.async_load(blk_idx_110, blk_idx_111, 512, blk_idx_109)  # blackwell_fa_ws_pipelined_persistent.py:1929
        curr_m = arg172 + 128  # blackwell_fa_ws_pipelined_persistent.py:1931
        blk_idx_113 = arg171 + 1  # blackwell_fa_ws_pipelined_persistent.py:1932
        arg171 = blk_idx_113
        arg172 = curr_m
      blk_idx_93 = arg171 - 1  # blackwell_fa_ws_pipelined_persistent.py:1935
      phase = blk_idx_93 & 1  # warp_spec.py:8
      blk_idx_94 = phase ^ 1  # blackwell_fa_ws_pipelined_persistent.py:1936
      tlx.barrier_wait(blk_idx_89, blk_idx_94)  # blackwell_fa_ws_pipelined_persistent.py:1936
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_91, 32768)  # blackwell_fa_ws_pipelined_persistent.py:1938
      blk_idx_95 = arg172 - 128  # blackwell_fa_ws_pipelined_persistent.py:1942
      tlx.async_descriptor_load(arg112, blk_idx_90, [arg102, arg141, blk_idx_95, blk_idx_63], blk_idx_92)  # blackwell_fa_ws_pipelined_persistent.py:1939
    with tlx.async_task(num_warps=1, registers=40):
      var_2 = tlx.remote_view(var_1, 0)  # blackwell_fa_ws_pipelined_persistent.py:2954
      for arg170 in range(0, arg152, 1):
        phase = arg170 & 1  # warp_spec.py:8
        tlx.barrier_wait(var_0, phase)  # blackwell_fa_ws_pipelined_persistent.py:2952
        tlx.fence("async_shared")  # blackwell_fa_ws_pipelined_persistent.py:2953
        tlx.barrier_arrive(var_2)  # blackwell_fa_ws_pipelined_persistent.py:2954

