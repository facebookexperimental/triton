import triton
import triton.language as tl
try:
    import triton.language.extra.cuda.tlx as tlx
except ModuleNotFoundError:
    import triton.language.extra.tlx as tlx

@triton.jit
def _attn_bwd_ws(desc_q, desc_k, desc_v, sm_scale, desc_do, desc_dq, desc_dk, desc_dv, desc_m, desc_delta, M_ptr, delta_ptr, H, N_CTX, desc_kt, desc_qt, desc_dot):
  start_n = tl.program_id(axis=0)
  head = tl.program_id(axis=1)
  batch = tl.program_id(axis=2)
  off_chz = batch * H
  off_chz_0 = off_chz + head
  off_chz_1 = off_chz_0 * N_CTX
  num_steps = N_CTX // 128
  start_block_n = start_n * 128
  k_mma_done = tlx.alloc_barriers(1)
  k_empties = tlx.alloc_barriers(1)
  q_fulls = tlx.alloc_barriers(1)
  q_empties = tlx.alloc_barriers(1)
  do_fulls = tlx.alloc_barriers(1)
  do_empties = tlx.alloc_barriers(1)
  m_fulls = tlx.alloc_barriers(1)
  m_empties = tlx.alloc_barriers(1)
  d_fulls = tlx.alloc_barriers(2)
  d_empties = tlx.alloc_barriers(2)
  ds_fulls = tlx.alloc_barriers(1)
  dsT_tmem_fulls = tlx.alloc_barriers(1)
  qk_fulls = tlx.alloc_barriers(1)
  qk_empties = tlx.alloc_barriers(1)
  p_fulls = tlx.alloc_barriers(1)
  dp_fulls = tlx.alloc_barriers(1)
  dq_fulls = tlx.alloc_barriers(1)
  dq_empties = tlx.alloc_barriers(1)
  dv_fulls = tlx.alloc_barriers(1)
  dv_empties = tlx.alloc_barriers(1)
  dk_fulls = tlx.alloc_barriers(1)
  dk_empties = tlx.alloc_barriers(1)
  dp_empties = tlx.alloc_barriers(1)
  k_fulls = tlx.alloc_barriers(1)
  v_fulls = tlx.alloc_barriers(1)
  kt_fulls = tlx.alloc_barriers(1)
  kt_empties = tlx.alloc_barriers(1)
  qt_fulls = tlx.alloc_barriers(1)
  qt_empties = tlx.alloc_barriers(1)
  dot_fulls = tlx.alloc_barriers(1)
  dot_empties = tlx.alloc_barriers(1)
  ds_peer_fulls = tlx.alloc_barriers(1)
  ds_empties = tlx.alloc_barriers(1)
  k_tiles = tlx.local_alloc((128, 128), tl.float16, 1)
  v_tiles = tlx.local_alloc((128, 128), tl.float16, 1)
  q_tiles = tlx.local_alloc((128, 64), tl.float16, 1)
  do_tiles = tlx.local_alloc((128, 64), tl.float16, 1)
  ds_tiles = tlx.local_alloc((256, 64), tl.float16, 1)
  dq_store_buf = tlx.local_alloc((128, 16), tl.float32, 2)
  v_tiles_38 = tlx.local_alloc((128, 64), tl.float16, 1, tlx.storage_kind.tmem, reuse=v_tiles)
  k_tiles_39 = tlx.local_alloc((128, 64), tl.float16, 1, tlx.storage_kind.tmem, reuse=k_tiles)
  sM_tiles = tlx.local_alloc((1, 128), tl.float32, 1)
  sD_tiles = tlx.local_alloc((2, 128), tl.float32, 1)
  qk_p_storage_alias = tlx.local_alloc((128,), tl.int32, 128, tlx.storage_kind.tmem)
  qk_p_storage_alias_40 = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem, reuse=qk_p_storage_alias)
  qk_p_storage_alias_41 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=qk_p_storage_alias)
  dp_dq_storage_alias = tlx.local_alloc((128,), tl.int32, 128, tlx.storage_kind.tmem)
  dp_dq_storage_alias_42 = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem, reuse=dp_dq_storage_alias)
  dp_dq_storage_alias_43 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=dp_dq_storage_alias)
  dv_tiles = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)
  dk_tiles = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)
  qk_p_storage_alias_44 = tlx.local_alloc((64, 128), tl.float32, 2, tlx.storage_kind.tmem, reuse=qk_p_storage_alias)
  qk_p_storage_alias_45 = tlx.local_alloc((128, 64), tl.float32, 2, tlx.storage_kind.tmem, reuse=qk_p_storage_alias)
  cluster_cta_rank = tlx.cluster_cta_rank()
  is_leader = cluster_cta_rank == 0
  kt_tiles = tlx.local_alloc((256, 64), tl.float16, 1)
  qt_tiles = tlx.local_alloc((64, 128), tl.float16, 1)
  dot_tiles = tlx.local_alloc((64, 128), tl.float16, 1)
  ds_xchg_tiles = tlx.local_alloc((128, 64), tl.float16, 1)
  with tlx.async_tasks():
    with tlx.async_task("default"):
      var_1 = tlx.remote_view(qk_empties_18, 0)
      var_2 = tlx.remote_view(p_fulls_19, 0)
      var_4 = tlx.remote_view(dsT_tmem_fulls_16, 0)
      peer_rank = 1 - cluster_cta_rank
      if is_leader:
        own_tmem = tlx.subslice(var_3)
        peer_tmem = tlx.subslice(var_3)
        own_smem_55 = tlx.local_slice(own_smem, [0, 0], [128, 64])
        var_5_0 = own_smem_55
        var_5_1 = own_tmem
        var_5_2 = peer_tmem
      else:
        own_tmem = tlx.subslice(var_3)
        peer_tmem = tlx.subslice(var_3)
        own_smem_55 = tlx.local_slice(own_smem, [128, 0], [128, 64])
        var_5_0 = own_smem_55
        var_5_1 = own_tmem
        var_5_2 = peer_tmem
      var_6 = tlx.remote_view(dp_empties_27, 0)
      arg102 = 0
      for arg101 in range(0, num_steps, 1):
        phase = arg102 & 1
        buf_idx = arg102 % 2
        phase_55 = arg102 // 2
        phase_56 = phase_55 & 1
        tlx.barrier_wait(qk_fulls_17, phase)
        tlx.barrier_wait(m_fulls_9, phase)
        qkT_57 = tlx.local_load(qkT)
        sT = tlx.local_load(m)
        sT_58 = tl.expand_dims(sT, axis=0)
        sT_60 = tl.inline_asm_elementwise(qkT_57, sT_58)
        pT = tl.math.exp2(sT_60)
        tlx.named_barrier_wait(10, 256)
        tlx.local_store(var_0, pT.to(tl.float16))
        tlx.barrier_arrive(var_1)
        tlx.barrier_arrive(var_2)
        tlx.barrier_wait(dp_fulls_20, phase)
        dpT_61 = tlx.local_load(dpT)
        tlx.barrier_wait(var_16, phase_56)
        dsT = tlx.local_load(Di)
        tlx.barrier_arrive(m_empties_10)
        tlx.barrier_arrive(var_17)
        dsT_62 = tl.expand_dims(dsT, axis=0)
        dsT_64 = tl.inline_asm_elementwise(dpT_61, dsT_62)
        dsT_65 = tl.inline_asm_elementwise(pT, dsT_64)
        tlx.named_barrier_wait(11, 256)
        tlx.local_store(var_3, dsT_65.to(tl.float16))
        tlx.barrier_arrive(var_4)
        var_18 = phase ^ 1
        tlx.barrier_wait(ds_empties_37, var_18)
        own_data = tlx.local_load(var_5_1)
        tlx.local_store(var_5_0, own_data)
        peer_data = tlx.local_load(var_5_2)
        tlx.barrier_arrive(var_6)
        tlx.local_store(var_7, peer_data)
        tlx.fence("async_shared")
        tlx.barrier_expect_bytes(ds_peer_fulls_36, 16384)
        tlx.async_remote_shmem_copy(var_7, var_5_0, peer_rank, ds_peer_fulls_36)
        blk_idx_67 = arg102 + 1
        arg102 = blk_idx_67
      tlx.barrier_wait(dv_fulls_23, 0)
      dv_slice_46 = tlx.subslice(dv_slice)
      dv = tlx.local_load(dv_slice_46)
      tlx.async_descriptor_store_wait(0)
      tlx.local_store(var_8, dv.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dv, var_8, [batch, head, start_block_n, 0])
      dv_slice_47 = tlx.subslice(dv_slice)
      dv_48 = tlx.local_load(dv_slice_47)
      tlx.async_descriptor_store_wait(0)
      tlx.local_store(var_8, dv_48.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dv, var_8, [batch, head, start_block_n, 64])
      var_11 = tlx.remote_view(dv_empties_24, 0)
      tlx.barrier_arrive(var_11)
      tlx.barrier_wait(dk_fulls_25, 0)
      tlx.barrier_wait(k_mma_done_3, 0)
      dk_slice_49 = tlx.subslice(dk_slice)
      dk = tlx.local_load(dk_slice_49)
      dk_51 = dk * sm_scale
      tlx.async_descriptor_store_wait(0)
      tlx.local_store(var_12, dk_51.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dk, var_12, [batch, head, start_block_n, 0])
      dk_slice_52 = tlx.subslice(dk_slice)
      dk_53 = tlx.local_load(dk_slice_52)
      dk_54 = dk_53 * sm_scale
      tlx.async_descriptor_store_wait(0)
      tlx.local_store(var_12, dk_54.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dk, var_12, [batch, head, start_block_n, 64])
      tlx.async_descriptor_store_wait(0)
      var_15 = tlx.remote_view(dk_empties_26, 0)
      tlx.barrier_arrive(var_15)
    with tlx.async_task(num_warps=4, registers=88):
      cst = tl.full([128, 64], 0.693147182, tl.float32)
      dq_m_offset = arg103 * 64
      var_2 = tlx.remote_view(var_1, 0)
      arg171 = 0
      arg172 = 0
      for arg170 in range(0, arg152, 1):
        phase = arg171 & 1
        tlx.barrier_wait(var_0, phase)
        packed_row_base = arg172 + dq_m_offset
        packed_row_base_53 = packed_row_base * 2
        dq_full_54 = tlx.local_load(dq_full)
        tlx.barrier_arrive(var_2)
        dq_full_55 = dq_full_54 * cst
        dq_slices = tl.reshape(dq_full_55, [128, 2, 32])
        dq_slices_56 = tl.trans(dq_slices)
        (dq_slices_57, dq_slices_58) = tl.split(dq_slices_56)
        dq_slices_59 = tl.reshape(dq_slices_57, [128, 2, 16])
        dq_slices_60 = tl.trans(dq_slices_59)
        (dq_slices_61, dq_slices_62) = tl.split(dq_slices_60)
        dq_slices_63 = tl.reshape(dq_slices_58, [128, 2, 16])
        dq_slices_64 = tl.trans(dq_slices_63)
        (dq_slices_65, dq_slices_66) = tl.split(dq_slices_64)
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(dq_smem, dq_slices_61)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(arg109, dq_smem, [arg102, arg141, packed_row_base_53, 0], store_reduce="add")
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(dq_smem_52, dq_slices_62)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(arg109, dq_smem_52, [arg102, arg141, packed_row_base_53, 16], store_reduce="add")
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(dq_smem, dq_slices_65)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(arg109, dq_smem, [arg102, arg141, packed_row_base_53, 32], store_reduce="add")
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(dq_smem_52, dq_slices_66)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(arg109, dq_smem_52, [arg102, arg141, packed_row_base_53, 48], store_reduce="add")
        curr_m_67 = arg172 + 128
        blk_idx = arg171 + 1
        arg171 = blk_idx
        arg172 = curr_m_67
      tlx.async_descriptor_store_wait(0)
    with tlx.async_task(num_warps=1, registers=88):
      if arg142:
        tlx.barrier_wait(blk_idx, 0)
        tlx.barrier_wait(blk_idx_50, 0)
        tlx.barrier_wait(blk_idx_51, 0)
        tlx.barrier_wait(blk_idx_52, 1)
        qT_53 = tlx.local_trans(qT)
        tlx.fence("async_shared")
        tlx.async_dot(blk_idx_56, qT_53, blk_idx_57, use_acc=False, pred=True, mBarriers=[blk_idx_54, blk_idx_55], two_ctas=True, force_async=True)
        tlx.barrier_wait(blk_idx_58, 0)
        doT_59 = tlx.local_trans(doT)
        tlx.fence("async_shared")
        tlx.async_dot(blk_idx_62, doT_59, blk_idx_63, use_acc=False, pred=True, mBarriers=[blk_idx_60, blk_idx_61], two_ctas=True, force_async=True)
        tlx.barrier_wait(blk_idx_64, 0)
        tlx.barrier_wait(blk_idx_65, 0)
        tlx.barrier_wait(blk_idx_66, 1)
        tlx.async_dot(blk_idx_68, blk_idx_69, blk_idx_70, use_acc=False, pred=True, mBarriers=[blk_idx_67], two_ctas=True, force_async=True)
        tlx.barrier_wait(blk_idx_71, 1)
        tlx.barrier_wait(blk_idx_72, 0)
        dsT_view_82 = tlx.local_trans(dsT_view)
        arg171 = 1
        for arg170 in range(1, arg152, 1):
          phase_98 = arg171 & 1
          tlx.barrier_wait(blk_idx_51, phase_98)
          blk_idx_99 = phase_98 ^ 1
          tlx.barrier_wait(blk_idx_52, blk_idx_99)
          prev_blk_idx_100 = arg171 - 1
          phase_101 = prev_blk_idx_100 & 1
          blk_idx_102 = phase_101 ^ 1
          tlx.barrier_wait(blk_idx_73, blk_idx_102)
          tlx.fence("async_shared")
          tlx.async_dot(blk_idx_56, qT_53, blk_idx_57, use_acc=False, pred=True, mBarriers=[blk_idx_54, blk_idx_55], two_ctas=True, force_async=True)
          tlx.barrier_wait(blk_idx_74, phase_101)
          tlx.barrier_wait(blk_idx_75, phase_101)
          blk_idx_103 = arg170 - 1
          blk_idx_104 = blk_idx_103 > 0
          tlx.async_dot(blk_idx_78, blk_idx_79, blk_idx_80, use_acc=blk_idx_104, pred=True, mBarriers=[blk_idx_76, blk_idx_77], two_ctas=True, force_async=True)
          tlx.barrier_wait(blk_idx_58, phase_98)
          tlx.barrier_wait(blk_idx_77, blk_idx_99)
          tlx.fence("async_shared")
          tlx.async_dot(blk_idx_62, doT_59, blk_idx_63, use_acc=False, pred=True, mBarriers=[blk_idx_60, blk_idx_61], two_ctas=True, force_async=True)
          tlx.barrier_wait(blk_idx_81, phase_101)
          tlx.barrier_wait(blk_idx_52, phase_98)
          tlx.async_dot(dsT_view_82, blk_idx_85, blk_idx_86, use_acc=False, pred=True, mBarriers=[blk_idx_83, blk_idx_84], two_ctas=True, force_async=True)
          tlx.barrier_wait(blk_idx_64, phase_98)
          tlx.barrier_wait(blk_idx_65, phase_98)
          tlx.async_dot(blk_idx_68, blk_idx_69, blk_idx_70, use_acc=True, pred=True, mBarriers=[blk_idx_67], two_ctas=True, force_async=True)
          blk_idx_105 = arg171 + 1
          arg171 = blk_idx_105
        blk_idx_89 = tlx.cluster_cta_rank()
        blk_idx_90 = blk_idx_89 % 2
        blk_idx_91 = blk_idx_90 == 0
        tlx.tcgen05_commit(blk_idx_88, two_ctas=True)
        prev_blk_idx = arg171 - 1
        phase = prev_blk_idx & 1
        tlx.barrier_wait(blk_idx_74, phase)
        tlx.barrier_wait(blk_idx_75, phase)
        blk_idx_92 = arg152 > 1
        tlx.async_dot(blk_idx_78, blk_idx_79, blk_idx_80, use_acc=blk_idx_92, pred=True, mBarriers=[blk_idx_76, blk_idx_93, blk_idx_77], two_ctas=True, force_async=True)
        tlx.barrier_wait(blk_idx_81, phase)
        blk_idx_94 = phase ^ 1
        tlx.barrier_wait(blk_idx_73, blk_idx_94)
        tlx.barrier_wait(blk_idx_72, 0)
        tlx.async_dot(dsT_view_82, blk_idx_85, blk_idx_86, use_acc=False, pred=True, mBarriers=[blk_idx_83, blk_idx_95, blk_idx_96, blk_idx_97], two_ctas=True, force_async=True)
    with tlx.async_task(num_warps=1, registers=88):
      start_block_n_52 = arg167 * 128
      tlx.barrier_wait(blk_idx, 1)
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 65536)
      blk_idx_55 = tlx.cluster_cta_rank()
      blk_idx_56 = blk_idx_55 & -2
      blk_idx_57 = tlx.remote_view(blk_idx_54, blk_idx_56)
      tlx.async_descriptor_load(arg110, blk_idx_53, [arg102, arg141, start_block_n_52, 0], blk_idx_57)
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 65536)
      blk_idx_60 = tlx.remote_view(blk_idx_59, blk_idx_56)
      tlx.async_descriptor_load(arg114, blk_idx_58, [arg102, arg141, start_block_n_52, 0], blk_idx_60)
      tlx.barrier_wait(blk_idx_61, 1)
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 32768)
      blk_idx_63 = arg103 * 64
      blk_idx_65 = tlx.remote_view(blk_idx_64, blk_idx_56)
      tlx.async_descriptor_load(arg113, blk_idx_62, [arg102, arg141, blk_idx_63, 0], blk_idx_65)
      tlx.barrier_wait(blk_idx_66, 1)
      tlx.barrier_expect_bytes(blk_idx_67, 512)
      blk_idx_68 = arg101 + arg153
      blk_idx_70 = tlx.async_load(blk_idx_68, blk_idx_69, 512, blk_idx_67)
      tlx.barrier_wait(blk_idx_71, 1)
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 32768)
      blk_idx_74 = tlx.remote_view(blk_idx_73, blk_idx_56)
      tlx.async_descriptor_load(arg107, blk_idx_72, [arg102, arg141, 0, blk_idx_63], blk_idx_74)
      tlx.barrier_wait(blk_idx_75, 1)
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 32768)
      blk_idx_78 = tlx.remote_view(blk_idx_77, blk_idx_56)
      tlx.async_descriptor_load(arg108, blk_idx_76, [arg102, arg141, blk_idx_63, 0], blk_idx_78)
      tlx.barrier_wait(blk_idx_79, 1)
      tlx.barrier_expect_bytes(blk_idx_80, 512)
      blk_idx_81 = arg106 + arg153
      blk_idx_83 = tlx.async_load(blk_idx_81, blk_idx_82, 512, blk_idx_80)
      tlx.barrier_wait(blk_idx_84, 1)
      lower_start_block_n = arg103 * 128
      lower_start_block_n_85 = start_block_n_52 - lower_start_block_n
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_96, 65536)
      blk_idx_88 = tlx.remote_view(blk_idx_87, blk_idx_56)
      tlx.async_descriptor_load(arg111, blk_idx_86, [arg102, arg141, lower_start_block_n_85, blk_idx_63], blk_idx_88)
      blk_idx_92 = tlx.remote_view(blk_idx_91, blk_idx_56)
      arg171 = 1
      arg172 = 128
      for arg170 in range(1, arg152, 1):
        phase_96 = arg171 & 1
        blk_idx_97 = phase_96 ^ 1
        tlx.barrier_wait(blk_idx_61, blk_idx_97)
        if arg142:
          tlx.barrier_expect_bytes(blk_idx_64, 32768)
        blk_idx_98 = arg172 + blk_idx_63
        tlx.async_descriptor_load(arg113, blk_idx_62, [arg102, arg141, blk_idx_98, 0], blk_idx_65)
        tlx.barrier_wait(blk_idx_75, blk_idx_97)
        if arg142:
          tlx.barrier_expect_bytes(blk_idx_77, 32768)
        tlx.async_descriptor_load(arg108, blk_idx_76, [arg102, arg141, blk_idx_98, 0], blk_idx_78)
        blk_idx_99 = arg171 - 1
        phase_100 = blk_idx_99 & 1
        blk_idx_101 = phase_100 ^ 1
        tlx.barrier_wait(blk_idx_89, blk_idx_101)
        if arg142:
          tlx.barrier_expect_bytes(blk_idx_91, 32768)
        blk_idx_102 = arg172 - 128
        tlx.async_descriptor_load(arg112, blk_idx_90, [arg102, arg141, blk_idx_102, blk_idx_63], blk_idx_92)
        tlx.barrier_wait(blk_idx_66, blk_idx_97)
        tlx.barrier_expect_bytes(blk_idx_67, 512)
        blk_idx_103 = blk_idx_68 + arg172
        blk_idx_104 = tlx.async_load(blk_idx_103, blk_idx_69, 512, blk_idx_67)
        tlx.barrier_wait(blk_idx_71, blk_idx_97)
        if arg142:
          tlx.barrier_expect_bytes(blk_idx_73, 32768)
        tlx.async_descriptor_load(arg107, blk_idx_72, [arg102, arg141, arg172, blk_idx_63], blk_idx_74)
        buf_idx = arg171 % 2
        phase_105 = arg171 // 2
        phase_106 = phase_105 & 1
        blk_idx_108 = phase_106 ^ 1
        tlx.barrier_wait(blk_idx_107, blk_idx_108)
        tlx.barrier_expect_bytes(blk_idx_109, 512)
        blk_idx_110 = blk_idx_81 + arg172
        blk_idx_112 = tlx.async_load(blk_idx_110, blk_idx_111, 512, blk_idx_109)
        curr_m = arg172 + 128
        blk_idx_113 = arg171 + 1
        arg171 = blk_idx_113
        arg172 = curr_m
      blk_idx_93 = arg171 - 1
      phase = blk_idx_93 & 1
      blk_idx_94 = phase ^ 1
      tlx.barrier_wait(blk_idx_89, blk_idx_94)
      if arg142:
        tlx.barrier_expect_bytes(blk_idx_91, 32768)
      blk_idx_95 = arg172 - 128
      tlx.async_descriptor_load(arg112, blk_idx_90, [arg102, arg141, blk_idx_95, blk_idx_63], blk_idx_92)
    with tlx.async_task(num_warps=1, registers=40):
      var_2 = tlx.remote_view(var_1, 0)
      for arg170 in range(0, arg152, 1):
        phase = arg170 & 1
        tlx.barrier_wait(var_0, phase)
        tlx.fence("async_shared")
        tlx.barrier_arrive(var_2)

