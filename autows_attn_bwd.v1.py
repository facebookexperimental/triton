import triton
import triton.language as tl
try:
    import triton.language.extra.cuda.tlx as tlx
except ModuleNotFoundError:
    import triton.language.extra.tlx as tlx

@triton.jit
def _attn_bwd(desc_q, desc_qt, desc_k, desc_kt, desc_v, sm_scale, desc_do, desc_dot, desc_dq, desc_dk, desc_dv, desc_m, desc_delta, stride_z, stride_h, stride_tok, BATCH, H, N_CTX):
  dq = tlx.alloc_barriers(1)
  dsT_dq_1 = tlx.alloc_barriers(1)
  dsT = tlx.alloc_barriers(1)
  Di = tlx.alloc_barriers(1)
  dpT = tlx.alloc_barriers(1)
  dpT_5 = tlx.alloc_barriers(1)
  ppT = tlx.alloc_barriers(1)
  do = tlx.alloc_barriers(1)
  do_9 = tlx.alloc_barriers(1)
  m = tlx.alloc_barriers(1)
  q = tlx.alloc_barriers(1)
  q_13 = tlx.alloc_barriers(1)
  qT = tlx.alloc_barriers(1)
  qT_16 = tlx.alloc_barriers(1)
  dk = tlx.alloc_barriers(1)
  dv = tlx.alloc_barriers(1)
  dpT_20 = tlx.alloc_barriers(1)
  qkT = tlx.alloc_barriers(1)
  v = tlx.alloc_barriers(1)
  v_24 = tlx.alloc_barriers(1)
  kt = tlx.alloc_barriers(1)
  kt_27 = tlx.alloc_barriers(1)
  k = tlx.alloc_barriers(1)
  k_30 = tlx.alloc_barriers(1)
  qkT_32 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  dpT_34 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  dv_36 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  dk_38 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  m_40 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  ppT_42 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  Di_44 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  dsT_46 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  dsT_dq_0 = tlx.alloc_barriers(1)
  dsT_dq_0_48 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  dsT_dq_1_51 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  dq_53 = tlx.alloc_barriers(1)
  tl.debug_barrier()
  k_55 = tlx.local_alloc((128, 128), tl.float16, 1)
  desc_dv_staging = tlx.local_alloc((128, 64), tl.float16, 1, tlx.storage_kind.tmem, reuse=k_55)
  kt_56 = tlx.local_alloc((256, 64), tl.float16, 1)
  v_57 = tlx.local_alloc((128, 128), tl.float16, 1)
  qkT_58 = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)
  dpT_59 = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)
  dv_60 = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)
  dk_61 = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)
  qT_62 = tlx.local_alloc((64, 128), tl.float16, 1)
  desc_dk_staging = tlx.local_alloc((128, 64), tl.float16, 1, tlx.storage_kind.tmem, reuse=qT_62)
  q_63 = tlx.local_alloc((128, 64), tl.float16, 1)
  m_64 = tlx.local_alloc((1, 128), tl.float32, 1)
  do_65 = tlx.local_alloc((128, 64), tl.float16, 1)
  dpT_66 = tlx.local_alloc((64, 128), tl.float16, 1)
  Di_67 = tlx.local_alloc((1, 128), tl.float32, 1)
  dsT_dq_0_68 = tlx.local_alloc((128, 64), tl.float16, 1)
  dsT_dq_1_69 = tlx.local_alloc((256, 64), tl.float16, 1)
  desc_dq_reduce_staging = tlx.local_alloc((128, 16), tl.float32, 1)
  desc_dk_staging_74 = tlx.alloc_barriers(5)
  desc_dk_staging_80 = tlx.alloc_barriers(1)
  desc_dk_staging_82 = tlx.alloc_barriers(1)
  desc_dk_staging_84 = tlx.alloc_barriers(1)
  desc_dk_staging_86 = tlx.alloc_barriers(1)
  desc_dk_staging_88 = tlx.alloc_barriers(1)
  with tlx.async_tasks():
    with tlx.async_task("default"):
      bhid = tl.program_id(axis=2)
      pid = tl.program_id(axis=0)
      off_bh = bhid % H
      off_bh_90 = stride_h * off_bh
      off_bh_91 = bhid // H
      off_bh_92 = stride_z * off_bh_91
      off_bh_93 = off_bh_90 + off_bh_92
      off_bh_96 = off_bh_93 // stride_tok
      start_n = pid * 128
      k_98 = off_bh_96 + start_n
      num_steps = N_CTX // 128
      arg64 = 0
      for arg63 in range(0, num_steps, 1):
        m_114 = arg64 & 1
        tlx.barrier_wait(m_117, m_114)
        pT = tlx.local_load(m_116)
        tlx.barrier_arrive(m_119)
        pT_120 = tl.expand_dims(pT, axis=0)
        tlx.barrier_wait(qkT_123, m_114)
        qkT_124 = tlx.local_load(qkT_122)
        tlx.barrier_arrive(qkT_125)
        pT_126 = tl.inline_asm_elementwise(qkT_124, pT_120)
        pT_127 = tl.math.exp2(pT_126)
        qkT_129 = tlx.subslice(qkT_58)
        qkT_130 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=qkT_129)
        dv_133 = m_114 ^ True
        tlx.barrier_wait(ppT_132, dv_133)
        tlx.local_store(ppT_131, pT_127.to(tl.float16))
        tlx.barrier_arrive(ppT_135)
        tlx.barrier_wait(Di_137, m_114)
        dsT_138 = tlx.local_load(Di_136)
        tlx.barrier_arrive(Di_139)
        dsT_140 = tl.expand_dims(dsT_138, axis=0)
        tlx.barrier_wait(dpT_143, m_114)
        dpT_144 = tlx.local_load(dpT_142)
        tlx.barrier_arrive(dpT_145)
        dsT_146 = tl.inline_asm_elementwise(dpT_144, dsT_140)
        dsT_147 = tl.inline_asm_elementwise(pT_127, dsT_146)
        dpT_149 = tlx.subslice(dpT_59)
        dpT_150 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=dpT_149)
        tlx.barrier_wait(dpT_145, m_114)
        tlx.local_store(dsT_151, dsT_147.to(tl.float16))
        tlx.barrier_arrive(dsT_153)
        dsT_dq = m_114 ^ True
        tlx.barrier_wait(dsT_dq_0_155, dsT_dq)
        tlx.barrier_wait(dsT_dq_1_159, dv_133)
        dsT_dq_160 = tlx.subslice(dsT_151)
        dsT_dq_161 = tlx.local_load(dsT_dq_160)
        dsT_dq_162 = tlx.subslice(dsT_151)
        dsT_dq_163 = tlx.local_load(dsT_dq_162)
        dsT_dq_164 = tlx.local_slice(dsT_dq_1_158, [0, 0], [128, 64])
        dsT_dq_165 = tlx.local_slice(dsT_dq_1_158, [128, 0], [128, 64])
        tlx.barrier_expect_bytes(dsT_dq_0_157, 16384)
        dsT_dq_166 = tlx.cluster_cta_rank()
        dsT_dq_167 = dsT_dq_166 == 0
        if dsT_dq_167:
          tlx.local_store(dsT_dq_164, dsT_dq_161)
          tlx.local_store(dsT_dq_0_154, dsT_dq_163)
          tlx.named_barrier_wait(7, 256)
          tlx.fence("async_shared")
          tlx.async_remote_shmem_copy(dsT_dq_0_154, dsT_dq_164, 1, dsT_dq_0_157)
        else:
          tlx.local_store(dsT_dq_165, dsT_dq_163)
          tlx.local_store(dsT_dq_0_154, dsT_dq_161)
          tlx.named_barrier_wait(7, 256)
          tlx.fence("async_shared")
          tlx.async_remote_shmem_copy(dsT_dq_0_154, dsT_dq_165, 0, dsT_dq_0_157)
        tlx.barrier_arrive(dsT_dq_1_168)
        accum_cnt = arg64 + 1
        arg64 = accum_cnt
      tlx.barrier_wait(dv_100, 0)
      dv_101 = tlx.subslice(dv_70)
      dv_102 = tlx.local_load(dv_101)
      dv_103 = tlx.subslice(dv_70)
      dv_104 = tlx.local_load(dv_103)
      tlx.barrier_arrive(dv_105)
      tlx.barrier_wait(dk_106, 0)
      dk_107 = tlx.subslice(dk_71)
      dk_108 = tlx.local_load(dk_107)
      dk_109 = tlx.subslice(dk_71)
      dk_110 = tlx.local_load(dk_109)
      tlx.barrier_arrive(dk_111)
      tlx.local_store(desc_dv_staging_72, dv_102.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dv, desc_dv_staging_72, [k_98, 0])
      tlx.async_descriptor_store_wait(0)
      tlx.local_store(desc_dv_staging_72, dv_104.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dv, desc_dv_staging_72, [k_98, 64])
      tlx.async_descriptor_store_wait(0)
      dkN_112 = dk_108 * sm_scale
      tlx.local_store(desc_dk_staging_73, dkN_112.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dk, desc_dk_staging_73, [k_98, 0])
      tlx.async_descriptor_store_wait(0)
      dkN_113 = dk_110 * sm_scale
      tlx.local_store(desc_dk_staging_73, dkN_113.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dk, desc_dk_staging_73, [k_98, 64])
      tlx.async_descriptor_store_wait(0)
    with tlx.async_task(num_warps=4, registers=88):
      cst = tl.full([128, 16], 0.693147182, tl.float32)
      bhid = tl.program_id(axis=2)
      pid = tl.program_id(axis=0)
      off_bh = bhid % H_90
      off_bh_169 = stride_h_91 * off_bh
      off_bh_170 = bhid // H_90
      off_bh_171 = stride_z_92 * off_bh_170
      off_bh_172 = off_bh_169 + off_bh_171
      off_bh_175 = off_bh_172 // stride_tok_93
      cluster_cta_rank = pid % 2
      num_steps = N_CTX_94 // 128
      dq_row = cluster_cta_rank * 64
      arg136 = 0
      arg137 = 0
      for arg135 in range(0, num_steps, 1):
        qt_177 = off_bh_175 + arg136
        dq_178 = arg137 & 1
        qkT_180 = tlx.subslice(qkT_95)
        qkT_181 = tlx.local_alloc((64, 128), tl.float32, 1, tlx.storage_kind.tmem, reuse=qkT_180)
        tlx.barrier_wait(dq_183, dq_178)
        dq_185 = tlx.local_load(dq_182)
        tlx.barrier_arrive(dq_186)
        dqs = tl.reshape(dq_185, [128, 2, 32])
        dqs_187 = tl.trans(dqs)
        (dqs_188, dqs_189) = tl.split(dqs_187)
        dqs_190 = tl.reshape(dqs_188, [128, 2, 16])
        dqs_191 = tl.trans(dqs_190)
        (dqs_192, dqs_193) = tl.split(dqs_191)
        dqs_194 = tl.reshape(dqs_189, [128, 2, 16])
        dqs_195 = tl.trans(dqs_194)
        (dqs_196, dqs_197) = tl.split(dqs_195)
        dq_row_198 = qt_177 + dq_row
        dq_row_199 = dq_row_198 * 2
        dqN = dqs_192 * cst
        tlx.local_store(desc_dq_reduce_staging_200, dqN)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_dq_98, desc_dq_reduce_staging_200, [dq_row_199, 0], store_reduce="add")
        tlx.async_descriptor_store_wait(0)
        dqN_201 = dqs_193 * cst
        tlx.local_store(desc_dq_reduce_staging_200, dqN_201)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_dq_98, desc_dq_reduce_staging_200, [dq_row_199, 16], store_reduce="add")
        tlx.async_descriptor_store_wait(0)
        dqN_202 = dqs_196 * cst
        tlx.local_store(desc_dq_reduce_staging_200, dqN_202)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_dq_98, desc_dq_reduce_staging_200, [dq_row_199, 32], store_reduce="add")
        tlx.async_descriptor_store_wait(0)
        dqN_203 = dqs_197 * cst
        tlx.local_store(desc_dq_reduce_staging_200, dqN_203)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_dq_98, desc_dq_reduce_staging_200, [dq_row_199, 48], store_reduce="add")
        tlx.async_descriptor_store_wait(0)
        curr_m_204 = arg136 + 128
        accum_cnt = arg137 + 1
        arg136 = curr_m_204
        arg137 = accum_cnt
    with tlx.async_task(num_warps=1, registers=88):
      num_steps = N_CTX_94 // 128
      tlx.barrier_wait(k_171, 0)
      tlx.barrier_wait(kt_172, 0)
      tlx.barrier_wait(v_173, 0)
      tlx.barrier_wait(dv_174, 1)
      tlx.barrier_wait(dk_175, 1)
      curr_m = num_steps > 0
      qT_177 = tlx.local_trans(qT_176)
      tlx.barrier_wait(qT_178, 0)
      tlx.barrier_wait(qkT_183, 1)
      qkT_185 = tlx.cluster_cta_rank()
      qkT_186 = qkT_185 & -2
      qkT_187 = tlx.remote_view(desc_dk_staging_184, qkT_186)
      tlx.barrier_arrive(qkT_187)
      qkT_188 = qkT_185 % 2
      qkT_189 = qkT_188 == 0
      tlx.barrier_wait(desc_dk_staging_184, 0)
      tlx.fence("async_shared")
      tlx.async_dot(k_179, qT_177, qkT_180, use_acc=False, pred=curr_m, mBarriers=[qT_181, qkT_182], two_ctas=True, force_async=True)
      dpT_191 = tlx.local_trans(dpT_190)
      tlx.barrier_wait(dpT_192, 0)
      tlx.barrier_wait(dpT_197, 1)
      tlx.barrier_wait(dsT_198, 1)
      dpT_200 = tlx.remote_view(desc_dk_staging_199, qkT_186)
      tlx.barrier_arrive(dpT_200)
      tlx.barrier_wait(desc_dk_staging_199, 0)
      tlx.async_dot(v_193, dpT_191, dpT_194, use_acc=False, pred=curr_m, mBarriers=[dpT_195, dpT_196], two_ctas=True, force_async=True)
      qkT_203 = tlx.subslice(qkT_95)
      qkT_204 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=qkT_203)
      tlx.barrier_wait(do_207, 0)
      tlx.barrier_wait(ppT_209, 0)
      dv_211 = tlx.remote_view(desc_dk_staging_210, qkT_186)
      tlx.barrier_arrive(dv_211)
      tlx.barrier_wait(desc_dk_staging_210, 0)
      tlx.async_dot(ppT_205, do_202, dv_201, use_acc=False, pred=curr_m, mBarriers=[do_206, ppT_208], two_ctas=True, force_async=True)
      curr_m_212 = num_steps - 1
      arg136 = False
      arg137 = 0
      arg138 = False
      for arg135 in range(0, curr_m_212, 1):
        accum_cnt = arg137 + 1
        qT_219 = accum_cnt & 1
        tlx.barrier_wait(qT_178, qT_219)
        q_222 = arg137 & 1
        qkT_224 = qT_219 ^ True
        tlx.barrier_wait(qkT_183, qkT_224)
        qkT_227 = tlx.remote_view(desc_dk_staging_226, qkT_186)
        tlx.barrier_arrive(qkT_227)
        qkT_229 = arg135 % 2
        tlx.barrier_wait(desc_dk_staging_226, qkT_229)
        tlx.fence("async_shared")
        tlx.async_dot(k_179, qT_177, qkT_180, use_acc=False, pred=True, mBarriers=[qT_181, qkT_182], two_ctas=True, force_async=True)
        dpT_233 = tlx.subslice(dpT_110)
        dpT_234 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=dpT_233)
        tlx.barrier_wait(q_237, q_222)
        tlx.barrier_wait(dsT_239, arg138)
        dk_242 = tlx.remote_view(desc_dk_staging_241, qkT_186)
        tlx.barrier_arrive(dk_242)
        tlx.barrier_wait(desc_dk_staging_241, qkT_229)
        tlx.async_dot(dsT_235, q_232, dk_231, use_acc=arg136, pred=True, mBarriers=[q_236, dsT_198], two_ctas=True, force_async=True)
        tlx.barrier_wait(dpT_192, qT_219)
        tlx.barrier_wait(dpT_197, qkT_224)
        dpT_243 = qT_219 ^ True
        tlx.barrier_wait(dsT_198, dpT_243)
        dpT_246 = tlx.remote_view(desc_dk_staging_245, qkT_186)
        tlx.barrier_arrive(dpT_246)
        tlx.barrier_wait(desc_dk_staging_245, qkT_229)
        tlx.async_dot(v_193, dpT_191, dpT_194, use_acc=False, pred=True, mBarriers=[dpT_195, dpT_196], two_ctas=True, force_async=True)
        dsT_dq = tlx.local_trans(dsT_dq_1_247)
        tlx.barrier_wait(dsT_dq_1_248, q_222)
        qkT_251 = tlx.subslice(qkT_95)
        qkT_252 = tlx.local_alloc((64, 128), tl.float32, 1, tlx.storage_kind.tmem, reuse=qkT_251)
        dq_257 = q_222 ^ True
        tlx.barrier_wait(dq_256, dq_257)
        dq_260 = tlx.remote_view(desc_dk_staging_259, qkT_186)
        tlx.barrier_arrive(dq_260)
        tlx.barrier_wait(desc_dk_staging_259, qkT_229)
        tlx.fence("async_shared")
        tlx.async_dot(dsT_dq, kt_250, dq_253, use_acc=False, pred=True, mBarriers=[dsT_dq_1_254, dq_255], two_ctas=True, force_async=True)
        tlx.barrier_wait(do_207, qT_219)
        tlx.barrier_wait(ppT_209, qT_219)
        dv_263 = tlx.remote_view(desc_dk_staging_262, qkT_186)
        tlx.barrier_arrive(dv_263)
        tlx.barrier_wait(desc_dk_staging_262, qkT_229)
        tlx.async_dot(ppT_205, do_202, dv_201, use_acc=True, pred=True, mBarriers=[do_206, ppT_208], two_ctas=True, force_async=True)
        arg136 = True
        arg137 = accum_cnt
        arg138 = qT_219
      if curr_m:
        q_219 = arg137 & 1
        dpT_223 = tlx.subslice(dpT_110)
        dpT_224 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=dpT_223)
        tlx.barrier_wait(q_227, q_219)
        tlx.barrier_wait(dsT_229, arg138)
        dk_232 = tlx.remote_view(desc_dk_staging_231, qkT_186)
        tlx.barrier_arrive(dk_232)
        tlx.barrier_wait(desc_dk_staging_231, 0)
        tlx.async_dot(dsT_225, q_222, dk_221, use_acc=arg136, pred=True, mBarriers=[q_226, dsT_198], two_ctas=True, force_async=True)
        dsT_dq = tlx.local_trans(dsT_dq_1_233)
        tlx.barrier_wait(dsT_dq_1_234, q_219)
        qkT_237 = tlx.subslice(qkT_95)
        qkT_238 = tlx.local_alloc((64, 128), tl.float32, 1, tlx.storage_kind.tmem, reuse=qkT_237)
        dq_243 = q_219 ^ True
        tlx.barrier_wait(dq_242, dq_243)
        dq_246 = tlx.remote_view(desc_dk_staging_245, qkT_186)
        tlx.barrier_arrive(dq_246)
        tlx.barrier_wait(desc_dk_staging_245, 0)
        tlx.fence("async_shared")
        tlx.async_dot(dsT_dq, kt_236, dq_239, use_acc=False, pred=True, mBarriers=[dsT_dq_1_240, dq_241], two_ctas=True, force_async=True)
      tlx.tcgen05_commit(dk_214)
      tlx.tcgen05_commit(dv_215)
      tlx.tcgen05_commit(v_216)
      tlx.tcgen05_commit(kt_217)
      tlx.tcgen05_commit(k_218)
    with tlx.async_task(num_warps=1, registers=88):
      bhid = tl.program_id(axis=2)
      pid = tl.program_id(axis=0)
      off_chz = bhid * N_CTX_94
      off_bh = bhid % H_90
      off_bh_171 = stride_h_91 * off_bh
      off_bh_172 = bhid // H_90
      off_bh_173 = stride_z_92 * off_bh_172
      off_bh_174 = off_bh_171 + off_bh_173
      off_bh_177 = off_bh_174 // stride_tok_93
      start_n = pid * 128
      cluster_cta_rank = pid % 2
      k_179 = off_bh_177 + start_n
      tlx.barrier_wait(k_181, 1)
      tlx.barrier_expect_bytes(k_182, 32768)
      tlx.async_descriptor_load(desc_k_131, k_183, [k_179, 0], k_182)
      kt_start_n = cluster_cta_rank * 128
      kt_start_n_184 = start_n - kt_start_n
      kt_186 = off_bh_177 + kt_start_n_184
      kt_188 = tlx.cluster_cta_rank()
      kt_189 = kt_188 % 2
      kt_190 = kt_189 * 64
      tlx.barrier_wait(kt_191, 1)
      tlx.barrier_expect_bytes(kt_192, 32768)
      tlx.async_descriptor_load(desc_kt_132, kt_193, [kt_186, kt_190], kt_192)
      tlx.barrier_wait(v_194, 1)
      tlx.barrier_expect_bytes(v_195, 32768)
      tlx.async_descriptor_load(desc_v_133, v_196, [k_179, 0], v_195)
      num_steps = N_CTX_94 // 128
      arg136 = 0
      arg137 = 0
      for arg135 in range(0, num_steps, 1):
        qt_197 = off_bh_177 + arg136
        qt_199 = qt_197 + kt_190
        qT_200 = arg137 & 1
        qt_203 = qT_200 ^ True
        tlx.barrier_wait(qT_202, qt_203)
        tlx.barrier_expect_bytes(qT_205, 16384)
        tlx.async_descriptor_load(desc_qt_134, qT_206, [qt_199, 0], qT_205)
        tlx.barrier_wait(q_207, qt_203)
        tlx.barrier_expect_bytes(q_208, 16384)
        tlx.async_descriptor_load(desc_q_135, q_209, [qt_197, kt_190], q_208)
        offs_m_start = off_chz + arg136
        m_212 = qT_200 ^ True
        tlx.barrier_wait(m_211, m_212)
        tlx.barrier_expect_bytes(m_214, 512)
        tlx.async_descriptor_load(desc_m_138, m_215, [offs_m_start], m_214)
        tlx.barrier_wait(do_216, qt_203)
        tlx.barrier_expect_bytes(do_217, 16384)
        tlx.async_descriptor_load(desc_do_139, do_218, [qt_197, kt_190], do_217)
        tlx.barrier_wait(dpT_219, qt_203)
        tlx.barrier_expect_bytes(dpT_220, 16384)
        tlx.async_descriptor_load(desc_dot_140, dpT_221, [qt_199, 0], dpT_220)
        tlx.barrier_wait(Di_222, m_212)
        tlx.barrier_expect_bytes(Di_223, 512)
        tlx.async_descriptor_load(desc_delta_143, Di_224, [offs_m_start], Di_223)
        curr_m_225 = arg136 + 128
        accum_cnt = arg137 + 1
        arg136 = curr_m_225
        arg137 = accum_cnt
    with tlx.async_task(num_warps=1, registers=40):
      num_steps = N_CTX_94 // 128
      arg136 = 0
      for arg135 in range(0, num_steps, 1):
        dsT_dq = arg136 & 1
        tlx.barrier_wait(dsT_dq_0_168, dsT_dq)
        tlx.fence("async_shared")
        tlx.barrier_arrive(dsT_dq_170)
        tlx.barrier_arrive(dsT_dq_0_171)
        accum_cnt = arg136 + 1
        arg136 = accum_cnt

