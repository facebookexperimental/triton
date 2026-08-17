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
  dpT_34 = tlx.alloc_barriers(1)
  dv_36 = tlx.alloc_barriers(1)
  dk_38 = tlx.alloc_barriers(1)
  m_40 = tlx.alloc_barriers(1)
  ppT_42 = tlx.alloc_barriers(1)
  Di_44 = tlx.alloc_barriers(1)
  dsT_46 = tlx.alloc_barriers(1)
  dsT_dq_0 = tlx.alloc_barriers(1)
  dsT_dq_0_48 = tlx.alloc_barriers(1)
  dsT_dq_1_51 = tlx.alloc_barriers(1)
  dq_53 = tlx.alloc_barriers(1)
  ttg.barrier()
  k_55 = tlx.local_alloc((128, 128), tl.float16, 1)
  desc_dv_staging = tlx.local_alloc((128, 64), tl.float16, 2, tlx.storage_kind.tmem, reuse=k_55)
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
  desc_dq_reduce_staging = tlx.local_alloc((128, 16), tl.float32, 2)
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
      tlx.barrier_wait(dk_105, 0)
      dk_106 = tlx.subslice(dk_71)
      dk_107 = tlx.subslice(dk_71)
      tlx.local_store(desc_dv_staging_72, dv_102.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dv, desc_dv_staging_72, [k_98, 0])
      dv_109 = tlx.local_load(dv_103)
      tlx.async_descriptor_store_wait(0)
      tlx.local_store(desc_dv_staging_72, dv_109.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dv, desc_dv_staging_72, [k_98, 64])
      dk_110 = tlx.local_load(dk_106)
      tlx.barrier_arrive(dk_108)
      dkN_111 = dk_110 * sm_scale
      tlx.async_descriptor_store_wait(0)
      tlx.local_store(desc_dk_staging_73, dkN_111.to(tl.float16))
      tlx.fence("async_shared")
      tlx.async_descriptor_store(desc_dk, desc_dk_staging_73, [k_98, 0])
      dk_112 = tlx.local_load(dk_107)
      tlx.barrier_arrive(dv_104)
      dkN_113 = dk_112 * sm_scale
      tlx.async_descriptor_store_wait(0)
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
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(desc_dq_reduce_staging_200, dqN)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_dq_98, desc_dq_reduce_staging_200, [dq_row_199, 0], store_reduce="add")
        dqN_201 = dqs_193 * cst
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(desc_dq_reduce_staging_202, dqN_201)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_dq_98, desc_dq_reduce_staging_202, [dq_row_199, 16], store_reduce="add")
        dqN_203 = dqs_196 * cst
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(desc_dq_reduce_staging_200, dqN_203)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_dq_98, desc_dq_reduce_staging_200, [dq_row_199, 32], store_reduce="add")
        dqN_204 = dqs_197 * cst
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(desc_dq_reduce_staging_202, dqN_204)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_dq_98, desc_dq_reduce_staging_202, [dq_row_199, 48], store_reduce="add")
        curr_m_205 = arg136 + 128
        accum_cnt = arg137 + 1
        arg136 = curr_m_205
        arg137 = accum_cnt
      tlx.async_descriptor_store_wait(0)
    with tlx.async_task(num_warps=1, registers=88):
      num_steps = N_CTX_94 // 128
      k_172 = tlx.cluster_cta_rank()
      k_173 = k_172 % 2
      k_174 = k_173 == 0
      k_175 = k_173 != 0
      k_176 = k_172 ^ 1
      k_177 = tlx.remote_view(k_171, k_176)
      tlx.barrier_wait(k_171, 0)
      tlx.fence("async_shared")
      tlx.barrier_arrive(k_177, k_174)
      tlx.barrier_wait(k_171, 0)
      kt_179 = tlx.remote_view(kt_178, k_176)
      tlx.barrier_wait(kt_178, 0)
      tlx.fence("async_shared")
      tlx.barrier_arrive(kt_179, k_174)
      tlx.barrier_wait(kt_178, 0)
      v_181 = tlx.remote_view(v_180, k_176)
      tlx.barrier_wait(v_180, 0)
      tlx.fence("async_shared")
      tlx.barrier_arrive(v_181, k_174)
      tlx.barrier_wait(v_180, 0)
      tlx.barrier_wait(dv_182, 1)
      tlx.barrier_wait(dk_183, 1)
      curr_m = num_steps > 0
      qT_185 = tlx.local_trans(qT_184)
      qT_187 = tlx.remote_view(qT_186, k_176)
      curr_m_188 = curr_m & k_174
      tlx.barrier_wait(qT_186, 0)
      tlx.fence("async_shared")
      tlx.barrier_arrive(qT_187, curr_m_188)
      curr_m_189 = curr_m & k_175
      tlx.barrier_wait(qT_186, 0)
      tlx.barrier_wait(qkT_194, 1)
      qkT_196 = tlx.cluster_cta_rank()
      qkT_197 = qkT_196 & -2
      qkT_198 = tlx.remote_view(desc_dk_staging_195, qkT_197)
      tlx.barrier_arrive(qkT_198)
      qkT_199 = qkT_196 % 2
      qkT_200 = qkT_199 == 0
      tlx.barrier_wait(desc_dk_staging_195, 0)
      tlx.fence("async_shared")
      tlx.async_dot(k_190, qT_185, qkT_191, use_acc=False, pred=curr_m, mBarriers=[qT_192, qkT_193], two_ctas=True, force_async=True)
      dpT_202 = tlx.local_trans(dpT_201)
      dpT_204 = tlx.remote_view(dpT_203, k_176)
      tlx.barrier_wait(dpT_203, 0)
      tlx.fence("async_shared")
      tlx.barrier_arrive(dpT_204, curr_m_188)
      tlx.barrier_wait(dpT_203, 0)
      tlx.barrier_wait(dpT_209, 1)
      tlx.barrier_wait(dsT_210, 1)
      dpT_212 = tlx.remote_view(desc_dk_staging_211, qkT_197)
      tlx.barrier_arrive(dpT_212)
      tlx.barrier_wait(desc_dk_staging_211, 0)
      tlx.async_dot(v_205, dpT_202, dpT_206, use_acc=False, pred=curr_m, mBarriers=[dpT_207, dpT_208], two_ctas=True, force_async=True)
      qkT_215 = tlx.subslice(qkT_95)
      qkT_216 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=qkT_215)
      do_220 = tlx.remote_view(do_219, k_176)
      tlx.barrier_wait(do_219, 0)
      tlx.fence("async_shared")
      tlx.barrier_arrive(do_220, curr_m_188)
      tlx.barrier_wait(do_219, 0)
      tlx.barrier_wait(ppT_222, 0)
      dv_224 = tlx.remote_view(desc_dk_staging_223, qkT_197)
      tlx.barrier_arrive(dv_224)
      tlx.barrier_wait(desc_dk_staging_223, 0)
      tlx.async_dot(ppT_217, do_214, dv_213, use_acc=False, pred=curr_m, mBarriers=[do_218, ppT_221], two_ctas=True, force_async=True)
      curr_m_225 = num_steps - 1
      arg136 = False
      arg137 = 0
      arg138 = False
      for arg135 in range(0, curr_m_225, 1):
        accum_cnt = arg137 + 1
        qT_232 = accum_cnt & 1
        tlx.barrier_wait(qT_186, qT_232)
        tlx.fence("async_shared")
        tlx.barrier_arrive(qT_187, k_174)
        tlx.barrier_wait(qT_186, qT_232)
        q_235 = arg137 & 1
        qkT_237 = qT_232 ^ True
        tlx.barrier_wait(qkT_194, qkT_237)
        qkT_240 = tlx.remote_view(desc_dk_staging_239, qkT_197)
        tlx.barrier_arrive(qkT_240)
        qkT_242 = arg135 % 2
        tlx.barrier_wait(desc_dk_staging_239, qkT_242)
        tlx.fence("async_shared")
        tlx.async_dot(k_190, qT_185, qkT_191, use_acc=False, pred=True, mBarriers=[qT_192, qkT_193], two_ctas=True, force_async=True)
        dpT_246 = tlx.subslice(dpT_110)
        dpT_247 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=dpT_246)
        q_252 = tlx.remote_view(q_250, k_176)
        tlx.barrier_wait(q_250, q_235)
        tlx.fence("async_shared")
        tlx.barrier_arrive(q_252, k_174)
        tlx.barrier_wait(q_250, q_235)
        tlx.barrier_wait(dsT_253, arg138)
        dk_256 = tlx.remote_view(desc_dk_staging_255, qkT_197)
        tlx.barrier_arrive(dk_256)
        tlx.barrier_wait(desc_dk_staging_255, qkT_242)
        tlx.async_dot(dsT_248, q_245, dk_244, use_acc=arg136, pred=True, mBarriers=[q_249, dsT_210], two_ctas=True, force_async=True)
        tlx.barrier_wait(dpT_203, qT_232)
        tlx.fence("async_shared")
        tlx.barrier_arrive(dpT_204, k_174)
        tlx.barrier_wait(dpT_203, qT_232)
        tlx.barrier_wait(dpT_209, qkT_237)
        dpT_257 = qT_232 ^ True
        tlx.barrier_wait(dsT_210, dpT_257)
        dpT_260 = tlx.remote_view(desc_dk_staging_259, qkT_197)
        tlx.barrier_arrive(dpT_260)
        tlx.barrier_wait(desc_dk_staging_259, qkT_242)
        tlx.async_dot(v_205, dpT_202, dpT_206, use_acc=False, pred=True, mBarriers=[dpT_207, dpT_208], two_ctas=True, force_async=True)
        dsT_dq = tlx.local_trans(dsT_dq_1_261)
        tlx.barrier_wait(dsT_dq_1_262, q_235)
        qkT_265 = tlx.subslice(qkT_95)
        qkT_266 = tlx.local_alloc((64, 128), tl.float32, 1, tlx.storage_kind.tmem, reuse=qkT_265)
        dq_271 = q_235 ^ True
        tlx.barrier_wait(dq_270, dq_271)
        dq_274 = tlx.remote_view(desc_dk_staging_273, qkT_197)
        tlx.barrier_arrive(dq_274)
        tlx.barrier_wait(desc_dk_staging_273, qkT_242)
        tlx.fence("async_shared")
        tlx.async_dot(dsT_dq, kt_264, dq_267, use_acc=False, pred=True, mBarriers=[dsT_dq_1_268, dq_269], two_ctas=True, force_async=True)
        tlx.barrier_wait(do_219, qT_232)
        tlx.fence("async_shared")
        tlx.barrier_arrive(do_220, k_174)
        tlx.barrier_wait(do_219, qT_232)
        tlx.barrier_wait(ppT_222, qT_232)
        dv_277 = tlx.remote_view(desc_dk_staging_276, qkT_197)
        tlx.barrier_arrive(dv_277)
        tlx.barrier_wait(desc_dk_staging_276, qkT_242)
        tlx.async_dot(ppT_217, do_214, dv_213, use_acc=True, pred=True, mBarriers=[do_218, ppT_221], two_ctas=True, force_async=True)
        arg136 = True
        arg137 = accum_cnt
        arg138 = qT_232
      if curr_m:
        q_232 = arg137 & 1
        dpT_236 = tlx.subslice(dpT_110)
        dpT_237 = tlx.local_alloc((128, 128), tl.float16, 1, tlx.storage_kind.tmem, reuse=dpT_236)
        q_242 = tlx.remote_view(q_240, k_176)
        tlx.barrier_wait(q_240, q_232)
        tlx.fence("async_shared")
        tlx.barrier_arrive(q_242, k_174)
        tlx.barrier_wait(q_240, q_232)
        tlx.barrier_wait(dsT_243, arg138)
        dk_246 = tlx.remote_view(desc_dk_staging_245, qkT_197)
        tlx.barrier_arrive(dk_246)
        tlx.barrier_wait(desc_dk_staging_245, 0)
        tlx.async_dot(dsT_238, q_235, dk_234, use_acc=arg136, pred=True, mBarriers=[q_239, dsT_210], two_ctas=True, force_async=True)
        dsT_dq = tlx.local_trans(dsT_dq_1_247)
        tlx.barrier_wait(dsT_dq_1_248, q_232)
        qkT_251 = tlx.subslice(qkT_95)
        qkT_252 = tlx.local_alloc((64, 128), tl.float32, 1, tlx.storage_kind.tmem, reuse=qkT_251)
        dq_257 = q_232 ^ True
        tlx.barrier_wait(dq_256, dq_257)
        dq_260 = tlx.remote_view(desc_dk_staging_259, qkT_197)
        tlx.barrier_arrive(dq_260)
        tlx.barrier_wait(desc_dk_staging_259, 0)
        tlx.fence("async_shared")
        tlx.async_dot(dsT_dq, kt_250, dq_253, use_acc=False, pred=True, mBarriers=[dsT_dq_1_254, dq_255], two_ctas=True, force_async=True)
      tlx.tcgen05_commit(dk_227)
      tlx.tcgen05_commit(dv_228)
      tlx.tcgen05_commit(v_229)
      tlx.tcgen05_commit(kt_230)
      tlx.tcgen05_commit(k_231)
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
      k_183 = tlx.cluster_cta_rank()
      k_184 = k_183 & -2
      k_185 = tlx.remote_view(k_182, k_184)
      k_186 = k_183 % 2
      k_187 = k_186 == 0
      tlx.barrier_expect_bytes(k_182, 65536)
      tlx.async_descriptor_load(desc_k_131, k_188, [k_179, 0], k_185, two_ctas=True)
      kt_start_n = cluster_cta_rank * 128
      kt_start_n_189 = start_n - kt_start_n
      kt_191 = off_bh_177 + kt_start_n_189
      kt_193 = k_183 % 2
      kt_194 = kt_193 * 64
      tlx.barrier_wait(kt_195, 1)
      kt_197 = tlx.remote_view(kt_196, k_184)
      tlx.barrier_expect_bytes(kt_196, 65536)
      tlx.async_descriptor_load(desc_kt_132, kt_198, [kt_191, kt_194], kt_197, two_ctas=True)
      tlx.barrier_wait(v_199, 1)
      v_201 = tlx.remote_view(v_200, k_184)
      tlx.barrier_expect_bytes(v_200, 65536)
      tlx.async_descriptor_load(desc_v_133, v_202, [k_179, 0], v_201, two_ctas=True)
      num_steps = N_CTX_94 // 128
      curr_m = num_steps > 0
      qt_203 = off_bh_177 + kt_194
      tlx.barrier_wait(qT_204, 1)
      qT_206 = tlx.remote_view(qT_205, k_184)
      curr_m_207 = curr_m & k_187
      tlx.barrier_expect_bytes(qT_205, 32768)
      tlx.async_descriptor_load(desc_qt_134, qT_208, [qt_203, 0], qT_206, two_ctas=True)
      tlx.barrier_wait(dpT_209, 1)
      dpT_211 = tlx.remote_view(dpT_210, k_184)
      tlx.barrier_expect_bytes(dpT_210, 32768)
      tlx.async_descriptor_load(desc_dot_140, dpT_212, [qt_203, 0], dpT_211, two_ctas=True)
      tlx.barrier_wait(m_213, 1)
      tlx.barrier_expect_bytes(m_214, 512)
      tlx.async_descriptor_load(desc_m_138, m_215, [off_chz], m_214)
      tlx.barrier_wait(do_216, 1)
      do_218 = tlx.remote_view(do_217, k_184)
      tlx.barrier_expect_bytes(do_217, 32768)
      tlx.async_descriptor_load(desc_do_139, do_219, [off_bh_177, kt_194], do_218, two_ctas=True)
      curr_m_220 = num_steps - 1
      arg136 = 0
      arg137 = 0
      arg138 = off_bh_177
      arg139 = off_chz
      for arg135 in range(0, curr_m_220, 1):
        curr_m_222 = arg136 + 128
        accum_cnt = arg137 + 1
        qt_224 = off_bh_177 + curr_m_222
        qt_226 = qt_224 + kt_194
        qT_227 = accum_cnt & 1
        qt_229 = qT_227 ^ True
        tlx.barrier_wait(qT_204, qt_229)
        tlx.barrier_expect_bytes(qT_205, 32768)
        tlx.async_descriptor_load(desc_qt_134, qT_208, [qt_226, 0], qT_206, two_ctas=True)
        q_231 = arg137 & 1
        q_234 = q_231 ^ True
        tlx.barrier_wait(q_233, q_234)
        q_237 = tlx.remote_view(q_236, k_184)
        tlx.barrier_expect_bytes(q_236, 32768)
        tlx.async_descriptor_load(desc_q_135, q_238, [arg138, kt_194], q_237, two_ctas=True)
        Di_240 = q_231 ^ True
        tlx.barrier_wait(Di_239, Di_240)
        tlx.barrier_expect_bytes(Di_242, 512)
        tlx.async_descriptor_load(desc_delta_143, Di_243, [arg139], Di_242)
        tlx.barrier_wait(dpT_209, qt_229)
        tlx.barrier_expect_bytes(dpT_210, 32768)
        tlx.async_descriptor_load(desc_dot_140, dpT_212, [qt_226, 0], dpT_211, two_ctas=True)
        offs_m_start = off_chz + curr_m_222
        m_245 = qT_227 ^ True
        tlx.barrier_wait(m_213, m_245)
        tlx.barrier_expect_bytes(m_214, 512)
        tlx.async_descriptor_load(desc_m_138, m_215, [offs_m_start], m_214)
        tlx.barrier_wait(do_216, qt_229)
        tlx.barrier_expect_bytes(do_217, 32768)
        tlx.async_descriptor_load(desc_do_139, do_219, [qt_224, kt_194], do_218, two_ctas=True)
        arg136 = curr_m_222
        arg137 = accum_cnt
        arg138 = qt_224
        arg139 = offs_m_start
      if curr_m:
        q_222 = arg137 & 1
        q_225 = q_222 ^ True
        tlx.barrier_wait(q_224, q_225)
        q_228 = tlx.remote_view(q_227, k_184)
        tlx.barrier_expect_bytes(q_227, 32768)
        tlx.async_descriptor_load(desc_q_135, q_229, [arg138, kt_194], q_228, two_ctas=True)
        Di_231 = q_222 ^ True
        tlx.barrier_wait(Di_230, Di_231)
        tlx.barrier_expect_bytes(Di_233, 512)
        tlx.async_descriptor_load(desc_delta_143, Di_234, [arg139], Di_233)
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
