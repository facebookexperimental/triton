# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TTGIR-emitted TLX baseline for the pinned gfx950 Triton V3 backward.

This is the specialized source emitted with TRITON_DUMP_TLX_BENCHMARK=1 from
the final TTGIR selected for B=2048, H=H_kv=1, Q=256, KV=2048, D=128,
BF16 softmax, shared K/V, and non-causal masking. The compile configuration is
BM32, BN128, MFMA16, four warps, one stage, and waves_per_eu=1.
"""

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _backward_delta(Delta, OUT, rows, mask, do):
    if OUT is None:
        return tl.load(Delta + rows, mask=mask)
    else:
        offsets = rows[:, None] * 128 + tl.arange(0, 128)[None, :]
        out = tl.load(OUT + offsets, mask=mask[:, None], other=0)
        return tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)


@triton.jit
def _dot(lhs, rhs, acc):
    # Triton TR011: preserve the captured specialization's TF32 policy.
    return tl.dot(lhs, rhs, acc, allow_tf32=True)


@triton.jit
def _tlx_gfx950_cross_attn_v3_ttgir_bwd_impl(  # noqa: C901
    Q,
    K,
    V,
    DO,
    total_seq_len_q,
    total_seq_len_kv,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    scaled_alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    num_targets,
    Z,
    AUTOTUNE_Z,
    AUTOTUNE_MAX_Q_LEN,
    AUTOTUNE_MAX_SEQ_LEN,
    DIRECT_FIRST_DQ_STORE: tl.constexpr,
    ATOMIC_DQ: tl.constexpr,
    PREFETCH_DQ: tl.constexpr,
    STAGE_QDO: tl.constexpr,
    PIPELINE_QDO: tl.constexpr,
    DQ_FINAL=None,
    OUT=None,
    KV_SPLITS: tl.constexpr = 1,
    CACHE_DELTA: tl.constexpr = False,
    FIXED_Q: tl.constexpr = False,
    EARLY_TAIL_QDO: tl.constexpr = False,
    CACHE_K: tl.constexpr = False,
    EARLY_COARSE_STATS: tl.constexpr = False,
    SINGLE_KV_TILE: tl.constexpr = False,
    PACKED_FIXED_Q: tl.constexpr = False,
    NATIVE_Q_SCORE: tl.constexpr = False,
    FIXED_KV: tl.constexpr = 0,
):
    # The pipeline writes the final FP32 sum directly to the BF16 output.
    if DQ_FINAL is not None:
        tl.static_assert(PIPELINE_QDO and DIRECT_FIRST_DQ_STORE and not ATOMIC_DQ)
        dq_output = DQ_FINAL
    else:
        dq_output = DQ
    cst = tl.full([128, 32], 0, tl.float32)
    cst_0 = tl.full([32, 128], 0, tl.float32)
    cst_1 = tl.full([128, 128], 0, tl.bfloat16)
    cst_2 = tl.full([32, 128], 0, tl.bfloat16)
    cst_4 = tl.full([128, 128], 0, tl.float32)
    cst_5 = tl.full([128, 32], 0, tl.float32)
    off_hz = tl.program_id(axis=0)  # triton_bw_cross_attention.py:1432
    seq_end_kv = off_hz + 1
    if FIXED_KV:
        tl.static_assert(FIXED_KV == 384 and PACKED_FIXED_Q and KV_SPLITS == 1)
        seq_start_kv_6 = off_hz.to(tl.int64) * FIXED_KV
        seq_end_kv_9 = seq_start_kv_6 + FIXED_KV
        single_key: tl.constexpr = False
    else:
        seq_start_kv_6 = tl.load(seq_offsets + off_hz)
        seq_end_kv_9 = tl.load(seq_offsets + seq_end_kv)
        single_key = seq_end_kv_9 - seq_start_kv_6 == 1
    if KV_SPLITS > 1:
        tl.static_assert(DQ_FINAL is None and PIPELINE_QDO and DIRECT_FIRST_DQ_STORE and not ATOMIC_DQ)
        tl.static_assert(DQ.dtype.element_ty == tl.float32)
        split = tl.program_id(1)
        split_span = tl.cdiv(tl.cdiv(seq_end_kv_9 - seq_start_kv_6, 128), KV_SPLITS) * 128
        seq_start_kv_6 = tl.minimum(seq_start_kv_6 + split * split_span, seq_end_kv_9)
        seq_end_kv_9 = tl.minimum(seq_start_kv_6 + split_span, seq_end_kv_9)
        DQ += split.to(tl.int64) * total_seq_len_q * stride_dqm
        dq_output = DQ
    if FIXED_KV:
        seq_len_kv: tl.constexpr = FIXED_KV
    else:
        seq_len_kv = seq_end_kv_9 - seq_start_kv_6
    if PACKED_FIXED_Q:
        tl.static_assert(FIXED_Q and PIPELINE_QDO and KV_SPLITS == 1)
        seq_start_q_11 = off_hz.to(tl.int64) * 256
        seq_end_q_13 = seq_start_q_11 + 256
    else:
        seq_start_q = seq_offsets_q + off_hz
        seq_start_q_11 = tl.load(seq_start_q)
        seq_end_q = seq_offsets_q + seq_end_kv
        seq_end_q_13 = tl.load(seq_end_q)
    if FIXED_Q:
        seq_len_q: tl.constexpr = 256
    else:
        seq_len_q = seq_end_q_13 - seq_start_q_11
    if seq_len_kv == 0:
        empty_q_offsets = tl.arange(0, 32)
        empty_d_offsets = tl.arange(0, 128)
        for empty_q_start in range(0, seq_len_q, 32):
            empty_q_rows = seq_start_q_11 + empty_q_start + empty_q_offsets
            empty_dq_ptrs = tl.expand_dims(empty_q_rows, axis=1) * stride_dqm + tl.expand_dims(empty_d_offsets, axis=0)
            tl.store(
                dq_output + empty_dq_ptrs,
                cst_0.to(DQ.dtype.element_ty),
                mask=tl.expand_dims(empty_q_rows < seq_end_q_13, axis=1),
            )
    elif KV_SPLITS == 1 and single_key:
        # A one-key softmax has zero score derivatives and unit probability.
        single_rows = tl.arange(0, 32)
        single_columns = tl.arange(0, 128)
        single_dkv = tl.full((128, ), 0, tl.float32)
        for single_start in range(0, seq_len_q, 32):
            single_q_rows = seq_start_q_11 + single_start + single_rows
            single_valid = single_q_rows < seq_end_q_13
            single_do = tl.load(DO + single_q_rows[:, None] * stride_dom + single_columns[None, :],
                                mask=single_valid[:, None], other=0).to(tl.float32)
            single_dkv += tl.sum(single_do, axis=0)
            tl.store(dq_output + single_q_rows[:, None] * stride_dqm + single_columns[None, :], 0.0,
                     mask=single_valid[:, None])
        tl.store(DK + seq_start_kv_6 * stride_dkn + single_columns, single_dkv)
    else:
        last_block_start_n = seq_len_kv + 127  # standard.py:43
        last_block_start_n_15 = last_block_start_n // 128  # standard.py:43
        last_block_start_n_16 = (last_block_start_n_15 - 1)  # triton_bw_cross_attention.py:2225
        last_block_start_n_17 = (last_block_start_n_16 * 128)  # triton_bw_cross_attention.py:2225
        if SINGLE_KV_TILE:
            # The caller guarantees that each sequence has at most 128 KV rows.
            tl.static_assert(KV_SPLITS == 1)
            last_block_start_n_18: tl.constexpr = 0
        else:
            last_block_start_n_18 = tl.maximum(last_block_start_n_17, 0)
        # Use the captured general-query path for fixed and ragged sequences.
        # The unused short-query path has conflicting emitted SSA names.
        # pyrefly: ignore [bad-assignment]
        short_q: tl.constexpr = False
        if short_q:
            offs_m = tl.arange(0, 32)  # triton_bw_cross_attention.py:2229
            offs_m_19 = tl.arange(0, 32)  # triton_bw_cross_attention.py:2229
            offs_m_20 = tl.arange(0, 32)  # triton_bw_cross_attention.py:2229
            mask_m_22 = offs_m_19 < seq_len_q  # triton_bw_cross_attention.py:2230
            mask_m_23 = offs_m_20 < seq_len_q  # triton_bw_cross_attention.py:2230
            DK_off_25 = seq_start_kv_6 * stride_dkn  # triton_bw_cross_attention.py:2238
            scaled_alpha_27 = (scaled_alpha * 1.44269502)  # triton_bw_cross_attention.py:2250
            offs_r_28 = seq_start_q_11 + offs_m  # triton_bw_cross_attention.py:1581
            offs_d = tl.arange(0, 128)  # triton_bw_cross_attention.py:1582
            offs_d_29 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1582
            offs_d_30 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1582
            ptrs_32 = tl.expand_dims(offs_r_28, axis=1)  # triton_bw_cross_attention.py:1584
            ptrs_34 = ptrs_32 * stride_qm  # triton_bw_cross_attention.py:1584
            ptrs_36 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1586
            ptrs_37 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1586
            ptrs_38 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1586
            ptrs_40 = tl.expand_dims(ptrs_37, axis=0)  # triton_bw_cross_attention.py:1584
            ptrs_42 = ptrs_34 + ptrs_40  # triton_bw_cross_attention.py:1584
            mask_43 = offs_r_28 < seq_end_q_13  # triton_bw_cross_attention.py:1588
            mask_44 = tl.expand_dims(mask_43, axis=1)  # triton_bw_cross_attention.py:1588
            q_45 = tl.load(Q + ptrs_42, mask=mask_44, other=cst_2)  # triton_bw_cross_attention.py:1589
            ptrs_49 = ptrs_32 * stride_dom  # triton_bw_cross_attention.py:1584
            ptrs_52 = ptrs_49 + ptrs_40  # triton_bw_cross_attention.py:1584
            do = tl.load(DO + ptrs_52, mask=mask_44, other=cst_2)  # triton_bw_cross_attention.py:1589
            q_trans = tl.trans(q_45)  # triton_bw_cross_attention.py:2275
            valid_mask_trans_57 = tl.expand_dims(offs_m_19, axis=0)  # triton_bw_cross_attention.py:2306
            valid_mask_trans_59 = (valid_mask_trans_57 < seq_len_q)  # triton_bw_cross_attention.py:2306
            m_61 = offs_m_19 + seq_start_q_11  # triton_attention_utils.py:245
            dact_qk_trans = tl.trans(do)  # triton_bw_cross_attention.py:2327
            dq_acc_72 = cst
            for start_n in range(0, seq_len_kv, 128):
                k_offset = seq_start_kv_6 + start_n  # triton_bw_cross_attention.py:2277
                offs_r_74 = k_offset + offs_d_29  # triton_bw_cross_attention.py:1581
                ptrs_76 = tl.expand_dims(offs_r_74, axis=1)  # triton_bw_cross_attention.py:1584
                ptrs_78 = ptrs_76 * stride_kn  # triton_bw_cross_attention.py:1584
                ptrs_82 = ptrs_78 + ptrs_40  # triton_bw_cross_attention.py:1584
                mask_83 = offs_r_74 < seq_end_kv_9  # triton_bw_cross_attention.py:1588
                mask_84 = tl.expand_dims(mask_83, axis=1)  # triton_bw_cross_attention.py:1588
                k_85 = tl.load(K + ptrs_82, mask=mask_84, other=cst_1)  # triton_bw_cross_attention.py:1589
                qk_trans_88 = _dot(k_85, q_trans, cst_5)  # triton_bw_cross_attention.py:2303
                offs_n_90 = start_n + offs_d_30  # triton_bw_cross_attention.py:2304
                offs_n_91 = start_n + offs_d  # triton_bw_cross_attention.py:2304
                valid_mask_trans_92 = tl.expand_dims(offs_n_90, axis=1)  # triton_bw_cross_attention.py:2305
                valid_mask_trans_93 = (valid_mask_trans_92 < seq_len_kv)  # triton_bw_cross_attention.py:2305
                valid_mask_trans_95 = (valid_mask_trans_93 & valid_mask_trans_59)  # triton_bw_cross_attention.py:2305
                qk_trans_96 = (qk_trans_88 * scaled_alpha_27)  # triton_attention_utils.py:244
                m_97 = tl.load(M + m_61, mask=mask_m_22)  # triton_attention_utils.py:245
                pT = tl.expand_dims(m_97, axis=0)  # triton_attention_utils.py:246
                pT_99 = qk_trans_96 - pT  # triton_attention_utils.py:246
                pT_100 = tl.math.exp2(pT_99)  # triton_attention_utils.py:246
                pT_101 = tl.where(valid_mask_trans_95, pT_100, cst_5)  # triton_attention_utils.py:247
                dact_qk_trans_102 = _dot(k_85, dact_qk_trans, cst_5)  # triton_bw_cross_attention.py:2327
                Di = _backward_delta(Delta, OUT, m_61, mask_m_22, do)  # triton_hstu_cross_attention.py:762
                dqk_trans = tl.expand_dims(Di, axis=0)  # triton_hstu_cross_attention.py:763
                dqk_trans_104 = (dact_qk_trans_102 - dqk_trans)  # triton_hstu_cross_attention.py:763
                dqk_trans_105 = (pT_101 * dqk_trans_104)  # triton_hstu_cross_attention.py:763
                dqk_trans_105 = dqk_trans_105.to(tl.bfloat16)
                dk_attn = _dot(dqk_trans_105, q_45, cst_4)  # triton_bw_cross_attention.py:2338
                dk_108 = dk_attn * scaled_alpha  # triton_bw_cross_attention.py:2340
                dk_110 = _dot(pT_101.to(tl.bfloat16), do, dk_108)  # triton_bw_cross_attention.py:2324
                dk_ptrs_111 = tl.expand_dims(offs_d, axis=1)  # triton_bw_cross_attention.py:2344
                dk_ptrs_112 = start_n * stride_dkn  # triton_bw_cross_attention.py:2344
                dk_ptrs_114 = (dk_ptrs_111 * stride_dkn)  # triton_bw_cross_attention.py:2344
                dk_ptrs_116 = tl.expand_dims(ptrs_36, axis=0)  # triton_bw_cross_attention.py:2344
                dk_ptrs_118 = (dk_ptrs_114 + dk_ptrs_116)  # triton_bw_cross_attention.py:2344
                dk_ptrs_119 = (dk_ptrs_118 + DK_off_25)  # triton_bw_cross_attention.py:2344
                dk_ptrs_121 = (dk_ptrs_112 + dk_ptrs_119)  # triton_bw_cross_attention.py:2344
                mask_n_122 = offs_n_91 < seq_len_kv  # triton_bw_cross_attention.py:2345
                var_4 = tl.expand_dims(mask_n_122, axis=1)  # triton_bw_cross_attention.py:2351
                tl.store(DK + dk_ptrs_121, dk_110, mask=var_4)  # triton_bw_cross_attention.py:2351
                dq_trans_123 = tl.trans(k_85)  # triton_bw_cross_attention.py:2367
                dq_trans_125 = _dot(dq_trans_123, dqk_trans_105, cst_5)  # triton_bw_cross_attention.py:2367
                dq_trans_127 = (dq_trans_125 * scaled_alpha)  # triton_bw_cross_attention.py:2368
                dq_acc_128 = (dq_acc_72 + dq_trans_127)  # triton_bw_cross_attention.py:2369
                dq_acc_72 = dq_acc_128
            dq = tl.trans(dq_acc_72)  # triton_bw_cross_attention.py:2370
            dq_ptrs = seq_start_q_11 * stride_dqm  # triton_bw_cross_attention.py:2372
            dq_ptrs_63 = tl.expand_dims(offs_m_20, axis=1)  # triton_bw_cross_attention.py:2372
            dq_ptrs_65 = dq_ptrs_63 * stride_dqm  # triton_bw_cross_attention.py:2372
            dq_ptrs_67 = tl.expand_dims(ptrs_38, axis=0)  # triton_bw_cross_attention.py:2372
            dq_ptrs_69 = dq_ptrs_65 + dq_ptrs_67  # triton_bw_cross_attention.py:2372
            dq_ptrs_71 = dq_ptrs + dq_ptrs_69  # triton_bw_cross_attention.py:2372
            var_1 = tl.expand_dims(mask_m_23, axis=1)  # triton_bw_cross_attention.py:2377
            tl.store(DQ + dq_ptrs_71, dq, mask=var_1)  # triton_bw_cross_attention.py:2377
        else:
            offs_r = tl.arange(0, 128)  # triton_bw_cross_attention.py:1581
            offs_r_19 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1581
            ptrs_20 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1586
            ptrs_21 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1586
            ptrs_22 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1586
            scaled_alpha_23 = (scaled_alpha * 1.44269502)  # triton_bw_cross_attention.py:3659
            offs_m = tl.arange(0, 32)  # triton_bw_cross_attention.py:3667
            offs_m_24 = tl.arange(0, 32)  # triton_bw_cross_attention.py:3667
            offs_m_25 = tl.arange(0, 32)  # triton_bw_cross_attention.py:3667
            if CACHE_DELTA:
                tl.static_assert(OUT is not None and KV_SPLITS > 1)
                delta_rows = seq_start_q_11 + tl.arange(0, 256)
                delta_mask = delta_rows[:, None] < seq_end_q_13
                delta_cols = tl.arange(0, 128)
                delta_out = tl.load(OUT + delta_rows[:, None] * 128 + delta_cols[None, :], delta_mask, other=0)
                delta_do = tl.load(DO + delta_rows[:, None] * stride_dom + delta_cols[None, :], delta_mask, other=0)
                cached_delta = tl.sum(delta_out.to(tl.float32) * delta_do.to(tl.float32), axis=1)
                delta_stage = tlx.local_alloc((256, ), tl.float32, 1)
                delta_view = tlx.local_view(delta_stage, 0)
                tlx.local_store(delta_view, cached_delta)
            else:
                delta_view = None
            if STAGE_QDO:
                if (SINGLE_KV_TILE or CACHE_DELTA or NATIVE_Q_SCORE) and PIPELINE_QDO:
                    native_parent: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32],
                                                                      transposed=True, warps_per_cta=[4, 1])
                    native_score_b: tl.constexpr = tlx.dot_operand_layout(1, native_parent, k_width=8)
                # pyrefly: ignore [bad-argument-type]
                q_stage = tlx.local_alloc((32, 128), tlx.dtype_of(Q), 1)
                # pyrefly: ignore [bad-argument-type]
                do_stage = tlx.local_alloc((32, 128), tlx.dtype_of(DO), 1)
                q_view = tlx.local_view(q_stage, 0)
                do_view = tlx.local_view(do_stage, 0)
            else:
                q_stage = None
                do_stage = None
                q_view = None
                do_view = None
            for start_n in range(0, last_block_start_n_18, 128):
                if STAGE_QDO and PIPELINE_QDO:
                    assert q_view is not None
                    assert do_view is not None
                    first_q_rows = seq_start_q_11 + offs_m
                    first_q_ptrs = tl.expand_dims(first_q_rows, axis=1) * stride_qm + tl.expand_dims(ptrs_21, axis=0)
                    first_q_mask = tl.expand_dims(first_q_rows < seq_end_q_13, axis=1)
                    first_q_token = tlx.async_load(
                        Q + first_q_ptrs,
                        q_view,
                        mask=first_q_mask,
                        other=cst_2,
                    )
                    first_do_token = tlx.async_load(
                        DO + tl.expand_dims(first_q_rows, axis=1) * stride_dom + tl.expand_dims(ptrs_21, axis=0),
                        do_view,
                        mask=first_q_mask,
                        other=cst_2,
                    )
                    tlx.async_load_commit_group([first_q_token, first_do_token])
                kv_offset_69 = (seq_start_kv_6 + start_n)  # triton_bw_cross_attention.py:3611
                offs_r_72 = kv_offset_69 + offs_r  # triton_bw_cross_attention.py:1581
                offs_r_73 = (kv_offset_69 + offs_r_19)  # triton_bw_cross_attention.py:1581
                ptrs_76 = tl.expand_dims(offs_r_73, axis=1)  # triton_bw_cross_attention.py:1584
                ptrs_78 = ptrs_76 * stride_kn  # triton_bw_cross_attention.py:1584
                ptrs_81 = tl.expand_dims(ptrs_21, axis=0)  # triton_bw_cross_attention.py:1584
                ptrs_83 = ptrs_78 + ptrs_81  # triton_bw_cross_attention.py:1584
                mask_84 = offs_r_73 < seq_end_kv_9  # triton_bw_cross_attention.py:1588
                mask_85 = tl.expand_dims(mask_84, axis=1)  # triton_bw_cross_attention.py:1588
                k_87 = tl.load(
                    K + ptrs_83, mask=mask_85, other=cst_1,
                    cache_modifier=".cg" if KV_SPLITS > 1 or CACHE_K else "")  # triton_bw_cross_attention.py:1589
                dq_trans_89 = tl.trans(k_87)  # triton_bw_cross_attention.py:3744
                dk_103 = cst_4
                for start_m in range(0, seq_len_q, 32):
                    offs_m_106 = (start_m + offs_m_25)  # triton_bw_cross_attention.py:3667
                    offs_m_107 = (start_m + offs_m_24)  # triton_bw_cross_attention.py:3667
                    mask_m_108 = (offs_m_106 < seq_len_q)  # triton_bw_cross_attention.py:3668
                    mask_m_109 = (offs_m_107 < seq_len_q)  # triton_bw_cross_attention.py:3668
                    q_offset = (seq_start_q_11 + start_m)  # triton_bw_cross_attention.py:3669
                    offs_r_111 = q_offset + offs_m  # triton_bw_cross_attention.py:1581
                    ptrs_113 = tl.expand_dims(offs_r_111, axis=1)  # triton_bw_cross_attention.py:1584
                    ptrs_115 = ptrs_113 * stride_qm  # triton_bw_cross_attention.py:1584
                    ptrs_119 = ptrs_115 + ptrs_81  # triton_bw_cross_attention.py:1584
                    mask_120 = (offs_r_111 < seq_end_q_13)  # triton_bw_cross_attention.py:1588
                    mask_121 = tl.expand_dims(mask_120, axis=1)  # triton_bw_cross_attention.py:1588
                    do = cst_2
                    if STAGE_QDO:
                        assert q_view is not None
                        assert do_view is not None
                        if not PIPELINE_QDO:
                            q_token = tlx.async_load(
                                Q + ptrs_119,
                                q_view,
                                mask=mask_121,
                                other=cst_2,
                            )
                            do_token = tlx.async_load(
                                DO + ptrs_113 * stride_dom + ptrs_81,
                                do_view,
                                mask=mask_121,
                                other=cst_2,
                            )
                            tlx.async_load_commit_group([q_token, do_token])
                        if CACHE_DELTA:
                            cached_rows = seq_start_q_11 + start_m + offs_m_25
                            cached_lse = tl.load(M + cached_rows, mask=mask_m_108)
                        # pyrefly: ignore [bad-argument-type]
                        qdo_wait = tlx.async_load_wait_group(0)
                        q_122 = tlx.local_load(q_view, token=qdo_wait)
                        if PIPELINE_QDO:
                            do = tlx.local_load(do_view, token=qdo_wait)
                            if CACHE_DELTA or NATIVE_Q_SCORE:
                                q_native_bulk_t = tlx.require_layout(
                                    tlx.local_load(tlx.local_trans(q_view), token=qdo_wait, layout=native_score_b),
                                    native_score_b)
                            if CACHE_DELTA:
                                do_native_bulk_t = tlx.require_layout(
                                    tlx.local_load(tlx.local_trans(do_view), token=qdo_wait, layout=native_score_b),
                                    native_score_b)
                            if CACHE_DELTA:
                                current_cached_delta = tlx.local_load(
                                    tlx.local_slice(delta_view, [start_m.to(tl.int32)], [32]))
                            if (EARLY_COARSE_STATS and KV_SPLITS > 1 or NATIVE_Q_SCORE
                                    or SINGLE_KV_TILE and KV_SPLITS == 1) and not CACHE_DELTA:
                                early_rows = seq_start_q_11 + start_m + offs_m_25
                                early_lse = tl.load(M + early_rows, mask=mask_m_108)
                                early_delta = _backward_delta(Delta, OUT, early_rows, mask_m_108, do)
                            next_q_rows = seq_start_q_11 + start_m + 32 + offs_m
                            next_q_ptrs = tl.expand_dims(next_q_rows, axis=1) * stride_qm + tl.expand_dims(
                                ptrs_21, axis=0)
                            next_q_mask = tl.expand_dims(next_q_rows < seq_end_q_13, axis=1)
                            # pyrefly: ignore [bad-argument-type]
                            next_q_token = tlx.async_load(
                                Q + next_q_ptrs,
                                q_view,
                                mask=next_q_mask,
                                other=cst_2,
                            )
                            # pyrefly: ignore [bad-argument-type]
                            next_do_token = tlx.async_load(
                                DO + tl.expand_dims(next_q_rows, axis=1) * stride_dom + tl.expand_dims(ptrs_21, axis=0),
                                do_view,
                                mask=next_q_mask,
                                other=cst_2,
                            )
                            tlx.async_load_commit_group([next_q_token, next_do_token])
                    else:
                        q_122 = tl.load(Q + ptrs_119, mask=mask_121, other=cst_2)  # triton_bw_cross_attention.py:1589
                    qk_trans_124 = q_native_bulk_t if (
                        CACHE_DELTA or NATIVE_Q_SCORE) and STAGE_QDO and PIPELINE_QDO else tl.trans(q_122)
                    qk_trans_125 = _dot(k_87, qk_trans_124, cst_5)  # triton_bw_cross_attention.py:3681
                    valid_mask_trans_126 = tl.expand_dims(offs_m_106, axis=0)  # triton_bw_cross_attention.py:3695
                    valid_mask_trans_127 = (valid_mask_trans_126 < seq_len_q)  # triton_bw_cross_attention.py:3695
                    qk_trans_128 = (qk_trans_125 * scaled_alpha_23)  # triton_attention_utils.py:244
                    m_129 = offs_m_25 + seq_start_q_11  # triton_attention_utils.py:245
                    m_130 = start_m + m_129  # triton_attention_utils.py:245
                    if CACHE_DELTA and PIPELINE_QDO:
                        m_131 = cached_lse
                    elif (EARLY_COARSE_STATS and KV_SPLITS > 1 or NATIVE_Q_SCORE
                          or SINGLE_KV_TILE and KV_SPLITS == 1) and not CACHE_DELTA and PIPELINE_QDO:
                        m_131 = early_lse
                    else:
                        m_131 = tl.load(M + m_130, mask=mask_m_108)  # triton_attention_utils.py:245
                    pT = tl.expand_dims(m_131, axis=0)  # triton_attention_utils.py:246
                    pT_133 = qk_trans_128 - pT  # triton_attention_utils.py:246
                    pT_134 = tl.math.exp2(pT_133)  # triton_attention_utils.py:246
                    pT_136 = tl.where(valid_mask_trans_127, pT_134, cst_5)  # triton_attention_utils.py:247
                    ptrs_138 = (ptrs_113 * stride_dom)  # triton_bw_cross_attention.py:1584
                    ptrs_141 = ptrs_138 + ptrs_81  # triton_bw_cross_attention.py:1584
                    if STAGE_QDO:
                        if not PIPELINE_QDO:
                            # pyrefly: ignore [bad-argument-type, unbound-name]
                            do = tlx.local_load(do_view, token=qdo_wait)
                    else:
                        do = tl.load(DO + ptrs_141, mask=mask_121, other=cst_2)  # triton_bw_cross_attention.py:1589
                    prefetched_delta = tl.zeros([32], dtype=tl.float32)
                    if PREFETCH_DQ:
                        if CACHE_DELTA:
                            if PIPELINE_QDO:
                                prefetched_delta = current_cached_delta
                            else:
                                prefetched_delta = tlx.local_load(
                                    tlx.local_slice(delta_view, [start_m.to(tl.int32)], [32]))
                        else:
                            if (EARLY_COARSE_STATS and KV_SPLITS > 1 or NATIVE_Q_SCORE
                                    or SINGLE_KV_TILE and KV_SPLITS == 1) and not CACHE_DELTA and PIPELINE_QDO:
                                prefetched_delta = early_delta
                            else:
                                prefetched_delta = _backward_delta(Delta, OUT, m_130, mask_m_108, do)
                    dk_144 = _dot(pT_136.to(tl.bfloat16), do, dk_103)  # triton_bw_cross_attention.py:3724
                    dact_qk_trans_145 = do_native_bulk_t if CACHE_DELTA and STAGE_QDO and PIPELINE_QDO else tl.trans(do)
                    dact_qk_trans_146 = _dot(k_87, dact_qk_trans_145, cst_5)  # triton_bw_cross_attention.py:3729
                    if PREFETCH_DQ:
                        Di = prefetched_delta
                    else:
                        Di = _backward_delta(Delta, OUT, m_130, mask_m_108, do)  # triton_hstu_cross_attention.py:762
                    dqk_trans = tl.expand_dims(Di, axis=0)  # triton_hstu_cross_attention.py:763
                    dqk_trans_148 = (dact_qk_trans_146 - dqk_trans)  # triton_hstu_cross_attention.py:763
                    dqk_trans_149 = (pT_136 * dqk_trans_148)  # triton_hstu_cross_attention.py:763
                    dqk_trans_149 = dqk_trans_149.to(tl.bfloat16)
                    prefetched_dq = cst_0
                    if PREFETCH_DQ:
                        prefetch_dq_row = seq_start_q_11 + offs_m_107
                        prefetch_dq_ptrs = tl.expand_dims(prefetch_dq_row, axis=1) * stride_dqm + tl.expand_dims(
                            ptrs_22, axis=0)
                        prefetch_dq_mask = tl.expand_dims(mask_m_109, axis=1)
                        prefetched_dq = tl.load(
                            DQ + prefetch_dq_ptrs,
                            mask=prefetch_dq_mask & (start_n != 0),
                            other=cst_0,
                        )  # triton_bw_cross_attention.py:3760
                    dk_attn = _dot(dqk_trans_149, q_122, cst_4)  # triton_bw_cross_attention.py:3740
                    dk_153 = dk_attn * scaled_alpha  # triton_bw_cross_attention.py:3741
                    dk_154 = dk_144 + dk_153  # triton_bw_cross_attention.py:3741
                    if PIPELINE_QDO:
                        # Compute dQ in its output orientation to reduce register use.
                        dq_trans_156 = tl.trans(_dot(tl.trans(dqk_trans_149), k_87, cst_0))
                    else:
                        dq_trans_156 = _dot(dq_trans_89, dqk_trans_149, cst_5)
                    dq_trans_158 = (dq_trans_156 * scaled_alpha)  # triton_bw_cross_attention.py:3745
                    dq = tl.trans(dq_trans_158)  # triton_bw_cross_attention.py:3746
                    dq_ptrs_159 = (seq_start_q_11 + offs_m_107)  # triton_bw_cross_attention.py:3757
                    dq_ptrs_161 = tl.expand_dims(dq_ptrs_159, axis=1)  # triton_bw_cross_attention.py:3756
                    dq_ptrs_163 = (dq_ptrs_161 * stride_dqm)  # triton_bw_cross_attention.py:3756
                    dq_ptrs_166 = tl.expand_dims(ptrs_22, axis=0)  # triton_bw_cross_attention.py:3756
                    dq_ptrs_168 = (dq_ptrs_163 + dq_ptrs_166)  # triton_bw_cross_attention.py:3756
                    cur = tl.expand_dims(mask_m_109, axis=1)  # triton_bw_cross_attention.py:3760
                    if PIPELINE_QDO and PREFETCH_DQ and DIRECT_FIRST_DQ_STORE and not ATOMIC_DQ:
                        # The first tile prefetch supplies zero, so all tiles share one store.
                        tl.store(DQ + dq_ptrs_168, prefetched_dq + dq, mask=cur)
                    elif DIRECT_FIRST_DQ_STORE and start_n == 0:
                        tl.store(DQ + dq_ptrs_168, dq, mask=cur)
                    elif ATOMIC_DQ:
                        tl.atomic_add(
                            DQ + dq_ptrs_168,
                            dq.to(DQ.dtype.element_ty),
                            mask=cur,
                            sem="relaxed",
                        )
                    elif PREFETCH_DQ:
                        var_9 = prefetched_dq + dq
                        tl.store(DQ + dq_ptrs_168, var_9, mask=cur)
                    else:
                        cur_170 = tl.load(DQ + dq_ptrs_168, mask=cur, other=cst_0)  # triton_bw_cross_attention.py:3760
                        var_9 = cur_170 + dq  # triton_bw_cross_attention.py:3763
                        tl.store(DQ + dq_ptrs_168, var_9, mask=cur)  # triton_bw_cross_attention.py:3761
                    dk_103 = dk_154
                    if CACHE_DELTA:
                        tlx.amd_iglp_opt(2)
                if PIPELINE_QDO:
                    # pyrefly: ignore [bad-argument-type]
                    tlx.async_load_wait_group(0)
                offs_n_93 = start_n + offs_r  # triton_bw_cross_attention.py:3767
                dk_ptrs_94 = tl.expand_dims(offs_r_72, axis=1)  # triton_bw_cross_attention.py:3814
                dk_ptrs_96 = (dk_ptrs_94 * stride_dkn)  # triton_bw_cross_attention.py:3814
                dk_ptrs_99 = tl.expand_dims(ptrs_20, axis=0)  # triton_bw_cross_attention.py:3814
                dk_ptrs_101 = (dk_ptrs_96 + dk_ptrs_99)  # triton_bw_cross_attention.py:3814
                mask_n_102 = offs_n_93 < seq_len_kv  # triton_bw_cross_attention.py:3819
                var_5 = tl.expand_dims(mask_n_102, axis=1)  # triton_bw_cross_attention.py:3828
                tl.store(DK + dk_ptrs_101, dk_103, mask=var_5)  # triton_bw_cross_attention.py:3828
            if PIPELINE_QDO and STAGE_QDO and PREFETCH_DQ and DIRECT_FIRST_DQ_STORE and not ATOMIC_DQ:
                # Apply the Q/dO pipeline to the final masked KV tile.
                start_n = last_block_start_n_18
                if KV_SPLITS > 1 or EARLY_TAIL_QDO:
                    assert q_view is not None
                    assert do_view is not None
                    first_q_rows = seq_start_q_11 + offs_m
                    first_q_ptrs = tl.expand_dims(first_q_rows, axis=1) * stride_qm + tl.expand_dims(ptrs_21, axis=0)
                    first_q_mask = tl.expand_dims(first_q_rows < seq_end_q_13, axis=1)
                    first_q_token = tlx.async_load(Q + first_q_ptrs, q_view, mask=first_q_mask, other=cst_2)
                    first_do_token = tlx.async_load(
                        DO + tl.expand_dims(first_q_rows, axis=1) * stride_dom + tl.expand_dims(ptrs_21, axis=0),
                        do_view, mask=first_q_mask, other=cst_2)
                    tlx.async_load_commit_group([first_q_token, first_do_token])
                kv_offset_69 = seq_start_kv_6 + start_n
                offs_r_72 = kv_offset_69 + offs_r
                offs_r_73 = kv_offset_69 + offs_r_19
                ptrs_76 = tl.expand_dims(offs_r_73, axis=1)
                ptrs_78 = ptrs_76 * stride_kn
                ptrs_81 = tl.expand_dims(ptrs_21, axis=0)
                ptrs_83 = ptrs_78 + ptrs_81
                mask_84 = offs_r_73 < seq_end_kv_9
                mask_85 = tl.expand_dims(mask_84, axis=1)
                k_87 = tl.load(K + ptrs_83, mask=mask_85, other=cst_1,
                               cache_modifier=".cg" if KV_SPLITS > 1 or CACHE_K else "")
                dk_103 = cst_4
                if KV_SPLITS == 1 and not EARLY_TAIL_QDO:
                    assert q_view is not None
                    assert do_view is not None
                    first_q_rows = seq_start_q_11 + offs_m
                    first_q_ptrs = tl.expand_dims(first_q_rows, axis=1) * stride_qm + tl.expand_dims(ptrs_21, axis=0)
                    first_q_mask = tl.expand_dims(first_q_rows < seq_end_q_13, axis=1)
                    first_q_token = tlx.async_load(Q + first_q_ptrs, q_view, mask=first_q_mask, other=cst_2)
                    first_do_token = tlx.async_load(
                        DO + tl.expand_dims(first_q_rows, axis=1) * stride_dom + tl.expand_dims(ptrs_21, axis=0),
                        do_view, mask=first_q_mask, other=cst_2)
                    tlx.async_load_commit_group([first_q_token, first_do_token])
                for start_m in range(0, seq_len_q, 32):
                    offs_m_106 = start_m + offs_m_25
                    offs_m_107 = start_m + offs_m_24
                    mask_m_108 = offs_m_106 < seq_len_q
                    mask_m_109 = offs_m_107 < seq_len_q
                    assert q_view is not None
                    assert do_view is not None
                    if CACHE_DELTA:
                        cached_rows = seq_start_q_11 + start_m + offs_m_25
                        cached_lse = tl.load(M + cached_rows, mask=mask_m_108)
                    qdo_wait = tlx.async_load_wait_group(0)
                    q_122 = tlx.local_load(q_view, token=qdo_wait)
                    do = tlx.local_load(do_view, token=qdo_wait)
                    if SINGLE_KV_TILE or CACHE_DELTA or NATIVE_Q_SCORE:
                        q_native_t = tlx.require_layout(
                            tlx.local_load(tlx.local_trans(q_view), token=qdo_wait, layout=native_score_b),
                            native_score_b)
                    if SINGLE_KV_TILE or CACHE_DELTA:
                        do_native_t = tlx.require_layout(
                            tlx.local_load(tlx.local_trans(do_view), token=qdo_wait, layout=native_score_b),
                            native_score_b)
                    if CACHE_DELTA:
                        current_cached_delta = tlx.local_load(tlx.local_slice(delta_view, [start_m.to(tl.int32)], [32]))
                    if (EARLY_COARSE_STATS and KV_SPLITS > 1 or NATIVE_Q_SCORE
                            or SINGLE_KV_TILE and KV_SPLITS == 1) and not CACHE_DELTA:
                        early_rows = seq_start_q_11 + start_m + offs_m_25
                        early_lse = tl.load(M + early_rows, mask=mask_m_108)
                        early_delta = _backward_delta(Delta, OUT, early_rows, mask_m_108, do)
                    # Reuse LDS after both local loads complete.
                    next_q_rows = seq_start_q_11 + start_m + 32 + offs_m
                    next_q_ptrs = tl.expand_dims(next_q_rows, axis=1) * stride_qm + tl.expand_dims(ptrs_21, axis=0)
                    next_q_mask = tl.expand_dims(next_q_rows < seq_end_q_13, axis=1)
                    next_q_token = tlx.async_load(Q + next_q_ptrs, q_view, mask=next_q_mask, other=cst_2)
                    next_do_token = tlx.async_load(
                        DO + tl.expand_dims(next_q_rows, axis=1) * stride_dom + tl.expand_dims(ptrs_21, axis=0),
                        do_view, mask=next_q_mask, other=cst_2)
                    tlx.async_load_commit_group([next_q_token, next_do_token])
                    qk_trans_124 = q_native_t if SINGLE_KV_TILE or CACHE_DELTA or NATIVE_Q_SCORE else tl.trans(q_122)
                    qk_trans_125 = _dot(k_87, qk_trans_124, cst_5)
                    valid_mask_trans_126 = tl.expand_dims(offs_m_106, axis=0)
                    valid_mask_trans_127 = (valid_mask_trans_126 < seq_len_q) & (
                        (start_n + offs_r)[:, None] < seq_len_kv)
                    qk_trans_128 = qk_trans_125 * scaled_alpha_23
                    m_129 = offs_m_25 + seq_start_q_11
                    m_130 = start_m + m_129
                    if CACHE_DELTA and PIPELINE_QDO:
                        m_131 = cached_lse
                    elif (EARLY_COARSE_STATS and KV_SPLITS > 1 or NATIVE_Q_SCORE
                          or SINGLE_KV_TILE and KV_SPLITS == 1) and not CACHE_DELTA and PIPELINE_QDO:
                        m_131 = early_lse
                    else:
                        m_131 = tl.load(M + m_130, mask=mask_m_108)
                    pT = tl.expand_dims(m_131, axis=0)
                    pT_133 = qk_trans_128 - pT
                    pT_134 = tl.math.exp2(pT_133)
                    pT_136 = tl.where(valid_mask_trans_127, pT_134, cst_5)
                    if CACHE_DELTA:
                        prefetched_delta = current_cached_delta
                    else:
                        if (EARLY_COARSE_STATS and KV_SPLITS > 1 or NATIVE_Q_SCORE
                                or SINGLE_KV_TILE and KV_SPLITS == 1) and not CACHE_DELTA and PIPELINE_QDO:
                            prefetched_delta = early_delta
                        else:
                            prefetched_delta = _backward_delta(Delta, OUT, m_130, mask_m_108, do)
                    dk_144 = _dot(pT_136.to(tl.bfloat16), do, dk_103)
                    dact_qk_trans_145 = do_native_t if SINGLE_KV_TILE or CACHE_DELTA else tl.trans(do)
                    dact_qk_trans_146 = _dot(k_87, dact_qk_trans_145, cst_5)
                    Di = prefetched_delta
                    dqk_trans = tl.expand_dims(Di, axis=0)
                    dqk_trans_148 = dact_qk_trans_146 - dqk_trans
                    dqk_trans_149 = pT_136 * dqk_trans_148
                    dqk_trans_149 = dqk_trans_149.to(tl.bfloat16)
                    prefetch_dq_row = seq_start_q_11 + offs_m_107
                    prefetch_dq_ptrs = tl.expand_dims(prefetch_dq_row, axis=1) * stride_dqm + tl.expand_dims(
                        ptrs_22, axis=0)
                    prefetch_dq_mask = tl.expand_dims(mask_m_109, axis=1)
                    prefetched_dq = tl.load(DQ + prefetch_dq_ptrs, mask=prefetch_dq_mask & (start_n != 0), other=cst_0)
                    dk_attn = _dot(dqk_trans_149, q_122, cst_4)
                    dk_153 = dk_attn * scaled_alpha
                    dk_154 = dk_144 + dk_153
                    dq_trans_156 = tl.trans(_dot(tl.trans(dqk_trans_149), k_87, cst_0))
                    dq_trans_158 = dq_trans_156 * scaled_alpha
                    dq = tl.trans(dq_trans_158)
                    dq_ptrs_159 = seq_start_q_11 + offs_m_107
                    dq_ptrs_161 = tl.expand_dims(dq_ptrs_159, axis=1)
                    dq_ptrs_163 = dq_ptrs_161 * stride_dqm
                    dq_ptrs_166 = tl.expand_dims(ptrs_22, axis=0)
                    dq_ptrs_168 = dq_ptrs_163 + dq_ptrs_166
                    cur = tl.expand_dims(mask_m_109, axis=1)
                    tl.store(dq_output + dq_ptrs_168, prefetched_dq + dq, mask=cur)
                    dk_103 = dk_154
                    if CACHE_DELTA:
                        tlx.amd_iglp_opt(2)
                # Drain the final masked prefetch before leaving this sequence.
                tlx.async_load_wait_group(0)
                offs_n_93 = start_n + offs_r
                dk_ptrs_94 = tl.expand_dims(offs_r_72, axis=1)
                dk_ptrs_96 = dk_ptrs_94 * stride_dkn
                dk_ptrs_99 = tl.expand_dims(ptrs_20, axis=0)
                dk_ptrs_101 = dk_ptrs_96 + dk_ptrs_99
                mask_n_102 = offs_n_93 < seq_len_kv
                var_5 = tl.expand_dims(mask_n_102, axis=1)
                tl.store(DK + dk_ptrs_101, dk_103, mask=var_5)
            else:
                kv_offset = (seq_start_kv_6 + last_block_start_n_18)  # triton_bw_cross_attention.py:3611
                offs_r_31 = tl.arange(0, 128)  # triton_bw_cross_attention.py:1581
                offs_r_34 = kv_offset + offs_r  # triton_bw_cross_attention.py:1581
                offs_r_35 = kv_offset + offs_r_19  # triton_bw_cross_attention.py:1581
                ptrs_38 = tl.expand_dims(offs_r_35, axis=1)  # triton_bw_cross_attention.py:1584
                ptrs_40 = ptrs_38 * stride_kn  # triton_bw_cross_attention.py:1584
                ptrs_43 = tl.expand_dims(ptrs_21, axis=0)  # triton_bw_cross_attention.py:1584
                ptrs_45 = ptrs_40 + ptrs_43  # triton_bw_cross_attention.py:1584
                mask_46 = offs_r_35 < seq_end_kv_9  # triton_bw_cross_attention.py:1588
                mask_47 = tl.expand_dims(mask_46, axis=1)  # triton_bw_cross_attention.py:1588
                k_48 = tl.load(K + ptrs_45, mask=mask_47, other=cst_1)  # triton_bw_cross_attention.py:1589
                offs_n_51 = (last_block_start_n_18 + offs_r_31)  # triton_bw_cross_attention.py:3657
                offs_n_52 = (last_block_start_n_18 + offs_r)  # triton_bw_cross_attention.py:3657
                valid_mask_trans_53 = tl.expand_dims(offs_n_51, axis=1)  # triton_hstu_cross_attention.py:793
                valid_mask_trans_55 = (valid_mask_trans_53 < seq_len_kv)  # triton_hstu_cross_attention.py:793
                dq_trans_57 = tl.trans(k_48)  # triton_bw_cross_attention.py:3744
                dk_69 = cst_4
                for start_m in range(0, seq_len_q, 32):
                    offs_m_72 = start_m + offs_m_25  # triton_bw_cross_attention.py:3667
                    offs_m_73 = start_m + offs_m_24  # triton_bw_cross_attention.py:3667
                    mask_m_74 = offs_m_72 < seq_len_q  # triton_bw_cross_attention.py:3668
                    mask_m_75 = offs_m_73 < seq_len_q  # triton_bw_cross_attention.py:3668
                    q_offset = seq_start_q_11 + start_m  # triton_bw_cross_attention.py:3669
                    offs_r_77 = q_offset + offs_m  # triton_bw_cross_attention.py:1581
                    ptrs_79 = tl.expand_dims(offs_r_77, axis=1)  # triton_bw_cross_attention.py:1584
                    ptrs_81 = ptrs_79 * stride_qm  # triton_bw_cross_attention.py:1584
                    ptrs_85 = ptrs_81 + ptrs_43  # triton_bw_cross_attention.py:1584
                    mask_86 = offs_r_77 < seq_end_q_13  # triton_bw_cross_attention.py:1588
                    mask_87 = tl.expand_dims(mask_86, axis=1)  # triton_bw_cross_attention.py:1588
                    q_88 = tl.load(Q + ptrs_85, mask=mask_87, other=cst_2)  # triton_bw_cross_attention.py:1589
                    qk_trans_90 = tl.trans(q_88)  # triton_bw_cross_attention.py:3683
                    qk_trans_91 = _dot(k_48, qk_trans_90, cst_5)  # triton_bw_cross_attention.py:3681
                    valid_mask_trans_92 = tl.expand_dims(offs_m_72, axis=0)  # triton_hstu_cross_attention.py:793
                    valid_mask_trans_93 = (valid_mask_trans_92 < seq_len_q)  # triton_hstu_cross_attention.py:793
                    valid_mask_trans_95 = (valid_mask_trans_93 & valid_mask_trans_55
                                           )  # triton_hstu_cross_attention.py:793
                    qk_trans_96 = (qk_trans_91 * scaled_alpha_23)  # triton_attention_utils.py:244
                    m_97 = offs_m_25 + seq_start_q_11  # triton_attention_utils.py:245
                    m_98 = start_m + m_97  # triton_attention_utils.py:245
                    m_99 = tl.load(M + m_98, mask=mask_m_74)  # triton_attention_utils.py:245
                    pT = tl.expand_dims(m_99, axis=0)  # triton_attention_utils.py:246
                    pT_101 = qk_trans_96 - pT  # triton_attention_utils.py:246
                    pT_102 = tl.math.exp2(pT_101)  # triton_attention_utils.py:246
                    pT_103 = tl.where(valid_mask_trans_95, pT_102, cst_5)  # triton_attention_utils.py:247
                    ptrs_105 = ptrs_79 * stride_dom  # triton_bw_cross_attention.py:1584
                    ptrs_108 = ptrs_105 + ptrs_43  # triton_bw_cross_attention.py:1584
                    do = tl.load(DO + ptrs_108, mask=mask_87, other=cst_2)  # triton_bw_cross_attention.py:1589
                    dk_111 = _dot(pT_103.to(tl.bfloat16), do, dk_69)  # triton_bw_cross_attention.py:3724
                    dact_qk_trans_112 = tl.trans(do)  # triton_bw_cross_attention.py:3729
                    dact_qk_trans_113 = _dot(k_48, dact_qk_trans_112, cst_5)  # triton_bw_cross_attention.py:3729
                    Di = _backward_delta(Delta, OUT, m_98, mask_m_74, do)  # triton_hstu_cross_attention.py:762
                    dqk_trans = tl.expand_dims(Di, axis=0)  # triton_hstu_cross_attention.py:763
                    dqk_trans_115 = (dact_qk_trans_113 - dqk_trans)  # triton_hstu_cross_attention.py:763
                    dqk_trans_116 = (pT_103 * dqk_trans_115)  # triton_hstu_cross_attention.py:763
                    dqk_trans_116 = dqk_trans_116.to(tl.bfloat16)
                    dk_attn = _dot(dqk_trans_116, q_88, cst_4)  # triton_bw_cross_attention.py:3740
                    dk_120 = dk_attn * scaled_alpha  # triton_bw_cross_attention.py:3741
                    dk_121 = dk_111 + dk_120  # triton_bw_cross_attention.py:3741
                    if PIPELINE_QDO:
                        dq_trans_123 = tl.trans(_dot(tl.trans(dqk_trans_116), k_48, cst_0))
                    else:
                        dq_trans_123 = _dot(dq_trans_57, dqk_trans_116, cst_5)
                    dq_trans_125 = (dq_trans_123 * scaled_alpha)  # triton_bw_cross_attention.py:3745
                    dq = tl.trans(dq_trans_125)  # triton_bw_cross_attention.py:3746
                    dq_ptrs_126 = (seq_start_q_11 + offs_m_73)  # triton_bw_cross_attention.py:3757
                    dq_ptrs_128 = tl.expand_dims(dq_ptrs_126, axis=1)  # triton_bw_cross_attention.py:3756
                    dq_ptrs_130 = (dq_ptrs_128 * stride_dqm)  # triton_bw_cross_attention.py:3756
                    dq_ptrs_133 = tl.expand_dims(ptrs_22, axis=0)  # triton_bw_cross_attention.py:3756
                    dq_ptrs_135 = (dq_ptrs_130 + dq_ptrs_133)  # triton_bw_cross_attention.py:3756
                    cur = tl.expand_dims(mask_m_75, axis=1)  # triton_bw_cross_attention.py:3760
                    if DIRECT_FIRST_DQ_STORE and last_block_start_n_18 == 0:
                        tl.store(dq_output + dq_ptrs_135, dq, mask=cur)
                    elif ATOMIC_DQ:
                        tl.atomic_add(
                            DQ + dq_ptrs_135,
                            dq.to(DQ.dtype.element_ty),
                            mask=cur,
                            sem="relaxed",
                        )
                    else:
                        cur_137 = tl.load(DQ + dq_ptrs_135, mask=cur, other=cst_0)  # triton_bw_cross_attention.py:3760
                        var_5 = cur_137 + dq  # triton_bw_cross_attention.py:3763
                        tl.store(dq_output + dq_ptrs_135, var_5, mask=cur)
                    dk_69 = dk_121
                dk_ptrs_60 = tl.expand_dims(offs_r_34, axis=1)  # triton_bw_cross_attention.py:3814
                dk_ptrs_62 = dk_ptrs_60 * stride_dkn  # triton_bw_cross_attention.py:3814
                dk_ptrs_65 = tl.expand_dims(ptrs_20, axis=0)  # triton_bw_cross_attention.py:3814
                dk_ptrs_67 = dk_ptrs_62 + dk_ptrs_65  # triton_bw_cross_attention.py:3814
                mask_n_68 = offs_n_52 < seq_len_kv  # triton_bw_cross_attention.py:3819
                var_1 = tl.expand_dims(mask_n_68, axis=1)  # triton_bw_cross_attention.py:3828
                tl.store(DK + dk_ptrs_67, dk_69, mask=var_1)  # triton_bw_cross_attention.py:3828


@triton.jit
def _tlx_gfx950_cross_attn_v3_ttgir_bwd(  # noqa: C901
    Q,
    K,
    V,
    DO,
    total_seq_len_q,
    total_seq_len_kv,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    scaled_alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    num_targets,
    Z,
    AUTOTUNE_Z,
    AUTOTUNE_MAX_Q_LEN,
    AUTOTUNE_MAX_SEQ_LEN,
    DIRECT_FIRST_DQ_STORE: tl.constexpr,
    ATOMIC_DQ: tl.constexpr,
    PREFETCH_DQ: tl.constexpr,
    STAGE_QDO: tl.constexpr,
    PIPELINE_QDO: tl.constexpr,
    DQ_FINAL=None,
    OUT=None,
    KV_SPLITS: tl.constexpr = 1,
    CACHE_DELTA: tl.constexpr = False,
    EARLY_TAIL_QDO: tl.constexpr = False,
    CACHE_K: tl.constexpr = False,
    EARLY_COARSE_STATS: tl.constexpr = False,
    SINGLE_KV_TILE: tl.constexpr = False,
    PACKED_FIXED_Q: tl.constexpr = False,
    NATIVE_Q_SCORE: tl.constexpr = False,
    FIXED_KV: tl.constexpr = 0,
):

    args = (
        Q,
        K,
        V,
        DO,
        total_seq_len_q,
        total_seq_len_kv,
        seq_offsets,
        seq_offsets_q,
        DQ,
        DK,
        DV,
        stride_qm,
        stride_qh,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_dom,
        stride_doh,
        stride_dqm,
        stride_dqh,
        stride_dkn,
        stride_dkh,
        stride_dvn,
        stride_dvh,
        scaled_alpha,
        max_seq_len,
        attn_scale,
        M,
        Delta,
        num_targets,
        Z,
        AUTOTUNE_Z,
        AUTOTUNE_MAX_Q_LEN,
        AUTOTUNE_MAX_SEQ_LEN,
    )
    if PACKED_FIXED_Q:
        _tlx_gfx950_cross_attn_v3_ttgir_bwd_impl(*args, DIRECT_FIRST_DQ_STORE=DIRECT_FIRST_DQ_STORE,
                                                 ATOMIC_DQ=ATOMIC_DQ, PREFETCH_DQ=PREFETCH_DQ, STAGE_QDO=STAGE_QDO,
                                                 PIPELINE_QDO=PIPELINE_QDO, DQ_FINAL=DQ_FINAL, OUT=OUT,
                                                 KV_SPLITS=KV_SPLITS, CACHE_DELTA=CACHE_DELTA, FIXED_Q=True,
                                                 EARLY_TAIL_QDO=EARLY_TAIL_QDO, CACHE_K=CACHE_K,
                                                 EARLY_COARSE_STATS=EARLY_COARSE_STATS, SINGLE_KV_TILE=SINGLE_KV_TILE,
                                                 PACKED_FIXED_Q=True, NATIVE_Q_SCORE=NATIVE_Q_SCORE, FIXED_KV=FIXED_KV)
    elif PIPELINE_QDO:
        seq = tl.program_id(0)
        q_start = tl.load(seq_offsets_q + seq)
        q_end = tl.load(seq_offsets_q + seq + 1)
        # Check each sequence before selecting the constant query bound.
        if q_end - q_start == 256:
            _tlx_gfx950_cross_attn_v3_ttgir_bwd_impl(
                *args, DIRECT_FIRST_DQ_STORE=DIRECT_FIRST_DQ_STORE, ATOMIC_DQ=ATOMIC_DQ, PREFETCH_DQ=PREFETCH_DQ,
                STAGE_QDO=STAGE_QDO, PIPELINE_QDO=PIPELINE_QDO, DQ_FINAL=DQ_FINAL, OUT=OUT, KV_SPLITS=KV_SPLITS,
                CACHE_DELTA=CACHE_DELTA, FIXED_Q=True, EARLY_TAIL_QDO=EARLY_TAIL_QDO, CACHE_K=CACHE_K,
                EARLY_COARSE_STATS=EARLY_COARSE_STATS, SINGLE_KV_TILE=SINGLE_KV_TILE)
        else:
            _tlx_gfx950_cross_attn_v3_ttgir_bwd_impl(
                *args, DIRECT_FIRST_DQ_STORE=DIRECT_FIRST_DQ_STORE, ATOMIC_DQ=ATOMIC_DQ, PREFETCH_DQ=PREFETCH_DQ,
                STAGE_QDO=STAGE_QDO, PIPELINE_QDO=PIPELINE_QDO, DQ_FINAL=DQ_FINAL, OUT=OUT, KV_SPLITS=KV_SPLITS,
                CACHE_DELTA=CACHE_DELTA, FIXED_Q=False, EARLY_TAIL_QDO=EARLY_TAIL_QDO, CACHE_K=CACHE_K,
                EARLY_COARSE_STATS=EARLY_COARSE_STATS, SINGLE_KV_TILE=SINGLE_KV_TILE)
    else:
        _tlx_gfx950_cross_attn_v3_ttgir_bwd_impl(*args, DIRECT_FIRST_DQ_STORE=DIRECT_FIRST_DQ_STORE,
                                                 ATOMIC_DQ=ATOMIC_DQ, PREFETCH_DQ=PREFETCH_DQ, STAGE_QDO=STAGE_QDO,
                                                 PIPELINE_QDO=PIPELINE_QDO, DQ_FINAL=DQ_FINAL, OUT=OUT,
                                                 KV_SPLITS=KV_SPLITS, CACHE_DELTA=CACHE_DELTA, FIXED_Q=False,
                                                 EARLY_TAIL_QDO=EARLY_TAIL_QDO, CACHE_K=CACHE_K,
                                                 EARLY_COARSE_STATS=EARLY_COARSE_STATS, SINGLE_KV_TILE=SINGLE_KV_TILE)
