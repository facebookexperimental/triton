"""AMD CDNA4 Flash Attention forward with a rotated 4-cluster pipeline."""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.core import _aggregate as aggregate
from triton.language.extra.libdevice import fast_dividef

CLUSTER_BUF_DEPTH = 2
CLUSTER_PIPELINE_STAGES = tl.constexpr(4)
LAZY_RESCALE_THRESHOLD = tl.constexpr(8.0)
DIAGONAL_LAZY_RESCALE_THRESHOLD = tl.constexpr(4.0)
DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16 = tl.constexpr(8.0)
CDNA_WAVE_SIZE = tl.constexpr(64)
CDNA_MFMA_ROWS_PER_WAVE = tl.constexpr(32)

_CLUSTER_PIPELINE_STAGE_COUNT = 4
_CLUSTER_AUTOTUNE_NUM_WARPS = tuple(range(1, 9))
_CLUSTER_VGPR_ONLY_LLVM_FN_ATTRS = (("amdgpu-agpr-alloc", "0,0"), )
_CLUSTER_SHORT_N512_LLVM_FN_ATTRS = (
    ("amdgpu-no-dispatch-id", ""),
    *_CLUSTER_VGPR_ONLY_LLVM_FN_ATTRS,
)
_CLUSTER_SHORT_N1024_LLVM_FN_ATTRS = _CLUSTER_VGPR_ONLY_LLVM_FN_ATTRS


def _cluster_short_load_configs():
    return [
        triton.Config({"USE_DIRECT_LOAD": use_direct_load}, num_warps=num_warps, num_stages=3)
        for num_warps in _CLUSTER_AUTOTUNE_NUM_WARPS
        for use_direct_load in (False, True)
    ]


def _cluster_meta_arg(name, named_args, kwargs):
    return kwargs[name] if name in kwargs else named_args[name]


def _cluster_has_short_range(n_ctx, block_m, block_n, is_causal):
    num_blocks_total = (n_ctx + block_n - 1) // block_n
    is_modulo_mn = n_ctx % block_n == 0 and n_ctx % block_m == 0
    num_m_blocks = (n_ctx + block_m - 1) // block_m if is_causal else 1

    for pid_m in range(num_m_blocks):
        if is_causal:
            causal_end = ((pid_m + 1) * block_m + block_n - 1) // block_n
            num_blocks = min(num_blocks_total, causal_end)
            masked_blocks = block_m // block_n + (not is_modulo_mn)
        else:
            num_blocks = num_blocks_total
            masked_blocks = 1 if n_ctx % block_n != 0 else 0

        masked_blocks = min(masked_blocks, num_blocks)
        num_full = num_blocks - masked_blocks
        if 0 < num_blocks <= _CLUSTER_PIPELINE_STAGE_COUNT:
            return True
        if (num_blocks > _CLUSTER_PIPELINE_STAGE_COUNT and (num_blocks - num_full) < _CLUSTER_PIPELINE_STAGE_COUNT
                and num_full != num_blocks):
            continue

        masked_start = num_full if num_full > _CLUSTER_PIPELINE_STAGE_COUNT else 0
        remaining_blocks = num_blocks - masked_start
        if 0 < remaining_blocks <= _CLUSTER_PIPELINE_STAGE_COUNT:
            return True
    return False


def _prune_cluster_short_load_configs(configs, named_args, **kwargs):
    """Tune LDS versus direct loads only when a short range is reachable."""
    block_m = _cluster_meta_arg("BLOCK_M", named_args, kwargs)
    block_n = _cluster_meta_arg("BLOCK_N", named_args, kwargs)
    n_ctx = _cluster_meta_arg("N_CTX", named_args, kwargs)
    is_causal = _cluster_meta_arg("IS_CAUSAL", named_args, kwargs)
    num_warps = min(8, max(1, block_m // 32))

    candidates = [config for config in configs if config.num_warps == num_warps]
    if not _cluster_has_short_range(n_ctx, block_m, block_n, is_causal):
        candidates = [config for config in candidates if not config.kwargs["USE_DIRECT_LOAD"]]
    return candidates


_CLUSTER_AUTOTUNE_KEY = ["Z", "H", "N_CTX", "HEAD_DIM", "BLOCK_M", "BLOCK_N", "IS_CAUSAL"]
_CLUSTER_PERSISTENT_AUTOTUNE_KEY = [*_CLUSTER_AUTOTUNE_KEY, "NUM_SMS", "NUM_XCDS"]


@triton.jit
def _assume_strides(
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_ok: tl.constexpr,
):
    tl.assume(stride_qz >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_qm > 0)
    tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_kk >= 0)
    tl.assume(stride_vz >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vk >= 0)
    tl.assume(stride_oz >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_ok >= 0)


@triton.jit
def _split_cols(x):
    """Split a matrix into logical column halves without changing values.

    The reshape must retain Triton's default order-preserving contract.  In
    particular, ``can_reorder=True`` would permit a different register
    interpretation and no longer make this operation the inverse of
    ``_concat_cols``.
    """
    tl.static_assert(x.shape[1] % 2 == 0)
    x0, x1 = tl.split(x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1))
    return x0, x1


@triton.jit
def _concat_cols(x0, x1):
    """Reassemble adjacent logical column halves without reordering them."""
    tl.static_assert(x0.shape[0] == x1.shape[0])
    tl.static_assert(x0.shape[1] == x1.shape[1])
    x = tl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] + x1.shape[1]])
    return x


@triton.jit
def _sum_rows_chain4(x, ROTATE_FINAL: tl.constexpr):
    """Reduce four zero-copy column slices through one dependency chain."""
    x_01, x_23 = _split_cols(x)
    x_0, x_1 = _split_cols(x_01)
    x_2, x_3 = _split_cols(x_23)
    partial = x_0 + x_1
    if ROTATE_FINAL:
        partial = partial + x_3
        partial = partial + x_2
    else:
        partial = partial + x_2
        partial = partial + x_3
    return tl.sum(partial, 1)


@triton.jit
def _sum_combine(a, b):
    return a + b


@aggregate
class LazyProbabilityState:
    p_0123: tl.tensor
    p_4: tl.tensor
    qk_5: tl.tensor
    qk_6: tl.tensor
    qk_7: tl.tensor


@aggregate
class SoftmaxState:
    acc: tl.tensor
    l_i: tl.tensor
    m_i: tl.tensor

    @triton.jit
    def create(BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
        return SoftmaxState(
            tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32),
            tl.full([BLOCK_M], 1.0, dtype=tl.float32),
            tl.full([BLOCK_M], float("-inf"), dtype=tl.float32),
        )

    @triton.jit
    def _nan_max_combine(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def _row_max(qk):
        return tl.reduce(qk, 1, SoftmaxState._nan_max_combine)

    @triton.jit
    def vec1(
        self,
        qk,
        start_n,
        offs_m,
        offs_n,
        N_CTX: tl.constexpr,
        QK_SCALE: tl.constexpr,
        DIAG_OFFSET: tl.constexpr,
        MASK_STEPS: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        if MASK_STEPS:
            qk_sm = qk * QK_SCALE
            kn = start_n + offs_n
            if IS_CAUSAL:
                qk_sm = tl.where(offs_m[:, None] + DIAG_OFFSET >= kn[None, :], qk_sm, float("-inf"))
            qk_sm = tl.where(kn[None, :] < N_CTX, qk_sm, float("-inf"))
            m_ij = SoftmaxState._row_max(qk_sm)
            m_new = tl.maximum(self.m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp2(qk_sm - m_new[:, None])
            alpha = tl.math.exp2(self.m_i - m_new)
        else:
            qk_sm = qk * QK_SCALE
            m_ij = SoftmaxState._row_max(qk_sm)
            m_new = tl.maximum(self.m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp2(qk_sm - m_new[:, None])
            alpha = tl.math.exp2(self.m_i - m_new)
        return SoftmaxState(self.acc, self.l_i, m_new), p, alpha

    @triton.jit
    def vec2(self, p, alpha, out_dtype: tl.constexpr):
        l_ij = tl.sum(p, 1)
        acc = self.acc * alpha[:, None]
        l_i = self.l_i * alpha + l_ij
        p_cast = p.to(out_dtype)
        return SoftmaxState(acc, l_i, self.m_i), p_cast

    @triton.jit
    def vec2_split_acc(self, acc_sub1, p, alpha, out_dtype: tl.constexpr):
        """Apply VEC2 while keeping a D128 accumulator as two D64 tiles."""
        tl.static_assert(self.acc.shape[1] == 64)
        tl.static_assert(acc_sub1.shape[0] == self.acc.shape[0])
        tl.static_assert(acc_sub1.shape[1] == self.acc.shape[1])
        acc_sub0 = self.acc * alpha[:, None]
        acc_sub1 = acc_sub1 * alpha[:, None]
        l_ij = tl.sum(p, 1)
        l_i = self.l_i * alpha + l_ij
        p_cast = p.to(out_dtype)
        return SoftmaxState(acc_sub0, l_i, self.m_i), acc_sub1, p_cast

    @triton.jit
    def vec1_lazy(self, qk, QK_SCALE: tl.constexpr):
        """Form probabilities in a lazily advanced base-2 softmax frame."""
        state, score_delta, advance = self.vec1_lazy_max(qk, QK_SCALE)
        p = state.vec1_lazy_exp(qk, QK_SCALE)
        return state, p, score_delta, advance

    @triton.jit
    def vec1_lazy_split(self, qk, QK_SCALE: tl.constexpr):
        """Form five eighths of P and defer three score fragments."""
        state, score_delta, advance = self.vec1_lazy_max(qk, QK_SCALE)
        p_state = state.vec1_lazy_exp_split(qk, QK_SCALE)
        return state, p_state, score_delta, advance

    @triton.jit
    def vec1_lazy_split4(self, qk, QK_SCALE: tl.constexpr):
        """Form four eighths of P and carry the remaining score fragments."""
        tl.static_assert(qk.shape[1] == 64)
        state, score_delta, advance = self.vec1_lazy_max(qk, QK_SCALE)
        qk_lo, qk_hi = _split_cols(qk)
        qk_a, qk_b = _split_cols(qk_lo)
        qk_c, qk_d = _split_cols(qk_hi)
        qk_0, qk_1 = _split_cols(qk_a)
        qk_2, qk_3 = _split_cols(qk_b)
        qk_4, qk_5 = _split_cols(qk_c)
        qk_6, qk_7 = _split_cols(qk_d)
        m_i = tlx.release_layout(state.m_i)[:, None]
        p_0 = tl.math.exp2(qk_0 * QK_SCALE - m_i)
        p_1 = tl.math.exp2(qk_1 * QK_SCALE - m_i)
        p_2 = tl.math.exp2(qk_2 * QK_SCALE - m_i)
        p_3 = tl.math.exp2(qk_3 * QK_SCALE - m_i)
        p_01 = _concat_cols(p_0, p_1)
        p_23 = _concat_cols(p_2, p_3)
        return (
            state,
            LazyProbabilityState(
                _concat_cols(p_01, p_23),
                qk_4,
                qk_5,
                qk_6,
                qk_7,
            ),
            score_delta,
            advance,
        )

    @triton.jit
    def vec1_lazy_split_threshold(self, qk, QK_SCALE: tl.constexpr, THRESHOLD: tl.constexpr):
        """Split lazy VEC1 with the threshold used by the causal diagonal."""
        state, score_delta, advance = self.vec1_lazy_max_threshold(qk, QK_SCALE, THRESHOLD)
        p_state = state.vec1_lazy_exp_split(qk, QK_SCALE)
        return state, p_state, score_delta, advance

    @triton.jit
    def vec1_lazy_max(self, qk, QK_SCALE: tl.constexpr):
        """Advance the lazy softmax frame while retaining the score tile."""
        m_ij = SoftmaxState._row_max(qk * QK_SCALE)
        score_delta = m_ij - self.m_i
        advance = score_delta > LAZY_RESCALE_THRESHOLD
        m_new = tl.where(advance, m_ij, self.m_i)
        return SoftmaxState(self.acc, self.l_i, m_new), score_delta, advance

    @triton.jit
    def vec1_lazy_max_threshold(self, qk, QK_SCALE: tl.constexpr, THRESHOLD: tl.constexpr):
        """Advance the lazy frame using a diagonal-specific threshold."""
        m_ij = SoftmaxState._row_max(qk * QK_SCALE)
        score_delta = m_ij - self.m_i
        advance = score_delta > THRESHOLD
        m_new = tl.where(advance, m_ij, self.m_i)
        return SoftmaxState(self.acc, self.l_i, m_new), score_delta, advance

    @triton.jit
    def vec1_lazy_exp(self, qk, QK_SCALE: tl.constexpr):
        return tl.math.exp2(qk * QK_SCALE - self.m_i[:, None])

    @triton.jit
    def vec1_lazy_exp_split(self, qk, QK_SCALE: tl.constexpr):
        """Compute P[0:5] as N8 fragments and carry QK[5:8]."""
        tl.static_assert(qk.shape[1] == 64)
        mma: tl.constexpr = tlx.amd_mfma_layout(
            version=4,
            instr_shape=[32, 32, 16],
            transposed=True,
            warps_per_cta=[tlx.num_warps(), 1],
        )
        qk = tlx.require_layout(qk, mma, pin=True)
        qk_lo, qk_hi = _split_cols(qk)
        qk_a, qk_b = _split_cols(qk_lo)
        qk_c, qk_d = _split_cols(qk_hi)
        qk_0, qk_1 = _split_cols(qk_a)
        qk_2, qk_3 = _split_cols(qk_b)
        qk_4, qk_5 = _split_cols(qk_c)
        qk_6, qk_7 = _split_cols(qk_d)
        # Each N8 fragment has the order-preserving linear layout inferred by
        # the split.  Release the row-state pin before broadcasting so the
        # elementwise probability work follows that fragment layout.
        m_i = tlx.release_layout(self.m_i)[:, None]
        p_0 = tl.math.exp2(qk_0 * QK_SCALE - m_i)
        p_1 = tl.math.exp2(qk_1 * QK_SCALE - m_i)
        p_2 = tl.math.exp2(qk_2 * QK_SCALE - m_i)
        p_3 = tl.math.exp2(qk_3 * QK_SCALE - m_i)
        p_4 = tl.math.exp2(qk_4 * QK_SCALE - m_i)
        p_01 = _concat_cols(p_0, p_1)
        p_23 = _concat_cols(p_2, p_3)
        return LazyProbabilityState(_concat_cols(p_01, p_23), p_4, qk_5, qk_6, qk_7)

    @triton.jit
    def vec2_lazy(self, p, out_dtype: tl.constexpr):
        l_i = self.l_i + tl.sum(p, 1)
        return SoftmaxState(self.acc, l_i, self.m_i), p.to(out_dtype)

    @triton.jit
    def vec2_lazy_split(
        self,
        p_state,
        QK_SCALE: tl.constexpr,
        out_dtype: tl.constexpr,
        FP16_CAST_ROWSUM: tl.constexpr,
        CHAIN_BF16_ROWSUM: tl.constexpr = 0,
    ):
        """Finish the deferred three eighths of P in the next DOT1 phase."""
        m_i = tlx.release_layout(self.m_i)[:, None]
        p_5 = tl.math.exp2(p_state.qk_5 * QK_SCALE - m_i)
        p_6 = tl.math.exp2(p_state.qk_6 * QK_SCALE - m_i)
        p_7 = tl.math.exp2(p_state.qk_7 * QK_SCALE - m_i)
        p_45 = _concat_cols(p_state.p_4, p_5)
        p_67 = _concat_cols(p_6, p_7)
        p = _concat_cols(p_state.p_0123, _concat_cols(p_45, p_67))
        p_cast = p.to(out_dtype)
        if FP16_CAST_ROWSUM and out_dtype == tl.float16:
            l_ij = tl.sum(p_cast, 1)
        elif CHAIN_BF16_ROWSUM:
            l_ij = _sum_rows_chain4(p, CHAIN_BF16_ROWSUM == 2)
        else:
            l_ij = tl.sum(p, 1)
        l_i = self.l_i + l_ij
        return SoftmaxState(self.acc, l_i, self.m_i), p_cast

    @triton.jit
    def vec2_lazy_split4(
        self,
        p_state,
        QK_SCALE: tl.constexpr,
        out_dtype: tl.constexpr,
        FP16_CAST_ROWSUM: tl.constexpr,
        CHAIN_BF16_ROWSUM: tl.constexpr = 0,
    ):
        """Finish four deferred score fragments from the BM128 prefix."""
        m_i = tlx.release_layout(self.m_i)[:, None]
        p_4 = tl.math.exp2(p_state.p_4 * QK_SCALE - m_i)
        p_5 = tl.math.exp2(p_state.qk_5 * QK_SCALE - m_i)
        p_6 = tl.math.exp2(p_state.qk_6 * QK_SCALE - m_i)
        p_7 = tl.math.exp2(p_state.qk_7 * QK_SCALE - m_i)
        p_45 = _concat_cols(p_4, p_5)
        p_67 = _concat_cols(p_6, p_7)
        p = _concat_cols(p_state.p_0123, _concat_cols(p_45, p_67))
        p_cast = p.to(out_dtype)
        if FP16_CAST_ROWSUM and out_dtype == tl.float16:
            l_ij = tl.sum(p_cast, 1)
        elif CHAIN_BF16_ROWSUM:
            l_ij = _sum_rows_chain4(p, CHAIN_BF16_ROWSUM == 2)
        else:
            l_ij = tl.sum(p, 1)
        l_i = self.l_i + l_ij
        return SoftmaxState(self.acc, l_i, self.m_i), p_cast

    @triton.jit
    def rescale_lazy(self, score_delta, advance):
        tl.static_assert(self.acc.shape[0] == 128 or self.acc.shape[0] == 256)
        tl.static_assert(self.acc.shape[1] == 128)
        tl.static_assert(self.l_i.shape[0] == self.acc.shape[0])
        tl.static_assert(score_delta.shape[0] == self.acc.shape[0])
        tl.static_assert(advance.shape[0] == self.acc.shape[0])
        tl.static_assert(tlx.num_warps() == self.acc.shape[0] // 32)
        acc_layout: tl.constexpr = tlx.amd_mfma_layout(
            version=4,
            instr_shape=[32, 32, 16],
            transposed=True,
            warps_per_cta=[tlx.num_warps(), 1],
        )
        row_layout: tl.constexpr = tlx.slice_layout(acc_layout, 1)
        advance = tlx.require_layout(advance, row_layout, pin=False)
        score_delta = tlx.require_layout(score_delta, row_layout, pin=False)
        l_i = tlx.require_layout(self.l_i, row_layout, pin=False)
        acc = tlx.require_layout(self.acc, acc_layout, pin=False)
        alpha = tl.math.exp2(-score_delta)
        alpha_2d = tlx.require_layout(alpha[:, None], acc_layout, pin=False)
        acc, l_i = tlx.warp_predicate(
            advance,
            (acc, l_i),
            _rescale_softmax_state,
            args=(alpha, alpha_2d),
        )
        return SoftmaxState(acc, l_i, self.m_i)


@triton.jit
def _rescale_softmax_state(acc, l_i, alpha, alpha_2d):
    """Apply a lane-local correction after layouts are fixed outside EXEC."""
    return acc * alpha_2d, l_i * alpha


@triton.jit
def _attn_dot_pv_vec1_lazy(
    state,
    p_dot,
    v_dot,
    qk,
    QK_SCALE: tl.constexpr,
    SPLIT_VEC1: tl.constexpr,
):
    """Weave lazy row-max work through four native K16 P-by-V steps."""
    tl.static_assert(state.acc.shape[0] == 128 or state.acc.shape[0] == 256)
    tl.static_assert(state.acc.shape[1] == 128)
    tl.static_assert(p_dot.shape[0] == state.acc.shape[0])
    tl.static_assert(p_dot.shape[1] == 64)
    tl.static_assert(v_dot.shape[0] == 64)
    tl.static_assert(v_dot.shape[1] == 128)
    tl.static_assert(qk.shape[0] == state.acc.shape[0])
    tl.static_assert(qk.shape[1] == 64)
    tl.static_assert(tlx.num_warps() == state.acc.shape[0] // 32)
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[tlx.num_warps(), 1],
    )
    p_layout: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    v_layout: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    p_dot = tlx.require_layout(p_dot, p_layout, pin=False)
    v_dot = tlx.require_layout(v_dot, v_layout, pin=False)
    acc = tlx.require_layout(state.acc, mma, pin=False)
    p0 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 0])
    p1 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 16])
    p2 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 32])
    p3 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 48])
    v0 = tlx.extract_slice(v_dot, [16, v_dot.shape[1]], [0, 0])
    v1 = tlx.extract_slice(v_dot, [16, v_dot.shape[1]], [16, 0])
    v2 = tlx.extract_slice(v_dot, [16, v_dot.shape[1]], [32, 0])
    v3 = tlx.extract_slice(v_dot, [16, v_dot.shape[1]], [48, 0])

    acc = tl.dot(p0, v0, acc)
    if state.acc.shape[0] == 128:
        acc = tl.dot(p1, v1, acc)
        max_state, score_delta, advance = state.vec1_lazy_max(qk, QK_SCALE)
    else:
        max_state, score_delta, advance = state.vec1_lazy_max(qk, QK_SCALE)
        acc = tl.dot(p1, v1, acc)
    acc = tl.dot(p2, v2, acc)
    acc = tl.dot(p3, v3, acc)
    state = SoftmaxState(acc, state.l_i, max_state.m_i)
    if SPLIT_VEC1:
        p = state.vec1_lazy_exp_split(qk, QK_SCALE)
    else:
        p = state.vec1_lazy_exp(qk, QK_SCALE)
    return state, p, score_delta, advance


@triton.jit
def _attn_dot_pv_mfma(acc, p_dot, v_dot):
    """Keep a fallback P-by-V dot in the woven accumulator's layout."""
    tl.static_assert(acc.shape[0] == 128 or acc.shape[0] == 256)
    tl.static_assert(acc.shape[1] == 128)
    tl.static_assert(p_dot.shape[0] == acc.shape[0])
    tl.static_assert(p_dot.shape[1] == 64)
    tl.static_assert(v_dot.shape[0] == 64)
    tl.static_assert(v_dot.shape[1] == 128)
    tl.static_assert(tlx.num_warps() == acc.shape[0] // 32)
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[tlx.num_warps(), 1],
    )
    p_layout: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    v_layout: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    p_dot = tlx.require_layout(p_dot, p_layout, pin=False)
    v_dot = tlx.require_layout(v_dot, v_layout, pin=False)
    acc = tlx.require_layout(acc, mma, pin=False)
    return tl.dot(p_dot, v_dot, acc)


@triton.jit
def _attn_dot_pv_mfma_split(acc, p_dot, v_dot):
    """Accumulate one D64 half of the BM256/BN64/D128 P-by-V dot."""
    tl.static_assert(acc.shape[0] == 256)
    tl.static_assert(acc.shape[1] == 64)
    tl.static_assert(p_dot.shape[0] == acc.shape[0])
    tl.static_assert(p_dot.shape[1] == 64)
    tl.static_assert(v_dot.shape[0] == 64)
    tl.static_assert(v_dot.shape[1] == acc.shape[1])
    tl.static_assert(tlx.num_warps() == 8)
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[tlx.num_warps(), 1],
    )
    p_layout: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    v_layout: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    p_dot = tlx.require_layout(p_dot, p_layout, pin=False)
    v_dot = tlx.require_layout(v_dot, v_layout, pin=False)
    acc = tlx.require_layout(acc, mma, pin=False)
    return tl.dot(p_dot, v_dot, acc)


@triton.jit
def _attn_dot_pv_mfma32(acc, p_dot, v_dot):
    """P-by-V fallback for the low N32 half of a causal diagonal."""
    tl.static_assert(acc.shape[0] == 128 or acc.shape[0] == 256)
    tl.static_assert(acc.shape[1] == 128)
    tl.static_assert(p_dot.shape[0] == acc.shape[0])
    tl.static_assert(p_dot.shape[1] == 32)
    tl.static_assert(v_dot.shape[0] == 32)
    tl.static_assert(v_dot.shape[1] == 128)
    tl.static_assert(tlx.num_warps() == acc.shape[0] // 32)
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[tlx.num_warps(), 1],
    )
    p_layout: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=2)
    v_layout: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=2)
    p_dot = tlx.require_layout(p_dot, p_layout, pin=False)
    v_dot = tlx.require_layout(v_dot, v_layout, pin=False)
    acc = tlx.require_layout(acc, mma, pin=False)
    return tl.dot(p_dot, v_dot, acc)


@triton.jit
def _attn_war_barrier(LIGHTWEIGHT: tl.constexpr):
    """Rendezvous before an LDS overwrite, with no visibility fence for WAR."""
    if LIGHTWEIGHT:
        tl.inline_asm_elementwise(
            "s_waitcnt lgkmcnt(0)\ns_barrier",
            "=s",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        tl.debug_barrier()


@triton.jit
def _attn_qk_war_barrier_relaxed(qk):
    """Anchor both N32 QK results before reusing their K slot."""
    tl.static_assert(qk.shape[0] == 128 and qk.shape[1] == 64)
    tl.static_assert(tlx.num_warps() == 4)
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[tlx.num_warps(), 1],
    )
    qk = tlx.require_layout(qk, mma, pin=False)
    # One zero-copy N8 view from each N32 result ties the rendezvous to the
    # final MFMA that consumes the old K slot in every wave.
    qk0 = tlx.extract_slice(qk, [qk.shape[0], 8], [0, 0])
    qk1 = tlx.extract_slice(qk, [qk.shape[0], 8], [0, qk.shape[1] // 2])
    tl.inline_asm_elementwise(
        "s_barrier",
        "=s,=s,=s,=s,v,v,v,v,v,v,v,v",
        [qk0, qk1],
        dtype=tl.int32,
        is_pure=False,
        pack=4,
    )


@triton.jit
def _attn_pv_war_barrier_relaxed(acc):
    """Anchor every D32 P-by-V result before reusing its V slot."""
    tl.static_assert(acc.shape[0] == 128 and acc.shape[1] == 128)
    tl.static_assert(tlx.num_warps() == 4)
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[tlx.num_warps(), 1],
    )
    acc = tlx.require_layout(acc, mma, pin=False)
    # D128 P-by-V has four D32 accumulator groups. One zero-copy D8 view from
    # each group makes the rendezvous depend on every final MFMA.
    acc_0 = tlx.extract_slice(acc, [acc.shape[0], 8], [0, 0])
    acc_1 = tlx.extract_slice(acc, [acc.shape[0], 8], [0, acc.shape[1] // 4])
    acc_2 = tlx.extract_slice(acc, [acc.shape[0], 8], [0, acc.shape[1] // 2])
    acc_3 = tlx.extract_slice(acc, [acc.shape[0], 8], [0, 3 * acc.shape[1] // 4])
    tl.inline_asm_elementwise(
        "s_barrier",
        "=s,=s,=s,=s," + ",".join(["v"] * 16),
        [acc_0, acc_1, acc_2, acc_3],
        dtype=tl.int32,
        is_pure=False,
        pack=4,
    )


@triton.jit
def _attn_dot_qk_step8_vec2(
    state,
    p_state,
    q,
    kt_dot,
    QK_SCALE: tl.constexpr,
    FP16_CAST_ROWSUM: tl.constexpr,
):
    """Interleave the deferred lazy-softmax fragments with QK's K16 MFMAs.

    CDNA4 lowers a regular ``tl.dot`` as one opaque reduction.  The Gluon
    kernel instead issues the eight K=16 reductions separately and uses the
    otherwise idle VALU slots to finish VEC2 between reductions 5--7.  Keep
    the same schedule here, with the explicit CDNA4 operand layouts required
    by ``amd_scheduled_mfma``.
    """
    tl.static_assert(q.shape[1] == 128)
    tl.static_assert(kt_dot.shape[0] == 128)
    tl.static_assert(kt_dot.shape[1] == 64)
    tl.static_assert(q.shape[0] == 256)
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[tlx.num_warps(), 1],
    )
    q_layout: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    kt_layout: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    q = tlx.require_layout(q, q_layout, pin=False)
    kt_dot = tlx.require_layout(kt_dot, kt_layout, pin=False)
    qk = tlx.zeros((q.shape[0], kt_dot.shape[1]), tl.float32, layout=mma)

    q0 = tlx.extract_slice(q, [q.shape[0], 16], [0, 0])
    q1 = tlx.extract_slice(q, [q.shape[0], 16], [0, 16])
    q2 = tlx.extract_slice(q, [q.shape[0], 16], [0, 32])
    q3 = tlx.extract_slice(q, [q.shape[0], 16], [0, 48])
    q4 = tlx.extract_slice(q, [q.shape[0], 16], [0, 64])
    q5 = tlx.extract_slice(q, [q.shape[0], 16], [0, 80])
    q6 = tlx.extract_slice(q, [q.shape[0], 16], [0, 96])
    q7 = tlx.extract_slice(q, [q.shape[0], 16], [0, 112])
    k0 = tlx.extract_slice(kt_dot, [16, kt_dot.shape[1]], [0, 0])
    k1 = tlx.extract_slice(kt_dot, [16, kt_dot.shape[1]], [16, 0])
    k2 = tlx.extract_slice(kt_dot, [16, kt_dot.shape[1]], [32, 0])
    k3 = tlx.extract_slice(kt_dot, [16, kt_dot.shape[1]], [48, 0])
    k4 = tlx.extract_slice(kt_dot, [16, kt_dot.shape[1]], [64, 0])
    k5 = tlx.extract_slice(kt_dot, [16, kt_dot.shape[1]], [80, 0])
    k6 = tlx.extract_slice(kt_dot, [16, kt_dot.shape[1]], [96, 0])
    k7 = tlx.extract_slice(kt_dot, [16, kt_dot.shape[1]], [112, 0])

    # The source-scheduled MFMA primitive is currently BF16-only in TLX;
    # explicit K16 ``tl.dot`` calls preserve the interleave for FP16 while
    # retaining the native FP16 dot lowering.
    qk = tl.dot(q0, k0, qk)
    qk = tl.dot(q1, k1, qk)
    qk = tl.dot(q2, k2, qk)
    qk = tl.dot(q3, k3, qk)
    qk = tl.dot(q4, k4, qk)

    m_i = tlx.release_layout(state.m_i)[:, None]
    p5 = tl.math.exp2(p_state.qk_5 * QK_SCALE - m_i)
    qk = tl.dot(q5, k5, qk)
    p6 = tl.math.exp2(p_state.qk_6 * QK_SCALE - m_i)
    qk = tl.dot(q6, k6, qk)
    p7 = tl.math.exp2(p_state.qk_7 * QK_SCALE - m_i)

    p45 = _concat_cols(p_state.p_4, p5)
    p67 = _concat_cols(p6, p7)
    p = _concat_cols(p_state.p_0123, _concat_cols(p45, p67))
    p_cast = p.to(q.dtype)
    if FP16_CAST_ROWSUM and q.dtype == tl.float16:
        # Gluon keeps the reassembled probability tile in the MFMA layout, so
        # its FP16 row sum uses the same short local tree before the cross-lane
        # reduction.  Pin that source layout until the reduction is complete;
        # release only the rank-1 result at this helper boundary.
        # The split fragments intentionally keep their inferred, order-
        # preserving layouts.  Stop the reduction pin at this explicit
        # conversion boundary instead of propagating it back through the
        # logical split/concat chain.
        p_reduce = tlx.require_layout(tlx.release_layout(p_cast), mma, pin=True)
        tlx.assert_same_layout(p_reduce, mma)
        l_ij = tl.reduce(p_reduce, 1, _sum_combine)
        l_ij = tlx.release_layout(l_ij)
    else:
        l_ij = tl.sum(p, 1)
    state = SoftmaxState(state.acc, state.l_i + l_ij, state.m_i)

    qk = tl.dot(q7, k7, qk)
    # Gluon keeps the score tile in its MFMA layout across the DOT1 -> DOT2
    # handoff.  Pin the same contract here so the split/concat probability
    # fragments reassemble trivially in that layout instead of selecting an
    # unrelated linear layout and converting back for the FP16 row sum.
    qk = tlx.require_layout(qk, mma, pin=True)
    return state, p_cast, qk


@triton.jit
def _attn_inner_pipelined_lazy_step(
    state,
    p_c,
    q,
    kt_dot,
    k_ptrs,
    v_ptrs,
    offs_n,
    block_n,
    k_buf,
    v_buf,
    stride_kn,
    stride_vn,
    QK_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CUR_SLOT: tl.constexpr,
    NEXT_SLOT: tl.constexpr,
    SPLIT_VEC1: tl.constexpr,
    STEP_PV_VEC1: tl.constexpr,
    STEP_QK_VEC2: tl.constexpr,
    FP16_CAST_ROWSUM: tl.constexpr,
    CHAIN_BF16_ROWSUM: tl.constexpr,
):
    """Advance one aligned lazy-rescale tile with static LDS slots."""
    ack_n = (block_n + 3) * BLOCK_N
    acv_n = (block_n + 2) * BLOCK_N

    with tlx.warp_pipeline_stage("dot1", priority=0):
        if STEP_QK_VEC2:
            state, p_dot, qk = _attn_dot_qk_step8_vec2(state, p_c, q, kt_dot, QK_SCALE, FP16_CAST_ROWSUM)
        else:
            qk = tl.dot(q, kt_dot)
            if SPLIT_VEC1:
                state, p_dot = state.vec2_lazy_split(
                    p_c,
                    QK_SCALE,
                    q.dtype,
                    FP16_CAST_ROWSUM,
                    CHAIN_BF16_ROWSUM,
                )
            else:
                state, p_dot = state.vec2_lazy(p_c, q.dtype)

    tlx.async_load_wait_group(1)

    with tlx.warp_pipeline_stage("mem1", priority=1):
        v_dot = tlx.local_load(tlx.local_view(v_buf, CUR_SLOT), relaxed=True)
        tok_k = tlx.async_load(k_ptrs + ack_n * stride_kn, tlx.local_view(k_buf, NEXT_SLOT))
        tlx.async_load_commit_group([tok_k])

    with tlx.warp_pipeline_stage("dot2", priority=0):
        if STEP_PV_VEC1:
            state, p_c, delta_c, advance_c = _attn_dot_pv_vec1_lazy(
                state,
                p_dot,
                v_dot,
                qk,
                QK_SCALE,
                SPLIT_VEC1,
            )
        else:
            acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
            state = SoftmaxState(acc, state.l_i, state.m_i)
            if SPLIT_VEC1:
                state, p_c, delta_c, advance_c = state.vec1_lazy_split(qk, QK_SCALE)
            else:
                state, p_c, delta_c, advance_c = state.vec1_lazy(qk, QK_SCALE)

    tlx.async_load_wait_group(1)

    with tlx.warp_pipeline_stage("mem2", priority=1):
        # Preserve the optimized Gluon lazy cadence exactly: consume K from
        # LDS before issuing the independent V copy for this slot.  The
        # V-before-K ordering belongs only to Gluon's non-lazy fallback path.
        kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, CUR_SLOT)), relaxed=True)
        tok_v = tlx.async_load(v_ptrs + acv_n * stride_vn, tlx.local_view(v_buf, CUR_SLOT))
        tlx.async_load_commit_group([tok_v])
        state = state.rescale_lazy(delta_c, advance_c)

    return state, p_c, kt_dot


@triton.jit
def _attn_inner_pipelined(
    state,
    q,
    k_ptrs,
    v_ptrs,
    offs_m,
    offs_n,
    block_start,
    block_end,
    k_buf,
    v_buf,
    stride_kn,
    stride_vn,
    N_CTX: tl.constexpr,
    QK_SCALE: tl.constexpr,
    DIAG_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    PREFETCH_CAUSAL_DIAGONAL: tl.constexpr,
):
    # The long causal BM256 object owns four physical LDS slots for its
    # diagonal pruning, but the unmasked steady-state pipeline itself is still
    # a two-slot ping-pong schedule.  Keep that logical depth separate from
    # the allocation depth so lazy softmax can run on the same object.
    LAZY_BUF_DEPTH: tl.constexpr = 2
    # N1024 BF16 loses overlap with the stock LLVM scheduler on the extended
    # drain. Keep that class on its established exact-rescale control flow.
    USE_LAZY_RESCALE: tl.constexpr = (
        not MASK_STEPS and BLOCK_N == 64 and q.shape[1] == 128 and tlx.num_warps() == q.shape[0] // 32
        and BUF_DEPTH >= LAZY_BUF_DEPTH and N_CTX % q.shape[0] == 0 and N_CTX % BLOCK_N == 0
        and (q.shape[0] == 256 or (q.shape[0] == 128 and IS_CAUSAL and
                                   (N_CTX == 8 * BLOCK_N or
                                    (PREFETCH_CAUSAL_DIAGONAL and N_CTX == 16 * BLOCK_N and q.dtype == tl.float16)))))
    # On the aligned short-causal classes, extend the steady-state loop through
    # all but the final prefix tile.  The two-slot BM128 schedule hands off its
    # complete diagonal ring; N2048 fills the two otherwise-unused BM256 slots
    # while consuming that final prefix tile.
    USE_ONE_TILE_TWO_SLOT_PREFIX_DRAIN: tl.constexpr = (PREFETCH_CAUSAL_DIAGONAL and USE_LAZY_RESCALE
                                                        and BUF_DEPTH == LAZY_BUF_DEPTH and IS_CAUSAL
                                                        and (N_CTX == 8 * BLOCK_N or N_CTX == 16 * BLOCK_N)
                                                        and q.shape[0] == 128 and BLOCK_N == 64 and q.shape[1] == 128
                                                        and tlx.num_warps() == 4)
    USE_ONE_TILE_FOUR_SLOT_PREFIX_DRAIN: tl.constexpr = (PREFETCH_CAUSAL_DIAGONAL and USE_LAZY_RESCALE
                                                         and BUF_DEPTH == LAZY_BUF_DEPTH and IS_CAUSAL and N_CTX == 2048
                                                         and q.shape[0] == 256 and BLOCK_N == 64 and q.shape[1] == 128
                                                         and tlx.num_warps() == 8)
    USE_ONE_TILE_PREFIX_DRAIN: tl.constexpr = (USE_ONE_TILE_TWO_SLOT_PREFIX_DRAIN
                                               or USE_ONE_TILE_FOUR_SLOT_PREFIX_DRAIN)
    if PREFETCH_CAUSAL_DIAGONAL:
        tl.static_assert(USE_ONE_TILE_PREFIX_DRAIN)
        tl.static_assert(not MASK_STEPS)
    # Deferring the final three N8 probability fragments benefits every
    # eligible causal prefix and full-attention rows from N2048 onward.
    SPLIT_VEC1: tl.constexpr = USE_LAZY_RESCALE and (IS_CAUSAL or N_CTX >= 2048)
    # Weaving row-max work between native K16 PV steps is consistently useful
    # for full attention and for causal rows once they reach N4096.
    STEP_PV_VEC1: tl.constexpr = (USE_LAZY_RESCALE
                                  and ((q.shape[0] == 256 and
                                        (not IS_CAUSAL or q.dtype == tl.float16 or N_CTX // BLOCK_N >= 64)) or
                                       (IS_CAUSAL and q.shape[0] == 128 and N_CTX == 512)))
    # The reference CDNA4 kernel also overlaps the deferred VEC2 exponentials
    # with QK's final three K16 reductions.  Restrict this first port to the
    # aligned BM256/BN64 FP16 class where the operand/layout contract is exact.
    STEP_QK_VEC2: tl.constexpr = (USE_LAZY_RESCALE and q.shape[0] == 256 and q.shape[1] == 128 and BLOCK_N == 64
                                  and q.dtype == tl.float16
                                  and (IS_CAUSAL and N_CTX >= 4096 or (not IS_CAUSAL and N_CTX >= 2048)))
    # Match the Gluon numerator/denominator contract: FP16 reduces the rounded
    # P operand that is consumed by P*V.  For BF16, its medium-row stage-three
    # schedule uses the ordinary four-slice chain, while the N8192+ stage-four
    # schedule rotates the final two additions.
    FP16_CAST_ROWSUM: tl.constexpr = USE_LAZY_RESCALE and q.dtype == tl.float16
    CHAIN_BF16_ROWSUM: tl.constexpr = (
        2 if USE_LAZY_RESCALE and q.dtype == tl.bfloat16 and q.shape[0] == 256 and IS_CAUSAL and N_CTX >= 8192 else
        1 if USE_LAZY_RESCALE and q.dtype == tl.bfloat16 and q.shape[0] == 256 and N_CTX // BLOCK_N >= 64 else 0)
    DRAIN_STEP_PV_VEC1: tl.constexpr = STEP_PV_VEC1 and not IS_CAUSAL and N_CTX >= 8 * BLOCK_N
    # Keep the BN64 accumulator in one MFMA layout for the complete helper.
    # In particular, a masked tail can enter from a released prefix layout;
    # allowing generic PV to choose a new layout inside its loop leaves an
    # illegal blocked-to-MFMA loop-carried conversion.
    EXPLICIT_PV_LAYOUT: tl.constexpr = (BLOCK_N == 64 and (q.shape[0] == 128 or q.shape[0] == 256) and q.shape[1] == 128
                                        and tlx.num_warps() == q.shape[0] // 32)
    # Match the Gluon fallback schedule: two persistent D64 accumulator tiles
    # expose VEC1 between independent PV MFMA chains in the unmasked hot loop.
    SPLIT_PV: tl.constexpr = (not MASK_STEPS and not USE_LAZY_RESCALE and BLOCK_N == 64 and q.shape[0] == 256
                              and q.shape[1] == 128 and tlx.num_warps() == 8)
    # BN32 has no lazy-rescale path. Explicit K16 operand layouts reduce the
    # generic blocked-to-MFMA handoff on medium BM256 rows, but increase live
    # state on the long rows, so keep the measured crossover local to N<=4096.
    EXPLICIT_PV_LAYOUT32: tl.constexpr = (BLOCK_N == 32 and q.shape[0] == 256 and q.shape[1] == 128
                                          and tlx.num_warps() == 8 and N_CTX <= 4096)
    BF16_LIGHTWEIGHT_WAR: tl.constexpr = (IS_CAUSAL and N_CTX == 512 and q.shape[0] == 128 and BLOCK_N == 64
                                          and tlx.num_warps() == 4 and q.dtype == tl.bfloat16)
    RELAXED_QK_WAR: tl.constexpr = (USE_LAZY_RESCALE and IS_CAUSAL and q.shape[0] == 128 and q.shape[1] == 128
                                    and tlx.num_warps() == 4 and q.dtype == tl.float16
                                    and (N_CTX == 8 * BLOCK_N or N_CTX == 16 * BLOCK_N))
    RELAXED_PV_WAR: tl.constexpr = (USE_LAZY_RESCALE and not MASK_STEPS and IS_CAUSAL and N_CTX == 8 * BLOCK_N
                                    and q.shape[0] == 128 and BLOCK_N == 64 and q.shape[1] == 128
                                    and tlx.num_warps() == 4 and q.dtype == tl.float16)
    # Prologue: prime the pipeline for output tile block_start.
    b0 = block_start
    n0 = b0 * BLOCK_N
    if MASK_STEPS:
        mask0 = (n0 + offs_n)[:, None] < N_CTX
        tok_k0 = tlx.async_load(k_ptrs + n0 * stride_kn, tlx.local_view(k_buf, 0), mask=mask0)
        tok_v0 = tlx.async_load(v_ptrs + n0 * stride_vn, tlx.local_view(v_buf, 0), mask=mask0)
    else:
        tok_k0 = tlx.async_load(k_ptrs + n0 * stride_kn, tlx.local_view(k_buf, 0))
        tok_v0 = tlx.async_load(v_ptrs + n0 * stride_vn, tlx.local_view(v_buf, 0))
    tlx.async_load_commit_group([tok_k0])
    tlx.async_load_commit_group([tok_v0])
    n1 = (b0 + 1) * BLOCK_N
    if MASK_STEPS:
        tok_k1 = tlx.async_load(k_ptrs + n1 * stride_kn, tlx.local_view(k_buf, 1), mask=(n1 + offs_n)[:, None] < N_CTX)
    else:
        tok_k1 = tlx.async_load(k_ptrs + n1 * stride_kn, tlx.local_view(k_buf, 1))
    tlx.async_load_commit_group([tok_k1])

    wait0 = tlx.async_load_wait_group(2)
    kt0 = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 0)), token=wait0, relaxed=True)
    qk = tl.dot(q, kt0)
    if USE_LAZY_RESCALE:
        if SPLIT_VEC1:
            state, p_c, _, _ = state.vec1_lazy_split(qk, QK_SCALE)
        else:
            state, p_c, _, _ = state.vec1_lazy(qk, QK_SCALE)
        # The unmasked prefix starts from an empty accumulator. Its first
        # correction is therefore equivalent to initializing the denominator
        # to zero and can be omitted entirely.
        state = SoftmaxState(state.acc, tl.zeros([q.shape[0]], tl.float32), state.m_i)
    else:
        state, p_c, alpha_c = state.vec1(
            qk,
            b0 * BLOCK_N,
            offs_m,
            offs_n,
            N_CTX,
            QK_SCALE,
            DIAG_OFFSET,
            MASK_STEPS,
            IS_CAUSAL,
        )
    if RELAXED_QK_WAR:
        # K2 may reuse K0 only after every wave has consumed both N32 results.
        _attn_qk_war_barrier_relaxed(qk)
    else:
        tl.debug_barrier()
    n2 = (b0 + 2) * BLOCK_N
    if MASK_STEPS:
        tok_k2 = tlx.async_load(k_ptrs + n2 * stride_kn, tlx.local_view(k_buf, 0), mask=(n2 + offs_n)[:, None] < N_CTX)
    else:
        tok_k2 = tlx.async_load(k_ptrs + n2 * stride_kn, tlx.local_view(k_buf, 0))
    tlx.async_load_commit_group([tok_k2])
    wait1 = tlx.async_load_wait_group(1)
    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 1)), token=wait1, relaxed=True)
    if MASK_STEPS:
        tok_v1 = tlx.async_load(v_ptrs + n1 * stride_vn, tlx.local_view(v_buf, 1), mask=(n1 + offs_n)[:, None] < N_CTX)
    else:
        tok_v1 = tlx.async_load(v_ptrs + n1 * stride_vn, tlx.local_view(v_buf, 1))
    tlx.async_load_commit_group([tok_v1])

    if SPLIT_PV:
        acc_sub0, acc_sub1 = _split_cols(state.acc)
        state = SoftmaxState(acc_sub0, state.l_i, state.m_i)

    if USE_LAZY_RESCALE:
        # Pair adjacent iterations so both ping-pong slots are compile-time
        # constants and the backend can schedule across the complete cadence.
        # The short-causal handoff drains one prefix tile; other schedules keep
        # the established three-tile drain.
        DRAIN_TILES: tl.constexpr = 1 if USE_ONE_TILE_PREFIX_DRAIN else 3
        ODD_TAIL: tl.constexpr = (N_CTX // BLOCK_N - DRAIN_TILES) % 2 == 1
        main_loop_pairs = (block_end - block_start - DRAIN_TILES) // 2
        for pair_idx in tl.range(0, main_loop_pairs, num_stages=1):
            block_n = block_start + pair_idx * 2
            state, p_c, kt_dot = _attn_inner_pipelined_lazy_step(
                state,
                p_c,
                q,
                kt_dot,
                k_ptrs,
                v_ptrs,
                offs_n,
                block_n,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                QK_SCALE,
                BLOCK_N,
                0,
                1,
                SPLIT_VEC1,
                STEP_PV_VEC1,
                STEP_QK_VEC2,
                FP16_CAST_ROWSUM,
                CHAIN_BF16_ROWSUM,
            )
            state, p_c, kt_dot = _attn_inner_pipelined_lazy_step(
                state,
                p_c,
                q,
                kt_dot,
                k_ptrs,
                v_ptrs,
                offs_n,
                block_n + 1,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                QK_SCALE,
                BLOCK_N,
                1,
                0,
                SPLIT_VEC1,
                STEP_PV_VEC1,
                STEP_QK_VEC2,
                FP16_CAST_ROWSUM,
                CHAIN_BF16_ROWSUM,
            )

        if ODD_TAIL:
            block_n = block_start + main_loop_pairs * 2
            state, p_c, kt_dot = _attn_inner_pipelined_lazy_step(
                state,
                p_c,
                q,
                kt_dot,
                k_ptrs,
                v_ptrs,
                offs_n,
                block_n,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                QK_SCALE,
                BLOCK_N,
                0,
                1,
                SPLIT_VEC1,
                STEP_PV_VEC1,
                STEP_QK_VEC2,
                FP16_CAST_ROWSUM,
                CHAIN_BF16_ROWSUM,
            )

    else:
        for block_n in tl.range(block_start, block_end - 3, num_stages=1):
            cur_slot = (block_n - block_start) % BUF_DEPTH
            nxt_slot = (block_n + 1 - block_start) % BUF_DEPTH
            ack_n = (block_n + 3) * BLOCK_N
            acv_n = (block_n + 2) * BLOCK_N
            ahead_n = (block_n + 1) * BLOCK_N

            with tlx.warp_pipeline_stage("dot1", priority=0):
                qk = tl.dot(q, kt_dot)
                if SPLIT_PV:
                    state, acc_sub1, p_dot = state.vec2_split_acc(acc_sub1, p_c, alpha_c, q.dtype)
                else:
                    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)

            tlx.async_load_wait_group(1)

            with tlx.warp_pipeline_stage("mem1", priority=1):
                v_dot = tlx.local_load(tlx.local_view(v_buf, cur_slot), relaxed=True)
                if MASK_STEPS:
                    tok_k = tlx.async_load(
                        k_ptrs + ack_n * stride_kn,
                        tlx.local_view(k_buf, nxt_slot),
                        mask=(ack_n + offs_n)[:, None] < N_CTX,
                    )
                else:
                    tok_k = tlx.async_load(k_ptrs + ack_n * stride_kn, tlx.local_view(k_buf, nxt_slot))
                tlx.async_load_commit_group([tok_k])

            with tlx.warp_pipeline_stage("dot2", priority=0):
                if SPLIT_PV:
                    split_mma: tl.constexpr = tlx.amd_mfma_layout(
                        version=4,
                        instr_shape=[32, 32, 16],
                        transposed=True,
                        warps_per_cta=[tlx.num_warps(), 1],
                    )
                    split_v_layout: tl.constexpr = tlx.dot_operand_layout(1, split_mma, k_width=4)
                    v_dot = tlx.require_layout(v_dot, split_v_layout, pin=False)
                    v_sub0, v_sub1 = tl.split(v_dot.reshape([64, 2, 64]).permute(0, 2, 1))
                    acc_sub0 = _attn_dot_pv_mfma_split(state.acc, p_dot, v_sub0)
                    state = SoftmaxState(acc_sub0, state.l_i, state.m_i)
                    state, p_c, alpha_c = state.vec1(
                        qk,
                        ahead_n,
                        offs_m,
                        offs_n,
                        N_CTX,
                        QK_SCALE,
                        DIAG_OFFSET,
                        MASK_STEPS,
                        IS_CAUSAL,
                    )
                    acc_sub1 = _attn_dot_pv_mfma_split(acc_sub1, p_dot, v_sub1)
                elif EXPLICIT_PV_LAYOUT32:
                    acc = _attn_dot_pv_mfma32(state.acc, p_dot, v_dot)
                elif EXPLICIT_PV_LAYOUT:
                    acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
                else:
                    acc = tl.dot(p_dot, v_dot, state.acc)
                if not SPLIT_PV:
                    state = SoftmaxState(acc, state.l_i, state.m_i)
                    state, p_c, alpha_c = state.vec1(
                        qk,
                        ahead_n,
                        offs_m,
                        offs_n,
                        N_CTX,
                        QK_SCALE,
                        DIAG_OFFSET,
                        MASK_STEPS,
                        IS_CAUSAL,
                    )

            tlx.async_load_wait_group(1)

            with tlx.warp_pipeline_stage("mem2", priority=1):
                if tlx.num_warps() == 8:
                    if MASK_STEPS:
                        tok_v = tlx.async_load(
                            v_ptrs + acv_n * stride_vn,
                            tlx.local_view(v_buf, cur_slot),
                            mask=(acv_n + offs_n)[:, None] < N_CTX,
                        )
                    else:
                        tok_v = tlx.async_load(v_ptrs + acv_n * stride_vn, tlx.local_view(v_buf, cur_slot))
                    tlx.async_load_commit_group([tok_v])
                    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, cur_slot)), relaxed=True)
                else:
                    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, cur_slot)), relaxed=True)
                    if MASK_STEPS:
                        tok_v = tlx.async_load(
                            v_ptrs + acv_n * stride_vn,
                            tlx.local_view(v_buf, cur_slot),
                            mask=(acv_n + offs_n)[:, None] < N_CTX,
                        )
                    else:
                        tok_v = tlx.async_load(v_ptrs + acv_n * stride_vn, tlx.local_view(v_buf, cur_slot))
                    tlx.async_load_commit_group([tok_v])

        if SPLIT_PV:
            split_pv_layout: tl.constexpr = tlx.amd_mfma_layout(
                version=4,
                instr_shape=[32, 32, 16],
                transposed=True,
                warps_per_cta=[tlx.num_warps(), 1],
            )
            acc = tl.join(state.acc, acc_sub1).permute(0, 2, 1).reshape([state.acc.shape[0], 128])
            acc = tlx.require_layout(acc, split_pv_layout, pin=False)
            state = SoftmaxState(acc, state.l_i, state.m_i)

    if USE_ONE_TILE_PREFIX_DRAIN:
        # The extended odd tail completed every prefix tile except n-1.  Its
        # final mem2 has K(diag0) resident in slot 0 and V(diag0) in flight,
        # while mem1 has K(diag1) in flight in slot 1.  Consume the final
        # prefix probability with V(n-1), then reuse that dead V slot for
        # V(diag1).  The short diagonal helper drains these groups in order.
        if USE_ONE_TILE_FOUR_SLOT_PREFIX_DRAIN:
            # Slots 2/3 are outside the steady-state ping-pong ring.  Start
            # their diagonal K/V pairs while the final prefix softmax and PV
            # remain available to cover the copies.
            for diagonal_slot in tl.static_range(2):
                diagonal_n = (block_end + diagonal_slot + 2) * BLOCK_N
                tok_k_diagonal = tlx.async_load(
                    k_ptrs + diagonal_n * stride_kn,
                    tlx.local_view(k_buf, diagonal_slot + 2),
                )
                tlx.async_load_commit_group([tok_k_diagonal])
                tok_v_diagonal = tlx.async_load(
                    v_ptrs + diagonal_n * stride_vn,
                    tlx.local_view(v_buf, diagonal_slot + 2),
                )
                tlx.async_load_commit_group([tok_v_diagonal])
        if SPLIT_VEC1:
            state, p_dot = state.vec2_lazy_split(
                p_c,
                QK_SCALE,
                q.dtype,
                FP16_CAST_ROWSUM,
                CHAIN_BF16_ROWSUM,
            )
        else:
            state, p_dot = state.vec2_lazy(p_c, q.dtype)
        v_dot = tlx.local_load(tlx.local_view(v_buf, 1), relaxed=True)
        acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
        state = SoftmaxState(acc, state.l_i, state.m_i)
        if RELAXED_PV_WAR:
            _attn_pv_war_barrier_relaxed(state.acc)
        else:
            _attn_war_barrier(BF16_LIGHTWEIGHT_WAR)
        diagonal_v1_n = (block_end + 1) * BLOCK_N
        tok_v_diagonal1 = tlx.async_load(
            v_ptrs + diagonal_v1_n * stride_vn,
            tlx.local_view(v_buf, 1),
        )
        tlx.async_load_commit_group([tok_v_diagonal1])
        return state

    # Drain the last three output tiles without out-of-bounds global prefetches.
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    DRAIN_BUF_DEPTH: tl.constexpr = LAZY_BUF_DEPTH if USE_LAZY_RESCALE else BUF_DEPTH
    s_nm3 = (nm3 - block_start) % DRAIN_BUF_DEPTH
    s_nm2 = (nm2 - block_start) % DRAIN_BUF_DEPTH
    s_nm1 = (nm1 - block_start) % DRAIN_BUF_DEPTH

    qk = tl.dot(q, kt_dot)
    tlx.async_load_wait_group(2)
    v_dot = tlx.local_load(tlx.local_view(v_buf, s_nm3), relaxed=True)
    if USE_LAZY_RESCALE:
        if SPLIT_VEC1:
            state, p_dot = state.vec2_lazy_split(p_c, QK_SCALE, q.dtype, FP16_CAST_ROWSUM, CHAIN_BF16_ROWSUM)
        else:
            state, p_dot = state.vec2_lazy(p_c, q.dtype)
    else:
        state, p_dot = state.vec2(p_c, alpha_c, q.dtype)
    if USE_LAZY_RESCALE and DRAIN_STEP_PV_VEC1:
        state, p_c, delta_c, advance_c = _attn_dot_pv_vec1_lazy(
            state,
            p_dot,
            v_dot,
            qk,
            QK_SCALE,
            SPLIT_VEC1,
        )
    else:
        if EXPLICIT_PV_LAYOUT32:
            acc = _attn_dot_pv_mfma32(state.acc, p_dot, v_dot)
        elif EXPLICIT_PV_LAYOUT:
            acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
        else:
            acc = tl.dot(p_dot, v_dot, state.acc)
        state = SoftmaxState(acc, state.l_i, state.m_i)
    if USE_LAZY_RESCALE:
        if not DRAIN_STEP_PV_VEC1:
            if SPLIT_VEC1:
                state, p_c, delta_c, advance_c = state.vec1_lazy_split(qk, QK_SCALE)
            else:
                state, p_c, delta_c, advance_c = state.vec1_lazy(qk, QK_SCALE)
        state = state.rescale_lazy(delta_c, advance_c)
    else:
        state, p_c, alpha_c = state.vec1(
            qk,
            nm2 * BLOCK_N,
            offs_m,
            offs_n,
            N_CTX,
            QK_SCALE,
            DIAG_OFFSET,
            MASK_STEPS,
            IS_CAUSAL,
        )
    if RELAXED_PV_WAR:
        # The next copy may reuse this V slot only after all four D32 results.
        _attn_pv_war_barrier_relaxed(state.acc)
    else:
        _attn_war_barrier(BF16_LIGHTWEIGHT_WAR)
    nm1_n = nm1 * BLOCK_N
    if MASK_STEPS:
        tok_vlast = tlx.async_load(
            v_ptrs + nm1_n * stride_vn,
            tlx.local_view(v_buf, s_nm1),
            mask=(nm1_n + offs_n)[:, None] < N_CTX,
        )
    else:
        tok_vlast = tlx.async_load(v_ptrs + nm1_n * stride_vn, tlx.local_view(v_buf, s_nm1))
    tlx.async_load_commit_group([tok_vlast])
    tlx.async_load_wait_group(2)
    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, s_nm1)), relaxed=True)

    qk = tl.dot(q, kt_dot)
    tlx.async_load_wait_group(1)
    v_dot = tlx.local_load(tlx.local_view(v_buf, s_nm2), relaxed=True)
    if USE_LAZY_RESCALE:
        if SPLIT_VEC1:
            state, p_dot = state.vec2_lazy_split(p_c, QK_SCALE, q.dtype, FP16_CAST_ROWSUM, CHAIN_BF16_ROWSUM)
        else:
            state, p_dot = state.vec2_lazy(p_c, q.dtype)
    else:
        state, p_dot = state.vec2(p_c, alpha_c, q.dtype)
    if USE_LAZY_RESCALE and DRAIN_STEP_PV_VEC1:
        state, p_c, delta_c, advance_c = _attn_dot_pv_vec1_lazy(
            state,
            p_dot,
            v_dot,
            qk,
            QK_SCALE,
            SPLIT_VEC1,
        )
    else:
        if EXPLICIT_PV_LAYOUT32:
            acc = _attn_dot_pv_mfma32(state.acc, p_dot, v_dot)
        elif EXPLICIT_PV_LAYOUT:
            acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
        else:
            acc = tl.dot(p_dot, v_dot, state.acc)
        state = SoftmaxState(acc, state.l_i, state.m_i)
    if USE_LAZY_RESCALE:
        if not DRAIN_STEP_PV_VEC1:
            if SPLIT_VEC1:
                state, p_c, delta_c, advance_c = state.vec1_lazy_split(qk, QK_SCALE)
            else:
                state, p_c, delta_c, advance_c = state.vec1_lazy(qk, QK_SCALE)
        state = state.rescale_lazy(delta_c, advance_c)
    else:
        state, p_c, alpha_c = state.vec1(
            qk,
            nm1 * BLOCK_N,
            offs_m,
            offs_n,
            N_CTX,
            QK_SCALE,
            DIAG_OFFSET,
            MASK_STEPS,
            IS_CAUSAL,
        )

    tlx.async_load_wait_group(0)
    v_dot = tlx.local_load(tlx.local_view(v_buf, s_nm1), relaxed=True)
    if USE_LAZY_RESCALE:
        if SPLIT_VEC1:
            state, p_dot = state.vec2_lazy_split(p_c, QK_SCALE, q.dtype, FP16_CAST_ROWSUM, CHAIN_BF16_ROWSUM)
        else:
            state, p_dot = state.vec2_lazy(p_c, q.dtype)
    else:
        state, p_dot = state.vec2(p_c, alpha_c, q.dtype)
    if EXPLICIT_PV_LAYOUT32:
        acc = _attn_dot_pv_mfma32(state.acc, p_dot, v_dot)
        state = SoftmaxState(acc, state.l_i, state.m_i)
    elif EXPLICIT_PV_LAYOUT:
        acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
        state = SoftmaxState(acc, state.l_i, state.m_i)
    else:
        acc = tl.dot(p_dot, v_dot, state.acc)
        state = SoftmaxState(acc, state.l_i, state.m_i)

    PRESERVE_ALIGNED_CAUSAL_ACC_LAYOUT: tl.constexpr = (EXPLICIT_PV_LAYOUT and not MASK_STEPS and IS_CAUSAL
                                                        and q.shape[0] == 256 and BLOCK_N == 64)
    if (EXPLICIT_PV_LAYOUT32 or EXPLICIT_PV_LAYOUT) and not PRESERVE_ALIGNED_CAUSAL_ACC_LAYOUT:
        state = SoftmaxState(tlx.release_layout(state.acc), state.l_i, state.m_i)

    return state


@triton.jit
def _attn_predicated_causal_tile(
    acc,
    l_i,
    m_i,
    q,
    offs_m,
    k_tile,
    v_tile,
    wait,
    start_n,
    N_CTX: tl.constexpr,
    QK_SCALE: tl.constexpr,
    DIAG_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_CLASS_DIAGONAL_LAZY: tl.constexpr,
):
    """Consume one diagonal tile inside a wave-uniform EXEC region."""
    if BLOCK_N == 32:
        k_tile = tlx.local_slice(k_tile, [0, 0], [32, q.shape[1]])
        v_tile = tlx.local_slice(v_tile, [0, 0], [32, q.shape[1]])
    offs_n = tl.arange(0, BLOCK_N)
    kt_dot = tlx.local_load(tlx.local_trans(k_tile), token=wait, relaxed=True)
    qk = tl.dot(q, kt_dot)
    state = SoftmaxState(acc, l_i, m_i)
    USE_CLASS_DIAGONAL_LAZY: tl.constexpr = (ENABLE_CLASS_DIAGONAL_LAZY and N_CTX <= 1024 and N_CTX % q.shape[0] == 0
                                             and N_CTX % BLOCK_N == 0 and q.shape[0] == 128 and q.shape[1] == 128
                                             and (BLOCK_N == 32 or BLOCK_N == 64) and tlx.num_warps() == 4)
    if USE_CLASS_DIAGONAL_LAZY:
        # The Gluon short-row path keeps the running max in a lazy frame on
        # both diagonal tiles. Scale before masking so a zero scale cannot form
        # ``-inf * 0`` and a negative scale cannot turn the mask sentinel into
        # ``+inf``. The lazy helpers therefore consume an already-scaled tile.
        kn = start_n + offs_n
        qk = qk * QK_SCALE
        qk = tl.where(
            offs_m[:, None] + DIAG_OFFSET >= kn[None, :],
            qk,
            float("-inf"),
        )
        qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))
        threshold: tl.constexpr = (DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16
                                   if q.dtype == tl.float16 else DIAGONAL_LAZY_RESCALE_THRESHOLD)
        if BLOCK_N == 64:
            state, p_state, score_delta, advance = state.vec1_lazy_split_threshold(qk, 1.0, threshold)
            state = state.rescale_lazy(score_delta, advance)
            state, p_dot = state.vec2_lazy_split(p_state, 1.0, q.dtype, q.dtype == tl.float16)
        else:
            state, score_delta, advance = state.vec1_lazy_max_threshold(qk, 1.0, threshold)
            p = state.vec1_lazy_exp(qk, 1.0)
            state = state.rescale_lazy(score_delta, advance)
            state, p_dot = state.vec2_lazy(p, q.dtype)
    else:
        state, p, alpha = state.vec1(
            qk,
            start_n,
            offs_m,
            offs_n,
            N_CTX,
            QK_SCALE,
            DIAG_OFFSET,
            True,
            True,
        )
        state, p_dot = state.vec2(p, alpha, q.dtype)
    # Delay V's register materialization until P is ready; this shortens its
    # live range across QK and softmax and matches its immediate PV consumer.
    v_dot = tlx.local_load(v_tile, token=wait, relaxed=True)
    # The low-only causal wave is intentionally passed as a 32-column tile
    # even when its parent diagonal tile is BN64.  Its accumulator still has
    # the MFMA layout, so use the matching explicit PV fragment helper for
    # both the lazy and ordinary softmax branches.
    if BLOCK_N == 64:
        acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
    else:
        acc = _attn_dot_pv_mfma32(state.acc, p_dot, v_dot)
    return acc, state.l_i, state.m_i


@triton.jit
def _attn_predicated_causal_tile_from_state(
    _unused_acc,
    _unused_l_i,
    _unused_m_i,
    acc,
    l_i,
    m_i,
    q,
    offs_m,
    k_tile,
    v_tile,
    wait,
    start_n,
    N_CTX: tl.constexpr,
    QK_SCALE: tl.constexpr,
    DIAG_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_CLASS_DIAGONAL_LAZY: tl.constexpr,
):
    """Compute from read-only state while carrying independent merge values."""
    return _attn_predicated_causal_tile(
        acc,
        l_i,
        m_i,
        q,
        offs_m,
        k_tile,
        v_tile,
        wait,
        start_n,
        N_CTX,
        QK_SCALE,
        DIAG_OFFSET,
        BLOCK_N,
        ENABLE_CLASS_DIAGONAL_LAZY,
    )


@triton.jit
def _attn_predicated_causal_regs_bn32(
    acc,
    l_i,
    m_i,
    q,
    offs_m,
    kt_dot,
    v_dot,
    start_n,
    N_CTX: tl.constexpr,
    QK_SCALE: tl.constexpr,
    DIAG_OFFSET: tl.constexpr,
):
    """Consume a BN32 diagonal tile after every wave has completed its LDS reads."""
    tl.static_assert(q.shape[0] == 256)
    tl.static_assert(q.shape[1] == 128)
    tl.static_assert(kt_dot.shape[0] == 128)
    tl.static_assert(kt_dot.shape[1] == 32)
    tl.static_assert(v_dot.shape[0] == 32)
    tl.static_assert(v_dot.shape[1] == 128)
    tl.static_assert(tlx.num_warps() == 8)
    offs_n = tl.arange(0, 32)
    qk = tl.dot(q, kt_dot)
    state = SoftmaxState(acc, l_i, m_i)
    state, p, alpha = state.vec1(
        qk,
        start_n,
        offs_m,
        offs_n,
        N_CTX,
        QK_SCALE,
        DIAG_OFFSET,
        True,
        True,
    )
    state, p_dot = state.vec2(p, alpha, q.dtype)
    acc = _attn_dot_pv_mfma32(state.acc, p_dot, v_dot)
    return acc, state.l_i, state.m_i


@triton.jit
def _attn_inner_full2_lazy(
    state,
    q,
    k_ptrs,
    v_ptrs,
    block_start,
    k_buf,
    v_buf,
    stride_kn,
    stride_vn,
    QK_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Consume an aligned two-tile prefix in the BM128 lazy frame."""
    tlx.async_load_wait_group(0)
    for slot in tl.static_range(2):
        start_n = (block_start + slot) * BLOCK_N
        tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf, slot))
        tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf, slot))
        tlx.async_load_commit_group([tok_k, tok_v])

    wait = tlx.async_load_wait_group(1)
    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 0)), token=wait, relaxed=True)
    qk = tl.dot(q, kt_dot)
    state, p_state, _, _ = state.vec1_lazy_split4(qk, QK_SCALE)
    # The first prefix tile starts from an empty softmax frame, so its old
    # denominator contributes exactly zero and needs no correction.
    state = SoftmaxState(state.acc, tl.zeros([q.shape[0]], tl.float32), state.m_i)
    state, p_dot = state.vec2_lazy_split4(
        p_state,
        QK_SCALE,
        q.dtype,
        q.dtype == tl.float16,
    )
    v_dot = tlx.local_load(tlx.local_view(v_buf, 0), token=wait, relaxed=True)
    acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
    state = SoftmaxState(acc, state.l_i, state.m_i)

    wait = tlx.async_load_wait_group(0)
    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 1)), token=wait, relaxed=True)
    qk = tl.dot(q, kt_dot)
    state, p_state, score_delta, advance = state.vec1_lazy_split4(qk, QK_SCALE)
    state = state.rescale_lazy(score_delta, advance)
    state, p_dot = state.vec2_lazy_split4(
        p_state,
        QK_SCALE,
        q.dtype,
        q.dtype == tl.float16,
    )
    v_dot = tlx.local_load(tlx.local_view(v_buf, 1), token=wait, relaxed=True)
    acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
    return SoftmaxState(tlx.release_layout(acc), state.l_i, state.m_i)


@triton.jit
def _attn_inner_short(
    state,
    q,
    k_ptrs,
    v_ptrs,
    offs_m,
    offs_n,
    block_start,
    block_end,
    k_buf,
    v_buf,
    stride_kn,
    stride_vn,
    N_CTX: tl.constexpr,
    QK_SCALE: tl.constexpr,
    DIAG_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    USE_DIRECT_LOAD: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    PREFIX_DIAGONAL_HANDOFF: tl.constexpr,
):
    """Process ranges too short to safely fill the rotated four-stage pipeline."""
    FOUR_SLOT_PREFIX_HANDOFF: tl.constexpr = (PREFIX_DIAGONAL_HANDOFF and not USE_DIRECT_LOAD and MASK_STEPS
                                              and IS_CAUSAL and q.shape[0] == 256 and q.shape[1] == 128
                                              and BLOCK_N == 64 and BUF_DEPTH == 2 and tlx.num_warps() == 8
                                              and N_CTX == 2048)
    if PREFIX_DIAGONAL_HANDOFF and not FOUR_SLOT_PREFIX_HANDOFF:
        # Prefix handoff order is K1, V0, V1; K0 is already resident.  Drain
        # through V0 and let diagonal tile 0 overlap the final V1 transfer.
        tlx.async_load_wait_group(1)
    else:
        tlx.async_load_wait_group(0)

    num_blocks = block_end - block_start
    PRUNE_CAUSAL_DIAGONAL: tl.constexpr = (MASK_STEPS and IS_CAUSAL and q.shape[1] == 128 and
                                           (BLOCK_N == 64 or (BLOCK_N == 32 and q.shape[0] == 256 and N_CTX == 2048))
                                           and N_CTX % q.shape[0] == 0 and N_CTX % BLOCK_N == 0
                                           and ((q.shape[0] == 128 and tlx.num_warps() == 4 and N_CTX <= 1024) or
                                                (q.shape[0] == 256 and tlx.num_warps() == 8)))
    if PREFIX_DIAGONAL_HANDOFF:
        tl.static_assert(not USE_DIRECT_LOAD)
        tl.static_assert(PRUNE_CAUSAL_DIAGONAL)
        tl.static_assert(BUF_DEPTH == 2 and BLOCK_N == 64)
        tl.static_assert((q.shape[0] == 128 and q.shape[1] == 128 and tlx.num_warps() == 4 and
                          (N_CTX == 8 * BLOCK_N or N_CTX == 16 * BLOCK_N)) or FOUR_SLOT_PREFIX_HANDOFF)
    if USE_DIRECT_LOAD:
        for block_offset in tl.range(0, num_blocks, num_stages=1):
            start_n = (block_start + block_offset) * BLOCK_N
            if MASK_STEPS:
                mask = (start_n + offs_n)[:, None] < N_CTX
                k = tl.load(k_ptrs + start_n * stride_kn, mask=mask, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn)
            qk = tl.dot(q, tl.trans(k))
            state, p, alpha = state.vec1(
                qk,
                start_n,
                offs_m,
                offs_n,
                N_CTX,
                QK_SCALE,
                DIAG_OFFSET,
                MASK_STEPS,
                IS_CAUSAL,
            )
            state, p_dot = state.vec2(p, alpha, q.dtype)
            if MASK_STEPS:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=mask, other=0.0)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn)
            USE_MFMA_DIRECT_PV: tl.constexpr = (IS_CAUSAL and q.shape[0] == 256 and q.shape[1] == 128 and BLOCK_N == 64
                                                and tlx.num_warps() == 8)
            if USE_MFMA_DIRECT_PV:
                acc = _attn_dot_pv_mfma(state.acc, p_dot, v)
            else:
                acc = tl.dot(p_dot, v, state.acc)
            state = SoftmaxState(acc, state.l_i, state.m_i)
    else:
        FOUR_SLOT_PRUNE: tl.constexpr = (PRUNE_CAUSAL_DIAGONAL and BLOCK_N == 64 and q.shape[0] == 256
                                         and tlx.num_warps() == 8)
        if FOUR_SLOT_PRUNE:
            # Consume the aligned BM256 diagonal in four unique LDS slots.
            # This avoids the two-slot ring's WAR rendezvous.  A nonzero
            # N2048 block_start means the prefix already supplied all slots;
            # early query tiles without a prefix still load them here.
            need_diagonal_load = block_start == 0 if FOUR_SLOT_PREFIX_HANDOFF else True
            if need_diagonal_load:
                for slot in tl.static_range(4):
                    start_n = (block_start + slot) * BLOCK_N
                    tok_k = tlx.async_load(
                        k_ptrs + start_n * stride_kn,
                        tlx.local_view(k_buf, slot),
                    )
                    tok_v = tlx.async_load(
                        v_ptrs + start_n * stride_vn,
                        tlx.local_view(v_buf, slot),
                    )
                    tlx.async_load_commit_group([tok_k, tok_v])

            wait = tlx.async_load_wait_group(0)
            mma: tl.constexpr = tlx.amd_mfma_layout(
                version=4,
                instr_shape=[32, 32, 16],
                transposed=True,
                warps_per_cta=[tlx.num_warps(), 1],
            )
            mma_rows: tl.constexpr = tlx.slice_layout(mma, 1)
            # Gluon carries both softmax row states in SliceLayout(mma, 1).
            # Materialize TLX's otherwise-blocked running max before entering
            # the wave-predicated loop so any register redistribution remains
            # convergent and the predicated bodies need no CTA-wide scratch.
            state = SoftmaxState(
                state.acc,
                tlx.require_layout(state.l_i, mma_rows, pin=True),
                tlx.require_layout(state.m_i, mma_rows, pin=True),
            )
            wave = tlx.thread_id(0) // CDNA_WAVE_SIZE
            wave = wave ^ ((wave // 4) * 3)
            wave_m_last = (wave * CDNA_MFMA_ROWS_PER_WAVE + CDNA_MFMA_ROWS_PER_WAVE - 1 + DIAG_OFFSET)
            # Keep one runtime loop body, matching Gluon's four-slot diagonal
            # consumer.  Unrolling this loop duplicates both the BN64 and
            # low-half BN32 MFMA bodies four times in the code object.
            for diag_slot in tl.range(0, num_blocks, num_stages=1):
                start_n = (block_start + diag_slot) * BLOCK_N
                diagonal_start = start_n % q.shape[0]
                active_full = wave_m_last >= diagonal_start + CDNA_MFMA_ROWS_PER_WAVE
                active_lo = wave_m_last >= diagonal_start
                active_lo_only = active_lo & ~active_full
                acc, l_i, m_i = tlx.warp_predicate(
                    active_full,
                    (state.acc, state.l_i, state.m_i),
                    _attn_predicated_causal_tile,
                    args=(
                        q,
                        offs_m,
                        tlx.local_view(k_buf, diag_slot),
                        tlx.local_view(v_buf, diag_slot),
                        wait,
                        start_n,
                        N_CTX,
                        QK_SCALE,
                        DIAG_OFFSET,
                        BLOCK_N,
                        True,
                    ),
                    wave_uniform=True,
                )
                state = SoftmaxState(acc, l_i, m_i)
                next_acc, next_l_i, next_m_i = tlx.warp_predicate(
                    active_lo_only,
                    (
                        tl.zeros_like(state.acc),
                        tl.zeros_like(state.l_i),
                        tl.zeros_like(state.m_i),
                    ),
                    _attn_predicated_causal_tile_from_state,
                    args=(
                        state.acc,
                        state.l_i,
                        state.m_i,
                        q,
                        offs_m,
                        tlx.local_view(k_buf, diag_slot),
                        tlx.local_view(v_buf, diag_slot),
                        wait,
                        start_n,
                        N_CTX,
                        QK_SCALE,
                        DIAG_OFFSET,
                        CDNA_MFMA_ROWS_PER_WAVE,
                        True,
                    ),
                    wave_uniform=True,
                )
                state = SoftmaxState(
                    tl.where(active_lo_only, next_acc, state.acc),
                    tl.where(active_lo_only, next_l_i, state.l_i),
                    tl.where(active_lo_only, next_m_i, state.m_i),
                )
            return state

        # N512's two-slot diagonal is always consumed as a pair.  Preserve
        # that target schedule for BF16/FP16 short classes while leaving
        # longer and ragged ranges on the general loop shape.
        PAIR_LOOP_UNROLL: tl.constexpr = 2 if N_CTX == 512 else 1
        for chunk_start in tl.range(
                0,
                num_blocks,
                BUF_DEPTH,
                num_stages=1,
                loop_unroll_factor=PAIR_LOOP_UNROLL,
        ):
            if not PREFIX_DIAGONAL_HANDOFF:
                for slot in tl.static_range(BUF_DEPTH):
                    block_offset = chunk_start + slot
                    if block_offset < num_blocks:
                        start_n = (block_start + block_offset) * BLOCK_N
                        if MASK_STEPS and N_CTX % BLOCK_N != 0:
                            mask = (start_n + offs_n)[:, None] < N_CTX
                            tok_k = tlx.async_load(
                                k_ptrs + start_n * stride_kn,
                                tlx.local_view(k_buf, slot),
                                mask=mask,
                            )
                            tok_v = tlx.async_load(
                                v_ptrs + start_n * stride_vn,
                                tlx.local_view(v_buf, slot),
                                mask=mask,
                            )
                        else:
                            tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf, slot))
                            tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf, slot))
                        tlx.async_load_commit_group([tok_k, tok_v])

            if PRUNE_CAUSAL_DIAGONAL:
                # Each K/V pair is one commit group. Start the first tile once
                # its pair is ready and overlap the second pair's DMA with the
                # first tile's matrix/softmax work.
                wait = tlx.async_load_wait_group(BUF_DEPTH - 1)
            else:
                wait = tlx.async_load_wait_group(0)

            for slot in tl.static_range(BUF_DEPTH):
                block_offset = chunk_start + slot
                if block_offset < num_blocks:
                    start_n = (block_start + block_offset) * BLOCK_N
                    if PRUNE_CAUSAL_DIAGONAL and BLOCK_N == 32:
                        if slot == 1:
                            wait = tlx.async_load_wait_group(0)
                        # Keep LDS consumption convergent. The barrier after
                        # this pair prevents either slot from being refilled
                        # until every wave has completed these register loads.
                        kt_dot = tlx.local_load(
                            tlx.local_trans(tlx.local_view(k_buf, slot)),
                            token=wait,
                            relaxed=True,
                        )
                        v_dot = tlx.local_load(
                            tlx.local_view(v_buf, slot),
                            token=wait,
                            relaxed=True,
                        )
                        wave = tlx.thread_id(0) // CDNA_WAVE_SIZE
                        wave = wave ^ ((wave // 4) * 3)
                        wave_m_last = (wave * CDNA_MFMA_ROWS_PER_WAVE + CDNA_MFMA_ROWS_PER_WAVE - 1 + DIAG_OFFSET)
                        diagonal_start = start_n % q.shape[0]
                        active = wave_m_last >= diagonal_start
                        acc, l_i, m_i = tlx.warp_predicate(
                            active,
                            (state.acc, state.l_i, state.m_i),
                            _attn_predicated_causal_regs_bn32,
                            args=(
                                q,
                                offs_m,
                                kt_dot,
                                v_dot,
                                start_n,
                                N_CTX,
                                QK_SCALE,
                                DIAG_OFFSET,
                            ),
                            wave_uniform=True,
                        )
                        state = SoftmaxState(acc, l_i, m_i)

                    if PRUNE_CAUSAL_DIAGONAL and BLOCK_N == 64:
                        if slot == 1:
                            wait = tlx.async_load_wait_group(0)
                        # The four MFMA waves own consecutive 32-row bands.
                        # A scalar predicate is uniform within each wave and
                        # directly controls physical lanes, so acc and the two
                        # row states may retain their independent layouts.
                        wave = tlx.thread_id(0) // CDNA_WAVE_SIZE
                        if q.shape[0] == 256:
                            wave = wave ^ ((wave // 4) * 3)
                        wave_m_last = (wave * CDNA_MFMA_ROWS_PER_WAVE + CDNA_MFMA_ROWS_PER_WAVE - 1 + DIAG_OFFSET)
                        diagonal_start = start_n % q.shape[0]
                        active_full = wave_m_last >= diagonal_start + CDNA_MFMA_ROWS_PER_WAVE
                        active_lo = wave_m_last >= diagonal_start
                        active_lo_only = active_lo & ~active_full
                        STAGGER_FINAL_NORMALIZE: tl.constexpr = (PRUNE_CAUSAL_DIAGONAL and q.shape[0] == 128
                                                                 and N_CTX <= 1024 and slot == 1)
                        if STAGGER_FINAL_NORMALIZE:
                            acc, l_i, m_i = tlx.warp_predicate(
                                active_full,
                                (state.acc, state.l_i, state.m_i),
                                _attn_predicated_causal_tile,
                                args=(
                                    q,
                                    offs_m,
                                    tlx.local_view(k_buf, slot),
                                    tlx.local_view(v_buf, slot),
                                    wait,
                                    start_n,
                                    N_CTX,
                                    QK_SCALE,
                                    DIAG_OFFSET,
                                    BLOCK_N,
                                    True,
                                ),
                                wave_uniform=True,
                            )
                            state = SoftmaxState(acc, l_i, m_i)
                            acc, l_i, m_i = tlx.warp_predicate(
                                active_lo_only,
                                (state.acc, state.l_i, state.m_i),
                                _attn_predicated_causal_tile,
                                args=(
                                    q,
                                    offs_m,
                                    tlx.local_view(k_buf, slot),
                                    tlx.local_view(v_buf, slot),
                                    wait,
                                    start_n,
                                    N_CTX,
                                    QK_SCALE,
                                    DIAG_OFFSET,
                                    CDNA_MFMA_ROWS_PER_WAVE,
                                    True,
                                ),
                                wave_uniform=True,
                            )
                            state = SoftmaxState(
                                acc * fast_dividef(1.0, l_i)[:, None],
                                l_i,
                                m_i,
                            )
                        else:
                            acc, l_i, m_i = tlx.warp_predicate(
                                active_full,
                                (state.acc, state.l_i, state.m_i),
                                _attn_predicated_causal_tile,
                                args=(
                                    q,
                                    offs_m,
                                    tlx.local_view(k_buf, slot),
                                    tlx.local_view(v_buf, slot),
                                    wait,
                                    start_n,
                                    N_CTX,
                                    QK_SCALE,
                                    DIAG_OFFSET,
                                    BLOCK_N,
                                    True,
                                ),
                                wave_uniform=True,
                            )
                            state = SoftmaxState(acc, l_i, m_i)
                            acc, l_i, m_i = tlx.warp_predicate(
                                active_lo_only,
                                (state.acc, state.l_i, state.m_i),
                                _attn_predicated_causal_tile,
                                args=(
                                    q,
                                    offs_m,
                                    tlx.local_view(k_buf, slot),
                                    tlx.local_view(v_buf, slot),
                                    wait,
                                    start_n,
                                    N_CTX,
                                    QK_SCALE,
                                    DIAG_OFFSET,
                                    CDNA_MFMA_ROWS_PER_WAVE,
                                    True,
                                ),
                                wave_uniform=True,
                            )
                            state = SoftmaxState(acc, l_i, m_i)
                    elif not PRUNE_CAUSAL_DIAGONAL:
                        kt_dot = tlx.local_load(
                            tlx.local_trans(tlx.local_view(k_buf, slot)),
                            token=wait,
                            relaxed=True,
                        )
                        v_dot = tlx.local_load(tlx.local_view(v_buf, slot), token=wait, relaxed=True)
                        qk = tl.dot(q, kt_dot)
                        state, p, alpha = state.vec1(
                            qk,
                            start_n,
                            offs_m,
                            offs_n,
                            N_CTX,
                            QK_SCALE,
                            DIAG_OFFSET,
                            MASK_STEPS,
                            IS_CAUSAL,
                        )
                        state, p_dot = state.vec2(p, alpha, q.dtype)
                        if BLOCK_N == 64 and q.shape[1] == 128:
                            acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
                        else:
                            acc = tl.dot(p_dot, v_dot, state.acc)
                        state = SoftmaxState(acc, state.l_i, state.m_i)

            if not PRUNE_CAUSAL_DIAGONAL or chunk_start + BUF_DEPTH < num_blocks:
                # Only a later refill needs a cross-wave read-completion
                # rendezvous. The final predicated chunk returns with no LDS
                # overwrite, matching the Gluon short-tail schedule.
                tl.debug_barrier()

    return state


@triton.jit
def _attn_cluster_tile(
    pid_m,
    off_z,
    off_h,
    Q,
    K,
    V,
    Out,
    k_buf,
    v_buf,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    N_CTX: tl.constexpr,
    QK_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_DIRECT_LOAD: tl.constexpr,
    USE_Q_LDS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    local_m = tl.arange(0, BLOCK_M)
    BALANCE_CAUSAL_WAVES: tl.constexpr = (IS_CAUSAL and N_CTX % BLOCK_M == 0 and HEAD_DIM == 128 and BLOCK_M == 256)
    if BALANCE_CAUSAL_WAVES:
        # Keep the first four logical row bands forward and reverse the last
        # four: [0, 1, 2, 3, 7, 6, 5, 4].
        wave_m = local_m // 32
        wave_m = wave_m ^ ((wave_m // 4) * 3)
        local_m = wave_m * 32 + local_m % 32
    offs_m = pid_m * BLOCK_M + local_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    ALIGNED_Q: tl.constexpr = N_CTX % BLOCK_M == 0
    Q_REGISTER_CG: tl.constexpr = (ALIGNED_Q and BLOCK_M == 256 and HEAD_DIM == 128 and tlx.num_warps() == 8
                                   and (BLOCK_N == 64 or N_CTX >= 8192))
    # Stage Q through the same padded LDS representation used by the target
    # Gluon kernel.  For the aligned BM256 causal diagonal, K/V use four
    # 64x128 slots (64 KiB); aliasing Q's 256x128 tile with that allocation
    # lets the allocator reclaim the Q storage before the ring is filled.
    USE_Q_LDS_BM256: tl.constexpr = (IS_CAUSAL and ALIGNED_Q and BLOCK_M == 256 and BLOCK_N == 64 and HEAD_DIM == 128
                                     and tlx.num_warps() == 8)
    LAZY_RESCALE_CANDIDATE: tl.constexpr = (BLOCK_N == 64 and HEAD_DIM == 128
                                            and ((BLOCK_M == 256 and tlx.num_warps() == 8) or
                                                 (IS_CAUSAL and BLOCK_M == 128 and tlx.num_warps() == 4))
                                            and N_CTX % BLOCK_M == 0 and N_CTX % BLOCK_N == 0)
    SCORE_SCALE_BM256: tl.constexpr = (BLOCK_M == 256 and BLOCK_N == 64 and HEAD_DIM == 128 and tlx.num_warps() == 8
                                       and ((IS_CAUSAL and N_CTX <= 4096) or
                                            (not IS_CAUSAL and (N_CTX <= 512 or
                                                                (Q.dtype.element_ty == tl.float16 and N_CTX <= 1024)))))
    # Avoid scaling through the storage dtype when the factor magnifies: finite
    # FP16 Q can overflow, and BF16 incurs extra logit error from the early cast.
    Q_PRESCALE_DOES_NOT_MAGNIFY: tl.constexpr = QK_SCALE >= -1.0 and QK_SCALE <= 1.0
    PRESCALE_Q: tl.constexpr = (
        Q_PRESCALE_DOES_NOT_MAGNIFY and HEAD_DIM <= 128
        and ((not IS_CAUSAL and Q.dtype.element_ty == tl.float16) or LAZY_RESCALE_CANDIDATE)
        and not (IS_CAUSAL and BLOCK_M == 128 and BLOCK_N == 64 and HEAD_DIM == 128 and tlx.num_warps() == 4)
        and not SCORE_SCALE_BM256)
    if USE_Q_LDS or USE_Q_LDS_BM256:
        tl.static_assert(IS_CAUSAL)
        tl.static_assert(ALIGNED_Q)
        if USE_Q_LDS:
            tl.static_assert(BLOCK_M == 128)
        else:
            tl.static_assert(BLOCK_M == 256)
            tl.static_assert(tlx.num_warps() == 8)
        tl.static_assert(BLOCK_N == 64)
        tl.static_assert(HEAD_DIM == 128)
        # The short-class kernel drains this one-shot copy before K/V starts.
        # Its 128x128 allocation is exactly the size of the two-slot 64x128 K
        # ring, so make that non-overlapping lifetime explicit to the allocator.
        q_shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(1024, 8)],
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [16, 0],
                [32, 0],
                [64, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ],
            [BLOCK_M, HEAD_DIM],
        )
        if USE_Q_LDS_BM256:
            # BM256 follows Gluon's register-load -> shared-store path.  A
            # direct async copy uses a different pointer distribution from
            # this padded K-ring alias and silently permutes Q; the synchronous
            # store lets the compiler map the existing register layout into
            # the aliased LDS tile.
            q_buf = tlx.local_alloc(
                (BLOCK_M, HEAD_DIM),
                Q.dtype.element_ty,
                1,
                reuse=k_buf,
                layout=tlx.swizzled_layout(4, 3, 4, order=[1, 0]),
            )
            q = tl.load(q_ptrs, cache_modifier=".cg")
            if PRESCALE_Q:
                q = (q.to(tl.float32) * QK_SCALE).to(Q.dtype.element_ty)
            tlx.local_store(tlx.local_view(q_buf, 0), q)
            tl.debug_barrier()
            q = tlx.local_load(tlx.local_view(q_buf, 0))
        else:
            q_buf = tlx.local_alloc(
                (BLOCK_M, HEAD_DIM),
                Q.dtype.element_ty,
                1,
                reuse=k_buf,
                layout=q_shared_layout,
            )
            q_token = tlx.async_load(
                q_ptrs,
                tlx.local_view(q_buf, 0),
                cache_modifier=".cg",
            )
            tlx.async_load_commit_group([q_token])
            q_wait = tlx.async_load_wait_group(0)
            q = tlx.local_load(tlx.local_view(q_buf, 0), token=q_wait)
        # All waves must finish consuming aliased Q LDS before the first K DMA
        # may overwrite either ring slot.
        tl.debug_barrier()
    elif Q_REGISTER_CG:
        # Q is consumed once; on BN64 and the long BN32 rows, bypassing L1 lets
        # neighboring query tiles retain K/V without the medium-row cache cost.
        q = tl.load(q_ptrs, cache_modifier=".cg")
    elif ALIGNED_Q:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Gluon's BM256 register-load path prescales before its shared-memory
    # store. Preserve that lifetime so the Q allocation can alias the K ring.
    if PRESCALE_Q and not USE_Q_LDS_BM256:
        q = (q.to(tl.float32) * QK_SCALE).to(Q.dtype.element_ty)
    if PRESCALE_Q:
        INNER_QK_SCALE: tl.constexpr = 1.0
    else:
        INNER_QK_SCALE: tl.constexpr = QK_SCALE

    DIAG_OFFSET: tl.constexpr = 0

    state = SoftmaxState.create(BLOCK_M, HEAD_DIM)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks_total: tl.constexpr = (N_CTX + BLOCK_N - 1) // BLOCK_N
    is_modulo_mn: tl.constexpr = N_CTX % BLOCK_N == 0 and N_CTX % BLOCK_M == 0
    aligned_self_causal: tl.constexpr = IS_CAUSAL and is_modulo_mn and BLOCK_M % BLOCK_N == 0

    if IS_CAUSAL:
        if aligned_self_causal:
            # The launch bounds pid_m and all divisions are exact.
            n_blocks = (pid_m + 1) * (BLOCK_M // BLOCK_N)
        else:
            causal_end = ((pid_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N
            n_blocks = min(n_blocks_total, causal_end)
        masked_blocks: tl.constexpr = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        n_blocks = n_blocks_total
        masked_blocks: tl.constexpr = 1 if N_CTX % BLOCK_N != 0 else 0

    masked_blocks = min(masked_blocks, n_blocks)
    n_full = n_blocks - masked_blocks

    # The rotated loop needs enough blocks to fill and drain all four stages.
    # Keep short ranges on the two-slot preload fallback, and fold tiny masked tails
    # into a larger masked pipeline range just like the original Gluon kernel.
    SPLIT_PRUNED_CAUSAL_PREFIX: tl.constexpr = (not USE_DIRECT_LOAD and IS_CAUSAL and BLOCK_M == 128 and BLOCK_N == 64
                                                and HEAD_DIM == 128 and tlx.num_warps() == 4 and N_CTX <= 1024
                                                and is_modulo_mn)
    BF16_LIGHTWEIGHT_WAR: tl.constexpr = (SPLIT_PRUNED_CAUSAL_PREFIX and N_CTX == 512 and q.dtype == tl.bfloat16)
    USE_REGISTER_PREDICATED_BN32_DIAGONAL: tl.constexpr = (IS_CAUSAL and is_modulo_mn and BLOCK_M == 256
                                                           and BLOCK_N == 32 and HEAD_DIM == 128
                                                           and tlx.num_warps() == 8 and N_CTX == 2048)
    USE_ALIGNED_BM256_BN64_CAUSAL: tl.constexpr = (IS_CAUSAL and is_modulo_mn and BLOCK_M == 256 and BLOCK_N == 64
                                                   and HEAD_DIM == 128 and tlx.num_warps() == 8)
    if USE_ALIGNED_BM256_BN64_CAUSAL:
        # Gluon folds this shape to exactly one unmasked prefix body followed
        # by its four-tile wave-predicated diagonal. Spell out that constexpr
        # route so TLX does not retain the impossible masked-pipeline peers in
        # the same code object.
        diagonal_mma: tl.constexpr = tlx.amd_mfma_layout(
            version=4,
            instr_shape=[32, 32, 16],
            transposed=True,
            warps_per_cta=[tlx.num_warps(), 1],
        )
        # Both arms of the dynamic prefix guard must carry the same layout.
        # Pin the empty state before the guard and preserve the prefix's final
        # PV layout so the following wave predicate never needs an LDS bridge.
        state = SoftmaxState(
            tlx.require_layout(state.acc, diagonal_mma, pin=True),
            state.l_i,
            state.m_i,
        )
        FOUR_SLOT_PREFIX_HANDOFF: tl.constexpr = (not USE_DIRECT_LOAD and N_CTX == 2048)
        if n_full >= CLUSTER_PIPELINE_STAGES:
            state = _attn_inner_pipelined(
                state,
                q,
                k_ptrs,
                v_ptrs,
                offs_m,
                offs_n,
                0,
                n_full,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                N_CTX,
                INNER_QK_SCALE,
                DIAG_OFFSET,
                BLOCK_N,
                BUF_DEPTH,
                False,
                IS_CAUSAL,
                FOUR_SLOT_PREFIX_HANDOFF,
            )
            masked_start = n_full
        else:
            masked_start = 0
        if n_blocks > masked_start:
            state = _attn_inner_short(
                state,
                q,
                k_ptrs,
                v_ptrs,
                offs_m,
                offs_n,
                masked_start,
                n_blocks,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                N_CTX,
                INNER_QK_SCALE,
                DIAG_OFFSET,
                BLOCK_N,
                BUF_DEPTH,
                USE_DIRECT_LOAD,
                True,
                IS_CAUSAL,
                FOUR_SLOT_PREFIX_HANDOFF,
            )
    elif SPLIT_PRUNED_CAUSAL_PREFIX:
        USE_ONE_TILE_PREFIX_HANDOFF: tl.constexpr = (N_CTX == 8 * BLOCK_N
                                                     or (N_CTX == 16 * BLOCK_N and q.dtype == tl.float16))
        if USE_ONE_TILE_PREFIX_HANDOFF:
            masked_start = 0
            if n_full >= CLUSTER_PIPELINE_STAGES:
                state = _attn_inner_pipelined(
                    state,
                    q,
                    k_ptrs,
                    v_ptrs,
                    offs_m,
                    offs_n,
                    0,
                    n_full,
                    k_buf,
                    v_buf,
                    stride_kn,
                    stride_vn,
                    N_CTX,
                    INNER_QK_SCALE,
                    DIAG_OFFSET,
                    BLOCK_N,
                    BUF_DEPTH,
                    False,
                    IS_CAUSAL,
                    True,
                )
                masked_start = n_full
                if n_blocks > masked_start:
                    state = _attn_inner_short(
                        state,
                        q,
                        k_ptrs,
                        v_ptrs,
                        offs_m,
                        offs_n,
                        masked_start,
                        n_blocks,
                        k_buf,
                        v_buf,
                        stride_kn,
                        stride_vn,
                        N_CTX,
                        INNER_QK_SCALE,
                        DIAG_OFFSET,
                        BLOCK_N,
                        BUF_DEPTH,
                        USE_DIRECT_LOAD,
                        True,
                        IS_CAUSAL,
                        True,
                    )
                masked_start = n_blocks
            elif n_full == 2:
                state = _attn_inner_full2_lazy(
                    state,
                    q,
                    k_ptrs,
                    v_ptrs,
                    0,
                    k_buf,
                    v_buf,
                    stride_kn,
                    stride_vn,
                    INNER_QK_SCALE,
                    BLOCK_N,
                )
                # The diagonal immediately reuses both prefix slots.
                _attn_war_barrier(BF16_LIGHTWEIGHT_WAR)
                masked_start = n_full
            elif n_full > 0:
                state = _attn_inner_short(
                    state,
                    q,
                    k_ptrs,
                    v_ptrs,
                    offs_m,
                    offs_n,
                    0,
                    n_full,
                    k_buf,
                    v_buf,
                    stride_kn,
                    stride_vn,
                    N_CTX,
                    INNER_QK_SCALE,
                    DIAG_OFFSET,
                    BLOCK_N,
                    BUF_DEPTH,
                    USE_DIRECT_LOAD,
                    False,
                    IS_CAUSAL,
                    False,
                )
                masked_start = n_full
            if n_blocks > masked_start:
                state = _attn_inner_short(
                    state,
                    q,
                    k_ptrs,
                    v_ptrs,
                    offs_m,
                    offs_n,
                    masked_start,
                    n_blocks,
                    k_buf,
                    v_buf,
                    stride_kn,
                    stride_vn,
                    N_CTX,
                    INNER_QK_SCALE,
                    DIAG_OFFSET,
                    BLOCK_N,
                    BUF_DEPTH,
                    USE_DIRECT_LOAD,
                    True,
                    IS_CAUSAL,
                    False,
                )
        else:
            # Preserve the established fallback control-flow shape for classes
            # where the prefetched handoff is not performance-positive.
            if n_full > CLUSTER_PIPELINE_STAGES:
                state = _attn_inner_pipelined(
                    state,
                    q,
                    k_ptrs,
                    v_ptrs,
                    offs_m,
                    offs_n,
                    0,
                    n_full,
                    k_buf,
                    v_buf,
                    stride_kn,
                    stride_vn,
                    N_CTX,
                    INNER_QK_SCALE,
                    DIAG_OFFSET,
                    BLOCK_N,
                    BUF_DEPTH,
                    False,
                    IS_CAUSAL,
                    False,
                )
                # The diagonal reuses both prefix slots. Rendezvous after the
                # final relaxed LDS reads before either slot is overwritten.
                _attn_war_barrier(BF16_LIGHTWEIGHT_WAR)
            elif n_full == 2:
                state = _attn_inner_full2_lazy(
                    state,
                    q,
                    k_ptrs,
                    v_ptrs,
                    0,
                    k_buf,
                    v_buf,
                    stride_kn,
                    stride_vn,
                    INNER_QK_SCALE,
                    BLOCK_N,
                )
                # The diagonal immediately reuses both prefix slots.
                _attn_war_barrier(BF16_LIGHTWEIGHT_WAR)
            elif n_full > 0:
                state = _attn_inner_short(
                    state,
                    q,
                    k_ptrs,
                    v_ptrs,
                    offs_m,
                    offs_n,
                    0,
                    n_full,
                    k_buf,
                    v_buf,
                    stride_kn,
                    stride_vn,
                    N_CTX,
                    INNER_QK_SCALE,
                    DIAG_OFFSET,
                    BLOCK_N,
                    BUF_DEPTH,
                    USE_DIRECT_LOAD,
                    False,
                    IS_CAUSAL,
                    False,
                )
            if n_blocks > n_full:
                state = _attn_inner_short(
                    state,
                    q,
                    k_ptrs,
                    v_ptrs,
                    offs_m,
                    offs_n,
                    n_full,
                    n_blocks,
                    k_buf,
                    v_buf,
                    stride_kn,
                    stride_vn,
                    N_CTX,
                    INNER_QK_SCALE,
                    DIAG_OFFSET,
                    BLOCK_N,
                    BUF_DEPTH,
                    USE_DIRECT_LOAD,
                    True,
                    IS_CAUSAL,
                    False,
                )
    elif n_blocks > CLUSTER_PIPELINE_STAGES and (n_blocks - n_full) < CLUSTER_PIPELINE_STAGES and n_full != n_blocks:
        state = _attn_inner_pipelined(
            state,
            q,
            k_ptrs,
            v_ptrs,
            offs_m,
            offs_n,
            0,
            n_blocks,
            k_buf,
            v_buf,
            stride_kn,
            stride_vn,
            N_CTX,
            INNER_QK_SCALE,
            DIAG_OFFSET,
            BLOCK_N,
            BUF_DEPTH,
            True,
            IS_CAUSAL,
            False,
        )
    elif n_blocks > CLUSTER_PIPELINE_STAGES:
        if n_full > CLUSTER_PIPELINE_STAGES:
            state = _attn_inner_pipelined(
                state,
                q,
                k_ptrs,
                v_ptrs,
                offs_m,
                offs_n,
                0,
                n_full,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                N_CTX,
                INNER_QK_SCALE,
                DIAG_OFFSET,
                BLOCK_N,
                BUF_DEPTH,
                False,
                IS_CAUSAL,
                False,
            )

        masked_start = n_full if n_full > CLUSTER_PIPELINE_STAGES else 0
        remaining_blocks = n_blocks - masked_start
        if USE_REGISTER_PREDICATED_BN32_DIAGONAL and remaining_blocks > 0:
            state = _attn_inner_short(
                state,
                q,
                k_ptrs,
                v_ptrs,
                offs_m,
                offs_n,
                masked_start,
                n_blocks,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                N_CTX,
                INNER_QK_SCALE,
                DIAG_OFFSET,
                BLOCK_N,
                BUF_DEPTH,
                USE_DIRECT_LOAD,
                True,
                IS_CAUSAL,
                False,
            )
        elif remaining_blocks > CLUSTER_PIPELINE_STAGES:
            state = _attn_inner_pipelined(
                state,
                q,
                k_ptrs,
                v_ptrs,
                offs_m,
                offs_n,
                masked_start,
                n_blocks,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                N_CTX,
                INNER_QK_SCALE,
                DIAG_OFFSET,
                BLOCK_N,
                BUF_DEPTH,
                True,
                IS_CAUSAL,
                False,
            )
        elif remaining_blocks > 0:
            state = _attn_inner_short(
                state,
                q,
                k_ptrs,
                v_ptrs,
                offs_m,
                offs_n,
                masked_start,
                n_blocks,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                N_CTX,
                INNER_QK_SCALE,
                DIAG_OFFSET,
                BLOCK_N,
                BUF_DEPTH,
                USE_DIRECT_LOAD,
                True,
                IS_CAUSAL,
                False,
            )
    elif n_blocks > 0:
        state = _attn_inner_short(
            state,
            q,
            k_ptrs,
            v_ptrs,
            offs_m,
            offs_n,
            0,
            n_blocks,
            k_buf,
            v_buf,
            stride_kn,
            stride_vn,
            N_CTX,
            INNER_QK_SCALE,
            DIAG_OFFSET,
            BLOCK_N,
            BUF_DEPTH,
            USE_DIRECT_LOAD,
            True,
            IS_CAUSAL,
            False,
        )

    USE_FAST_SHORT_EPILOGUE: tl.constexpr = N_CTX <= 32 * BLOCK_N
    USE_WARP_LOCAL_OUTPUT: tl.constexpr = (IS_CAUSAL and N_CTX % BLOCK_M == 0 and N_CTX == 4096 and BLOCK_M == 256
                                           and (BLOCK_N == 32 or BLOCK_N == 64) and HEAD_DIM == 128
                                           and tlx.num_warps() == 8 and Out.dtype.element_ty == tl.bfloat16)
    NORMALIZED_IN_PRUNED_SHORT: tl.constexpr = SPLIT_PRUNED_CAUSAL_PREFIX
    acc = state.acc
    if not NORMALIZED_IN_PRUNED_SHORT:
        if USE_FAST_SHORT_EPILOGUE:
            # Output is rounded to FP16/BF16, so the native reciprocal's error
            # is below the store precision while avoiding the corrected IEEE
            # divide.
            l_recip = fast_dividef(1.0, state.l_i)
            acc = state.acc * l_recip[:, None]
        else:
            acc = state.acc / state.l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    out = acc.to(Out.dtype.element_ty)
    if USE_Q_LDS:
        # Swap the native MFMA column-4/column-8 ownership bits in-wave.  Each
        # lane then owns eight adjacent values for a 128-bit store, without
        # the full-tile LDS transpose selected by generic coalescing.
        out_layout: tl.constexpr = tlx.layout(
            shape=((32, 2, 4), (8, 8)),
            stride=((128, 8, 4096), (1, 16)),
        )
        o_ptrs = tlx.require_layout(o_ptrs, out_layout)
        out = tlx.require_layout(out, out_layout)
    elif USE_WARP_LOCAL_OUTPUT:
        # Eight-warp D128 store mapping: five row bits and one D8 lane bit
        # belong to each wave, while three additional row bits select the
        # eight-warp MFMA groups.  This is the shape/stride form of the
        # reference's in-wave output permutation.
        out_layout: tl.constexpr = tlx.layout(
            shape=((32, 2, 8), (8, 8)),
            stride=((128, 8, 4096), (1, 16)),
        )
        o_ptrs = tlx.require_layout(o_ptrs, out_layout)
        out = tlx.require_layout(out, out_layout)
    if N_CTX % BLOCK_M == 0:
        tl.store(o_ptrs, out)
    else:
        tl.store(o_ptrs, out, mask=offs_m[:, None] < N_CTX)


@triton.jit
def _cluster_causal_query_tile(raw_pid_m, N_CTX: tl.constexpr, BLOCK_M: tl.constexpr):
    # Later causal query tiles have more K/V blocks to process. Reverse the
    # complete program-id range so that those heavier CTAs are dispatched first.
    NUM_M_BLOCKS: tl.constexpr = (N_CTX + BLOCK_M - 1) // BLOCK_M
    return NUM_M_BLOCKS - 1 - raw_pid_m


@triton.jit
def _cluster_direct_workgroup_window(
    raw_off_h,
    raw_pid_m,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_24_HEAD_WINDOW: tl.constexpr,
    USE_FACTORED_24: tl.constexpr,
):
    """Transpose bounded head/query windows without changing the work set."""
    NUM_M_BLOCKS: tl.constexpr = (N_CTX + BLOCK_M - 1) // BLOCK_M
    if IS_CAUSAL:
        # TLX retains the measured H64 24/24/16 long-row phase. The Gluon
        # 32-head causal window does not improve this pipeline.
        WINDOW_HEADS: tl.constexpr = 24
        USE_WINDOW: tl.constexpr = (BLOCK_M == 256 and NUM_M_BLOCKS >= 8 and USE_24_HEAD_WINDOW and WINDOW_HEADS < H
                                    and (H % WINDOW_HEADS == 0 or H == 64))
    else:
        # Keep a 128-CTA locality set for the N2048/N4096 D128 classes where
        # this TLX pipeline benefits from head/query transposition.
        WINDOW_HEADS: tl.constexpr = max(1, 128 // NUM_M_BLOCKS)
        USE_WINDOW: tl.constexpr = (BLOCK_M == 256 and NUM_M_BLOCKS >= 8 and NUM_M_BLOCKS <= 16 and WINDOW_HEADS < H
                                    and H % WINDOW_HEADS == 0)

    off_h = raw_off_h
    pid_m = raw_pid_m
    if USE_WINDOW:
        linear_wg = raw_off_h + raw_pid_m * H
        GROUP_SPAN: tl.constexpr = WINDOW_HEADS * NUM_M_BLOCKS
        if H % WINDOW_HEADS == 0:
            within_group = linear_wg % GROUP_SPAN
            off_h = (linear_wg // GROUP_SPAN) * WINDOW_HEADS + within_group % WINDOW_HEADS
            pid_m = within_group // WINDOW_HEADS
        else:
            # H64 is partitioned into two 24-head windows and one 16-head tail.
            # Decode each range separately to keep this a bijection.
            TAIL_HEADS: tl.constexpr = H - 2 * WINDOW_HEADS
            if USE_FACTORED_24:
                # Match the Gluon H64/G24 decoder.  Factoring 24 as 8*3
                # avoids carrying the large-span quotient through the hot
                # FP16 kernel while preserving the same 24/24/16 bijection.
                FULL_GROUP_ROWS: tl.constexpr = 3 * NUM_M_BLOCKS // 8
                TAIL_ROW_BEGIN: tl.constexpr = 2 * FULL_GROUP_ROWS
                if raw_pid_m < FULL_GROUP_ROWS:
                    packed = raw_off_h + raw_pid_m * H
                    packed8 = packed // 8
                    mapped_m = packed8 // 3
                    off_h = packed % 8 + (packed8 - mapped_m * 3) * 8
                    pid_m = mapped_m
                elif raw_pid_m < TAIL_ROW_BEGIN:
                    packed = raw_off_h + (raw_pid_m - FULL_GROUP_ROWS) * H
                    packed8 = packed // 8
                    mapped_m = packed8 // 3
                    off_h = WINDOW_HEADS + packed % 8 + (packed8 - mapped_m * 3) * 8
                    pid_m = mapped_m
                else:
                    packed = raw_off_h + (raw_pid_m - TAIL_ROW_BEGIN) * H
                    off_h = 2 * WINDOW_HEADS + packed % TAIL_HEADS
                    pid_m = packed // TAIL_HEADS
            else:
                FULL_HEADS: tl.constexpr = 2 * WINDOW_HEADS
                FULL_SPAN: tl.constexpr = FULL_HEADS * NUM_M_BLOCKS
                if linear_wg < FULL_SPAN:
                    within_group = linear_wg % GROUP_SPAN
                    off_h = (linear_wg // GROUP_SPAN) * WINDOW_HEADS + within_group % WINDOW_HEADS
                    pid_m = within_group // WINDOW_HEADS
                else:
                    within_tail = linear_wg - FULL_SPAN
                    off_h = FULL_HEADS + within_tail % TAIL_HEADS
                    pid_m = within_tail // TAIL_HEADS
    return off_h, pid_m


@triton.jit
def _attn_fwd_cluster_pipeline(
    Q,
    K,
    V,
    Out,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm,
    stride_qk,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn,
    stride_kk,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn,
    stride_vk,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om,
    stride_ok,
    Z,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_DIRECT_LOAD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    _assume_strides(
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
    )

    if IS_CAUSAL:
        raw_off_h = tl.program_id(0)
        raw_pid_m = tl.program_id(1)
    else:
        raw_pid_m = tl.program_id(0)
        raw_off_h = tl.program_id(1)
    NUM_M_BLOCKS: tl.constexpr = (N_CTX + BLOCK_M - 1) // BLOCK_M
    USE_24_HEAD_WINDOW: tl.constexpr = (IS_CAUSAL and H == 64
                                        and ((Q.dtype.element_ty == tl.float16 and NUM_M_BLOCKS >= 32) or
                                             (Q.dtype.element_ty == tl.bfloat16 and NUM_M_BLOCKS >= 64)))
    USE_FACTORED_24: tl.constexpr = (USE_24_HEAD_WINDOW and Q.dtype.element_ty == tl.float16 and NUM_M_BLOCKS % 8 == 0)
    off_h, pid_m = _cluster_direct_workgroup_window(
        raw_off_h,
        raw_pid_m,
        H,
        N_CTX,
        BLOCK_M,
        IS_CAUSAL,
        USE_24_HEAD_WINDOW,
        USE_FACTORED_24,
    )
    if IS_CAUSAL:
        pid_m = _cluster_causal_query_tile(pid_m, N_CTX, BLOCK_M)
    off_z = tl.program_id(2)

    # Only the aligned LDS diagonal specialization consumes four unique slots.
    # Direct-load and ragged tiles use the regular two-slot ring.
    FOUR_SLOT_PRUNE: tl.constexpr = (not USE_DIRECT_LOAD and IS_CAUSAL and BLOCK_M == 256 and BLOCK_N == 64
                                     and HEAD_DIM == 128 and tlx.num_warps() == 8 and N_CTX % BLOCK_M == 0
                                     and N_CTX % BLOCK_N == 0)
    ALLOC_BUF_DEPTH: tl.constexpr = 4 if FOUR_SLOT_PRUNE else BUF_DEPTH
    USE_TARGET_PADDED_KV: tl.constexpr = (HEAD_DIM == 128 and (BLOCK_N == 32 or BLOCK_N == 64) and tlx.num_warps() == 8)
    if USE_TARGET_PADDED_KV:
        if BLOCK_N == 64:
            kv_offset_bases: tl.constexpr = [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [16, 0],
                [32, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ]
        else:
            kv_offset_bases: tl.constexpr = [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [16, 0],
                [8, 0],
                [1, 0],
                [2, 0],
                [4, 0],
            ]
        k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 8)], kv_offset_bases,
                                                                              [BLOCK_N, HEAD_DIM])
        v_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 32)], kv_offset_bases,
                                                                              [BLOCK_N, HEAD_DIM])
        k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, ALLOC_BUF_DEPTH, layout=k_layout)
        v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, ALLOC_BUF_DEPTH, layout=v_layout)
    else:
        k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, ALLOC_BUF_DEPTH)
        v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, ALLOC_BUF_DEPTH)
    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089
    _attn_cluster_tile(
        pid_m,
        off_z,
        off_h,
        Q,
        K,
        V,
        Out,
        k_buf,
        v_buf,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        N_CTX,
        QK_SCALE,
        BLOCK_M,
        BLOCK_N,
        BUF_DEPTH,
        HEAD_DIM,
        USE_DIRECT_LOAD,
        False,
        IS_CAUSAL,
    )


@triton.jit
def _attn_fwd_cluster_short_causal_pipeline(
    Q,
    K,
    V,
    Out,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_ok: tl.constexpr,
    Z,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_DIRECT_LOAD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SPECIALIZE_QUERY_CLASSES: tl.constexpr,
):
    tl.static_assert(IS_CAUSAL)
    tl.static_assert(HEAD_DIM == 128)
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_N == 64)
    tl.static_assert(N_CTX <= 1024)
    tl.static_assert(N_CTX % BLOCK_M == 0)

    _assume_strides(
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
    )

    off_h = tl.program_id(0)
    query_class = tl.program_id(1)
    off_z = tl.program_id(2)
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, BUF_DEPTH)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, BUF_DEPTH)
    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089
    NUM_M_BLOCKS: tl.constexpr = N_CTX // BLOCK_M

    if SPECIALIZE_QUERY_CLASSES:
        # All classes share one launch so heavy and light CTAs stay interleaved.
        # Each branch sees a constant query tile and drops unreachable work.
        for i in tl.static_range(NUM_M_BLOCKS):
            if query_class == i:
                pid_m: tl.constexpr = NUM_M_BLOCKS - 1 - i
                _attn_cluster_tile(
                    pid_m,
                    off_z,
                    off_h,
                    Q,
                    K,
                    V,
                    Out,
                    k_buf,
                    v_buf,
                    stride_qz,
                    stride_qh,
                    stride_qm,
                    stride_qk,
                    stride_kz,
                    stride_kh,
                    stride_kn,
                    stride_kk,
                    stride_vz,
                    stride_vh,
                    stride_vn,
                    stride_vk,
                    stride_oz,
                    stride_oh,
                    stride_om,
                    stride_ok,
                    N_CTX,
                    QK_SCALE,
                    BLOCK_M,
                    BLOCK_N,
                    BUF_DEPTH,
                    HEAD_DIM,
                    USE_DIRECT_LOAD,
                    True,
                    IS_CAUSAL,
                )
    else:
        # Cloning eight N=1024 classes raises the unified register allocation
        # past the two-wave boundary.  Keep the heavy-first mapping but share
        # one dynamic tile body, as in the optimized source kernel.
        pid_m = NUM_M_BLOCKS - 1 - query_class
        _attn_cluster_tile(
            pid_m,
            off_z,
            off_h,
            Q,
            K,
            V,
            Out,
            k_buf,
            v_buf,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_oz,
            stride_oh,
            stride_om,
            stride_ok,
            N_CTX,
            QK_SCALE,
            BLOCK_M,
            BLOCK_N,
            BUF_DEPTH,
            HEAD_DIM,
            USE_DIRECT_LOAD,
            True,
            IS_CAUSAL,
        )


@triton.jit
def _attn_fwd_cluster_persistent_pipeline(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_DIRECT_LOAD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    _assume_strides(
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
    )

    # Persistent scheduler: pin flattened (batch, head) work to an XCD for K/V
    # locality, then round-robin constant-cost work units across local programs.
    # Causal units bundle zig-zag tile pairs; non-causal units are one tile.
    pid = tl.program_id(0)
    xcd = pid % NUM_XCDS
    local = pid // NUM_XCDS
    NUM_LOCAL: tl.constexpr = NUM_SMS // NUM_XCDS

    # Match the non-persistent launcher: only the aligned LDS diagonal consumes
    # four slots; direct-load and ragged tiles retain the normal ring depth.
    FOUR_SLOT_PRUNE: tl.constexpr = (not USE_DIRECT_LOAD and IS_CAUSAL and BLOCK_M == 256 and BLOCK_N == 64
                                     and HEAD_DIM == 128 and tlx.num_warps() == 8 and N_CTX % BLOCK_M == 0
                                     and N_CTX % BLOCK_N == 0)
    ALLOC_BUF_DEPTH: tl.constexpr = 4 if FOUR_SLOT_PRUNE else BUF_DEPTH
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, ALLOC_BUF_DEPTH)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, ALLOC_BUF_DEPTH)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089
    TILES_PER_UNIT: tl.constexpr = 2 if IS_CAUSAL else 1
    units_per_hz: tl.constexpr = (NUM_M_BLOCKS + TILES_PER_UNIT - 1) // TILES_PER_UNIT
    hz_per_xcd = (Z * H + NUM_XCDS - 1) // NUM_XCDS
    units = hz_per_xcd * units_per_hz

    for unit in tl.range(local, units, NUM_LOCAL, num_stages=1):
        local_hz = unit // units_per_hz
        bundle = unit % units_per_hz
        pid_hz = xcd + local_hz * NUM_XCDS
        if pid_hz < Z * H:
            off_z = pid_hz // H
            off_h = pid_hz % H
            for j in tl.static_range(TILES_PER_UNIT):
                idx = bundle * TILES_PER_UNIT + j
                if idx < NUM_M_BLOCKS:
                    if IS_CAUSAL:
                        half = idx // 2
                        pid_m = tl.where(idx % 2 == 0, half, NUM_M_BLOCKS - 1 - half)
                    else:
                        pid_m = idx
                    # Safe to reuse the LDS slots across units: the outer loop
                    # has num_stages=1 and _attn_cluster_tile drains all async
                    # load groups before it returns.
                    _attn_cluster_tile(
                        pid_m,
                        off_z,
                        off_h,
                        Q,
                        K,
                        V,
                        Out,
                        k_buf,
                        v_buf,
                        stride_qz,
                        stride_qh,
                        stride_qm,
                        stride_qk,
                        stride_kz,
                        stride_kh,
                        stride_kn,
                        stride_kk,
                        stride_vz,
                        stride_vh,
                        stride_vn,
                        stride_vk,
                        stride_oz,
                        stride_oh,
                        stride_om,
                        stride_ok,
                        N_CTX,
                        QK_SCALE,
                        BLOCK_M,
                        BLOCK_N,
                        BUF_DEPTH,
                        HEAD_DIM,
                        USE_DIRECT_LOAD,
                        False,
                        IS_CAUSAL,
                    )


# Short cluster kernels can exhaust ROCm event resources in the entropy
# benchmarker. Use the standard benchmarker for this small two-config sweep.
_attn_fwd_cluster_pipeline_autotuned = triton.autotune(
    configs=_cluster_short_load_configs(),
    key=_CLUSTER_AUTOTUNE_KEY,
    prune_configs_by={"early_config_prune": _prune_cluster_short_load_configs},
    do_bench=triton.testing.do_bench,
)(_attn_fwd_cluster_pipeline)

_attn_fwd_cluster_persistent_pipeline_autotuned = triton.autotune(
    configs=_cluster_short_load_configs(),
    key=_CLUSTER_PERSISTENT_AUTOTUNE_KEY,
    prune_configs_by={"early_config_prune": _prune_cluster_short_load_configs},
    do_bench=triton.testing.do_bench,
)(_attn_fwd_cluster_persistent_pipeline)


def _cluster_default_block_n(causal):
    return 64


def _validate_cluster_inputs(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("cluster attention expects rank-4 [B, H, N, D] inputs")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("cluster attention currently requires Q, K, and V to have the same shape")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("cluster attention requires Q, K, and V to have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"cluster attention supports only FP16/BF16, got {q.dtype}")
    if q.device != k.device or q.device != v.device or not q.is_cuda:
        raise ValueError("cluster attention requires Q, K, and V on the same GPU")
    if any(size <= 0 for size in q.shape):
        raise ValueError(f"cluster attention dimensions must be positive, got {tuple(q.shape)}")
    if q.shape[-1] not in (64, 128):
        raise ValueError(f"cluster attention supports head dimensions 64 and 128, got {q.shape[-1]}")
    for name, tensor in (("Q", q), ("K", k), ("V", v)):
        if tensor.stride(2) <= 0 or tensor.stride(3) < 0:
            raise ValueError(f"cluster attention requires nonnegative {name} feature and positive sequence strides")


def _validate_cluster_tiles(block_m, block_n):
    if block_m not in (128, 256):
        raise ValueError(f"cluster attention supports BLOCK_M 128 or 256, got {block_m}")
    if block_n not in (32, 64):
        raise ValueError(f"cluster attention supports BLOCK_N 32 or 64, got {block_n}")


def flash_attn_cluster_pipeline(q, k, v, sm_scale, causal=False, **kw):
    _validate_cluster_inputs(q, k, v)
    B, H, N_CTX, D = q.shape
    use_short_causal_defaults = (causal and D == 128 and N_CTX <= 1024 and N_CTX % 128 == 0 and "BLOCK_M" not in kw
                                 and "BLOCK_N" not in kw and kw.get("num_warps", 4) == 4
                                 and kw.get("num_stages", 3) == 3)
    mfma_m = 32
    block_m = kw.pop("BLOCK_M", 128 if use_short_causal_defaults else 256)
    block_n = kw.pop(
        "BLOCK_N",
        64 if use_short_causal_defaults and block_m == 128 else _cluster_default_block_n(causal),
    )
    _validate_cluster_tiles(block_m, block_n)
    has_explicit_num_warps = "num_warps" in kw
    has_explicit_waves_per_eu = "waves_per_eu" in kw
    num_warps = kw.pop("num_warps", min(8, max(1, block_m // mfma_m)))
    waves_per_eu = kw.pop(
        "waves_per_eu",
        2 if (not causal or (block_m == 256 and num_warps == 8)) else 0,
    )
    use_direct_load = kw.pop("USE_DIRECT_LOAD", None)
    o = torch.empty_like(q)
    m_blocks = triton.cdiv(N_CTX, block_m)
    grid = (H, m_blocks, B) if causal else (m_blocks, H, B)
    use_autotune = (use_direct_load is None and not has_explicit_num_warps and not has_explicit_waves_per_eu and not kw)
    use_pruned_n2048_defaults = (use_autotune and causal and D == 128 and N_CTX == 2048 and block_m == 256
                                 and block_n == 32 and num_warps == 8)
    if use_pruned_n2048_defaults:
        # The register-predicated diagonal has no direct-load peer to tune and
        # its spill-free object wins with the target's stage-two plan.
        use_autotune = False
    # The class-specialized async schedule is validated for exactly four waves
    # and three compiler stages; explicit alternatives retain the general path.
    use_short_causal_classes = (causal and D == 128 and N_CTX <= 1024 and N_CTX % 128 == 0 and block_m == 128
                                and block_n == 64 and num_warps == 4 and kw.get("num_stages", 3) == 3)
    if use_short_causal_classes:
        kernel = _attn_fwd_cluster_short_causal_pipeline
        use_autotune = False
    else:
        kernel = _attn_fwd_cluster_pipeline_autotuned if use_autotune else _attn_fwd_cluster_pipeline
    launch_meta = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BUF_DEPTH": CLUSTER_BUF_DEPTH,
        "HEAD_DIM": D,
        "IS_CAUSAL": causal,
        "waves_per_eu": waves_per_eu,
        **kw,
    }
    # This port must remain correct and performant with the stock LLVM
    # scheduler.  Keep Triton's experimental sched-group pass opt-in even if
    # the process-wide environment knob is enabled.
    launch_meta.setdefault("enable_sched_group_barrier_scheduler", False)
    if use_short_causal_classes:
        launch_meta.setdefault("num_stages", 3)
        launch_meta["SPECIALIZE_QUERY_CLASSES"] = N_CTX <= 512
        if N_CTX == 512:
            launch_meta.setdefault("llvm_fn_attrs", _CLUSTER_SHORT_N512_LLVM_FN_ATTRS)
        elif N_CTX == 1024:
            launch_meta.setdefault("llvm_fn_attrs", _CLUSTER_SHORT_N1024_LLVM_FN_ATTRS)
    elif D == 128 and block_m == 256 and num_warps == 8:
        # Keeping both MFMA accumulators in VGPRs removes mixed AGPR/VGPR
        # handoffs; the full-attention object also becomes spill-free.
        launch_meta.setdefault(
            "llvm_fn_attrs",
            _CLUSTER_VGPR_ONLY_LLVM_FN_ATTRS,
        )
    if causal and D == 128 and block_m == 256 and block_n == 32 and N_CTX == 4096:
        # Reverse assignment wins the N4096 causal object. The spill-free
        # N2048 pruned object instead selects the forward stage-two plan above.
        launch_meta.setdefault("reverse_local_assignment", True)
    if use_pruned_n2048_defaults:
        launch_meta.setdefault("num_stages", 2)
    if not use_autotune:
        launch_meta.update({
            "USE_DIRECT_LOAD": False if use_direct_load is None else use_direct_load,
            "num_warps": num_warps,
        })
    kernel[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B,
        H,
        N_CTX,
        sm_scale,
        **launch_meta,
    )
    return o


def flash_attn_cluster_persistent_pipeline(q, k, v, sm_scale, causal=False, **kw):
    _validate_cluster_inputs(q, k, v)
    mfma_m = 32
    block_m = kw.pop("BLOCK_M", 256)
    block_n = kw.pop("BLOCK_N", _cluster_default_block_n(causal))
    _validate_cluster_tiles(block_m, block_n)
    has_explicit_num_warps = "num_warps" in kw
    has_explicit_waves_per_eu = "waves_per_eu" in kw
    num_warps = kw.pop("num_warps", min(8, max(1, block_m // mfma_m)))
    waves_per_eu = kw.pop(
        "waves_per_eu",
        2 if (not causal or (block_m == 256 and num_warps == 8)) else 0,
    )
    use_direct_load = kw.pop("USE_DIRECT_LOAD", None)
    B, H, N_CTX, D = q.shape
    num_xcds = kw.pop("NUM_XCDS", 8)
    if num_xcds <= 0:
        raise ValueError(f"cluster attention: NUM_XCDS must be positive, got {num_xcds}")
    cu_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_sms = kw.pop("NUM_SMS", (cu_count // num_xcds) * num_xcds)
    if num_sms < num_xcds:
        raise ValueError(f"cluster attention: NUM_SMS ({num_sms}) must be >= NUM_XCDS ({num_xcds})")
    if num_sms % num_xcds != 0:
        raise ValueError(f"cluster attention: NUM_SMS ({num_sms}) must be divisible by NUM_XCDS ({num_xcds})")

    o = torch.empty_like(q)
    m_blocks = triton.cdiv(N_CTX, block_m)
    grid = (num_sms, )
    use_autotune = (use_direct_load is None and not has_explicit_num_warps and not has_explicit_waves_per_eu and not kw)
    kernel = (_attn_fwd_cluster_persistent_pipeline_autotuned
              if use_autotune else _attn_fwd_cluster_persistent_pipeline)
    launch_meta = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BUF_DEPTH": CLUSTER_BUF_DEPTH,
        "HEAD_DIM": D,
        "IS_CAUSAL": causal,
        "NUM_M_BLOCKS": m_blocks,
        "NUM_SMS": num_sms,
        "NUM_XCDS": num_xcds,
        "waves_per_eu": waves_per_eu,
        **kw,
    }
    launch_meta.setdefault("enable_sched_group_barrier_scheduler", False)
    if not causal and D == 128 and block_m == 256 and num_warps == 8:
        # The persistent causal schedule loses overlap with VGPR-only
        # allocation, while full attention benefits from removing AGPR moves.
        launch_meta.setdefault("llvm_fn_attrs", _CLUSTER_VGPR_ONLY_LLVM_FN_ATTRS)
    if not use_autotune:
        launch_meta.update({
            "USE_DIRECT_LOAD": False if use_direct_load is None else use_direct_load,
            "num_warps": num_warps,
        })
    kernel[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B,
        H,
        N_CTX,
        sm_scale,
        **launch_meta,
    )
    return o


def attention(q, k, v, sm_scale, causal, config=None):
    config = {} if config is None else dict(config)
    return flash_attn_cluster_pipeline(q, k, v, sm_scale, causal=causal, **config)


def persistent_attention(q, k, v, sm_scale, causal, config=None):
    config = {} if config is None else dict(config)
    return flash_attn_cluster_persistent_pipeline(q, k, v, sm_scale, causal=causal, **config)
