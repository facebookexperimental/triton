// Minimal C ABI over aiter's existing ASM-only host library. No torch or CK.
#include "mha_bwd.h"

extern "C" float mi450_aiter_bwd(void *q, void *k, void *v, void *o, void *dout,
                                 void *lse, void *delta, void *dq_acc, void *dq,
                                 void *dk_expanded, void *dv_expanded,
                                 int batch, int hq, int hk, int n, float scale,
                                 void *stream) {
  aiter::mha_bwd_args a{};
  a.use_asm_v3 = true;
  a.v3_atomic_fp32 = true;
  a.v3_bf16_cvt = 0;
  a.hdim_q = a.hdim_v = 128;
  a.data_type = "bf16";
  a.batch = batch;
  a.nhead_q = hq;
  a.nhead_k = hk;
  a.seqlen_q = a.seqlen_k = a.max_seqlen_q = a.max_seqlen_k = n;
  a.window_size_left = a.window_size_right = -1;
  a.scale = scale;
  a.q_ptr = q;
  a.k_ptr = k;
  a.v_ptr = v;
  a.o_ptr = o;
  a.do_ptr = dout;
  a.lse_ptr = lse;
  a.d_ptr = delta;
  a.dq_ptr = dq;
  a.dk_ptr = dk_expanded;
  a.dv_ptr = dv_expanded;
  a.stride_q = a.stride_k = a.stride_v = a.stride_o = a.stride_do =
      a.stride_dq = a.stride_dk = a.stride_dv = 128;
  a.nhead_stride_q = a.nhead_stride_k = a.nhead_stride_v = a.nhead_stride_o =
      a.nhead_stride_do = a.nhead_stride_dq = a.nhead_stride_dk =
          a.nhead_stride_dv = n * 128;
  a.nhead_stride_lsed = n;
  a.batch_stride_q = a.batch_stride_o = a.batch_stride_do = a.batch_stride_dq =
      a.batch_stride_dk = a.batch_stride_dv = hq * n * 128;
  a.batch_stride_k = a.batch_stride_v = hk * n * 128;
  a.batch_stride_lsed = hq * n;
  auto hip_stream = reinterpret_cast<hipStream_t>(stream);
  a.workspace_alloc = [=](size_t bytes, bool zero) -> void * {
    if (bytes > size_t(batch) * hq * n * 128 * sizeof(float))
      throw std::runtime_error("unexpected aiter workspace size");
    if (zero) {
      auto status = hipMemsetAsync(dq_acc, 0, bytes, hip_stream);
      if (status != hipSuccess)
        throw std::runtime_error("failed to zero aiter dQ workspace");
    }
    return dq_acc;
  };
  ck_tile::stream_config config{};
  config.stream_id_ = hip_stream;
  return aiter::mha_bwd(a, config);
}
