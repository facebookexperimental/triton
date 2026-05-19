import triton
import triton.language as tl


@triton.jit
def get_bufidx_phase(accum_cnt, NUM_BUFFERS: tl.constexpr):
    buf_idx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return buf_idx, phase
