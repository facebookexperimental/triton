// Minimal gfx950 repro: v_pk_add_f32 with op_sel on src1 returns a wrong low result
// while another wave on the same SIMD executes v_mfma_f32_16x16x32_f16.
//
// Each workgroup has 8 waves. Waves are placed round-robin on the CU's 4 SIMDs, so
// SIMD k hosts wave k and wave k+4. Waves 0-3 ("MFMA waves") loop on MFMAs; waves 4-7
// ("ADD waves") loop on the packed add and compare the low half against a scalar subtract.
//
// Modes for the MFMA waves:
//   0: v_mfma_f32_16x16x32_f16  (expected to trigger)
//   1: v_mfma_f32_16x16x16_f16  (control)
//   2: plain VALU v_mov loop    (control)
//   3: idle (exit immediately)  (control)
// Forms for the ADD waves: 0 failing op_sel form, 1 commuted operands, 2 no op_sel.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatx4 __attribute__((ext_vector_type(4)));
typedef float floatx2 __attribute__((ext_vector_type(2)));
typedef int intx4 __attribute__((ext_vector_type(4)));
typedef int intx2 __attribute__((ext_vector_type(2)));

#define CHECK(x)                                                                 \
  do {                                                                           \
    hipError_t e = (x);                                                          \
    if (e != hipSuccess) {                                                       \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

struct Failure {
  int block, wave, lane, iter;
  float src0_lo, src1_hi, got_lo, want_lo, got_hi, want_hi;
};

__global__ __launch_bounds__(512) void repro(int mode, int form, int mfma_iters, int add_iters,
                                             unsigned long long* lane_mask, int* fail_count,
                                             Failure* first, float* sink) {
  const int wave = threadIdx.x / 64;
  const int lane = threadIdx.x % 64;

  if (wave < 4) {
    if (mode == 3) return;
    floatx4 acc0 = {0, 0, 0, 0}, acc1 = acc0, acc2 = acc0, acc3 = acc0;
    intx4 a = {lane, lane + 1, lane + 2, lane + 3};
    intx4 b = {1, 2, 3, 4};
    intx2 a2 = {lane, lane + 1};
    intx2 b2 = {1, 2};
    for (int i = 0; i < mfma_iters; ++i) {
      if (mode == 0) {
        asm volatile(
            "v_mfma_f32_16x16x32_f16 %0, %4, %5, %0\n"
            "v_mfma_f32_16x16x32_f16 %1, %4, %5, %1\n"
            "v_mfma_f32_16x16x32_f16 %2, %4, %5, %2\n"
            "v_mfma_f32_16x16x32_f16 %3, %4, %5, %3\n"
            : "+v"(acc0), "+v"(acc1), "+v"(acc2), "+v"(acc3)
            : "v"(a), "v"(b));
      } else if (mode == 1) {
        asm volatile(
            "v_mfma_f32_16x16x16_f16 %0, %4, %5, %0\n"
            "v_mfma_f32_16x16x16_f16 %1, %4, %5, %1\n"
            "v_mfma_f32_16x16x16_f16 %2, %4, %5, %2\n"
            "v_mfma_f32_16x16x16_f16 %3, %4, %5, %3\n"
            : "+v"(acc0), "+v"(acc1), "+v"(acc2), "+v"(acc3)
            : "v"(a2), "v"(b2));
      } else {
        asm volatile(
            "v_mov_b32 %0, %4\n"
            "v_mov_b32 %1, %4\n"
            "v_mov_b32 %2, %4\n"
            "v_mov_b32 %3, %4\n"
            : "+v"(acc0[0]), "+v"(acc1[0]), "+v"(acc2[0]), "+v"(acc3[0])
            : "v"((float)i));
      }
    }
    asm volatile("s_nop 7\n s_nop 7" ::: "memory");
    sink[blockIdx.x * 512 + threadIdx.x] = acc0[0] + acc1[1] + acc2[2] + acc3[3];
    return;
  }

  // ADD waves: same operand form as the failing kernel instruction
  //   v_pk_add_f32 v[116:117], v[98:99], v[72:73] op_sel:[0,1] neg_lo:[0,1] neg_hi:[0,1]
  // i.e. lo = src0.lo - src1.hi, hi = src0.hi - src1.hi.
  const float qlo = 110.0f + lane;
  const float qhi = 253.0f + lane;
  const float mlo = 0.0f;
  const float mhi = 640.0f + 0.25f * lane;
  const float want_lo = qlo - mhi;
  const float want_hi = qhi - mhi;
  floatx2 src0 = {qlo, qhi};
  floatx2 src1 = {mlo, mhi};
  int bad = 0;
  for (int i = 0; i < add_iters; ++i) {
    floatx2 r;
    if (form == 0) {
      // Failing form from the kernel.
      asm volatile("v_pk_add_f32 %0, %1, %2 op_sel:[0,1] neg_lo:[0,1] neg_hi:[0,1]"
                   : "=v"(r)
                   : "v"(src0), "v"(src1));
    } else if (form == 1) {
      // Same arithmetic with operands commuted.
      asm volatile("v_pk_add_f32 %0, %2, %1 op_sel:[1,0] neg_lo:[1,0] neg_hi:[1,0]"
                   : "=v"(r)
                   : "v"(src0), "v"(src1));
    } else {
      // Same arithmetic without op_sel: src1 = {mhi, mhi}.
      floatx2 s1 = {mhi, mhi};
      asm volatile("v_pk_add_f32 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                   : "=v"(r)
                   : "v"(src0), "v"(s1));
    }
    if (r[0] != want_lo || r[1] != want_hi) {
      if (!bad) {
        atomicOr(lane_mask, 1ull << lane);
        if (atomicAdd(fail_count, 1) == 0) {
          *first = Failure{(int)blockIdx.x, wave, lane, i, qlo, mhi, r[0], want_lo, r[1], want_hi};
        }
      }
      bad = 1;
    }
  }
}

// Usage: repro [mode=0] [launches=50] [blocks=1024] [mfma_iters=4096] [add_iters=65536] [form=0]
int main(int argc, char** argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0;
  int launches = argc > 2 ? atoi(argv[2]) : 50;
  int blocks = argc > 3 ? atoi(argv[3]) : 1024;
  int mfma_iters = argc > 4 ? atoi(argv[4]) : 4096;
  int add_iters = argc > 5 ? atoi(argv[5]) : 65536;
  int form = argc > 6 ? atoi(argv[6]) : 0;

  unsigned long long* d_mask;
  int* d_count;
  Failure* d_first;
  float* d_sink;
  CHECK(hipMalloc(&d_mask, sizeof(*d_mask)));
  CHECK(hipMalloc(&d_count, sizeof(*d_count)));
  CHECK(hipMalloc(&d_first, sizeof(*d_first)));
  CHECK(hipMalloc(&d_sink, sizeof(float) * blocks * 512));

  const char* names[] = {"mfma_16x16x32_f16", "mfma_16x16x16_f16", "valu_mov", "idle"};
  int bad_launches = 0;
  long long total_bad_lanes = 0;
  unsigned long long mask_all = 0;
  Failure first{};
  bool have_first = false;
  for (int l = 0; l < launches; ++l) {
    CHECK(hipMemset(d_mask, 0, sizeof(*d_mask)));
    CHECK(hipMemset(d_count, 0, sizeof(*d_count)));
    hipLaunchKernelGGL(repro, dim3(blocks), dim3(512), 0, 0, mode, form, mfma_iters, add_iters, d_mask,
                       d_count, d_first, d_sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    unsigned long long mask;
    int count;
    CHECK(hipMemcpy(&mask, d_mask, sizeof(mask), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&count, d_count, sizeof(count), hipMemcpyDeviceToHost));
    if (count) {
      ++bad_launches;
      total_bad_lanes += count;
      mask_all |= mask;
      if (!have_first) {
        CHECK(hipMemcpy(&first, d_first, sizeof(first), hipMemcpyDeviceToHost));
        have_first = true;
      }
    }
  }
  printf("mode=%d (%s) form=%d blocks=%d mfma_iters=%d add_iters=%d: bad launches %d/%d, bad lane-waves %lld, lane mask 0x%016llx\n",
         mode, names[mode], form, blocks, mfma_iters, add_iters, bad_launches, launches, total_bad_lanes,
         mask_all);
  if (have_first) {
    printf("  first: block %d wave %d lane %d iter %d: src0.lo=%g src1.hi=%g got lo=%g (want %g) hi=%g (want %g)\n",
           first.block, first.wave, first.lane, first.iter, first.src0_lo, first.src1_hi,
           first.got_lo, first.want_lo, first.got_hi, first.want_hi);
  }
  return bad_launches ? 2 : 0;
}
