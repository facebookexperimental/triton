"""Compare generated (auto-emitted) vs hand-written WS perf for case7.

case7 = fused wgrad GEMM (dW = doutᵀ @ act) + bias reduce (db = dout.sum(0)).
Reports ms, TFLOPS (the GEMM's 2*K_out*N_in*M flops; the bias reduce is
negligible) and generated/hand-written ratio per shape. The hand-written
kernel mirrors the geo best-perf `addmm_1d_bias_reduce` structure, so this is
the apples-to-apples reference for the emitter's output.
"""

import sys

sys.path.insert(0, ".")

import generated
import handwritten as hw
import torch
import triton


def alloc_fn(s, a, st):
    return torch.empty(s, device="cuda", dtype=torch.int8)


BLOCK_KO, BLOCK_NI, BLOCK_M = 128, 128, 64
NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS = 3, 2


def _bench(call, n_warmup=10, n_iter=50):
    for _ in range(n_warmup):
        call()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter


def bench_gen(dout, act, dw, db, M, K_out, N_in, NUM_SMS):
    grid = (NUM_SMS,)

    def run():
        generated.wgrad_bias_nows[grid](
            dout,
            act,
            dw,
            db,
            M,
            K_out,
            N_in,
            num_warps=4,
            num_ctas=1,
            num_stages=2,
        )

    return _bench(run)


def bench_hw(dout, act, dw, db, M, K_out, N_in, NUM_SMS):
    grid = (NUM_SMS,)

    def run():
        hw.wgrad_bias_ws[grid](
            dout,
            act,
            dw,
            db,
            M,
            K_out,
            N_in,
            BLOCK_KO=BLOCK_KO,
            BLOCK_NI=BLOCK_NI,
            BLOCK_M=BLOCK_M,
            NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
            NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
            num_warps=4,
            num_ctas=1,
        )

    return _bench(run)


def main():
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    shapes = [
        (1024, 256, 256),
        (4096, 1024, 1024),
        (8192, 2048, 1024),
        (16384, 1024, 1024),
    ]
    print(
        f"{'Shape (M,Ko,Ni)':<22} {'HW ms':<10} {'HW TF':<8} "
        f"{'GEN ms':<10} {'GEN TF':<8} {'GEN/HW':<8}"
    )
    print("-" * 74)
    for M, K_out, N_in in shapes:
        dout = torch.randn(M, K_out, device="cuda", dtype=torch.float16)
        act = torch.randn(M, N_in, device="cuda", dtype=torch.float16)
        dw = torch.empty(K_out, N_in, device="cuda", dtype=torch.float16)
        db = torch.empty(K_out, device="cuda", dtype=torch.float32)
        ms_hw = bench_hw(dout, act, dw, db, M, K_out, N_in, NUM_SMS)
        ms_gen = bench_gen(dout, act, dw, db, M, K_out, N_in, NUM_SMS)
        flops = 2 * K_out * N_in * M
        tf_hw = flops / (ms_hw / 1000) / 1e12
        tf_gen = flops / (ms_gen / 1000) / 1e12
        ratio = tf_gen / tf_hw
        print(
            f"({M},{K_out},{N_in})".ljust(22)
            + f"{ms_hw:<10.4f}{tf_hw:<8.1f}{ms_gen:<10.4f}{tf_gen:<8.1f}{ratio:<8.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
