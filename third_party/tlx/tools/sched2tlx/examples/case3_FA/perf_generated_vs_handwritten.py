"""Compare generated (auto-emitted) vs handwritten TLX FA fwd perf.

Sweeps the same 6 shapes as run_generated.py.  Reports ms, TFLOPS and
generated/handwritten ratio per shape so we can spot regressions after
schedule/emitter changes.
"""

import sys

sys.path.insert(0, ".")

import generated
import handwritten as hw
import torch
import triton


def alloc_fn(s, a, st):
    return torch.empty(s, device="cuda", dtype=torch.int8)


BLOCK_M, BLOCK_N, HEAD_DIM, NUM_BUFFERS_KV = 128, 64, 128, 2


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


def bench_gen(qf, kf, vf, of, m_lse, sm, Z, H, N_CTX):
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    def run():
        generated.fa_fwd_kernel_nows[grid](
            qf,
            kf,
            vf,
            of,
            m_lse,
            sm,
            Z * H,
            N_CTX,
            num_warps=4,
            num_ctas=1,
            num_stages=2,
        )

    return _bench(run)


def bench_hw(qf, kf, vf, of, m_lse, sm, Z, H, N_CTX):
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    def run():
        hw.fa_fwd_kernel[grid](
            qf,
            kf,
            vf,
            of,
            m_lse,
            sm,
            Z,
            H,
            N_CTX,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM,
            NUM_BUFFERS_KV=NUM_BUFFERS_KV,
            num_warps=4,
            num_ctas=1,
            num_stages=2,
        )

    return _bench(run)


def main():
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    shapes = [
        (1, 4, 512),
        (1, 8, 1024),
        (2, 16, 2048),
        (1, 16, 4096),
        (2, 16, 4096),
        (1, 32, 8192),
    ]
    print(
        f"{'Shape':<18} {'HW ms':<10} {'HW TF':<8} {'GEN ms':<10} {'GEN TF':<8} {'GEN/HW':<8}"
    )
    print("-" * 70)
    for Z, H, N_CTX in shapes:
        q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(q)
        m_lse = torch.empty(Z * H, N_CTX, device="cuda", dtype=torch.float32)
        sm = 1.0 / (HEAD_DIM**0.5)
        qf = q.contiguous().view(-1, HEAD_DIM)
        kf = k.contiguous().view(-1, HEAD_DIM)
        vf = v.contiguous().view(-1, HEAD_DIM)
        of = out.view(-1, HEAD_DIM)
        ms_hw = bench_hw(qf, kf, vf, of, m_lse, sm, Z, H, N_CTX)
        ms_gen = bench_gen(qf, kf, vf, of, m_lse, sm, Z, H, N_CTX)
        flops = 4 * Z * H * N_CTX * N_CTX * HEAD_DIM
        tf_hw = flops / (ms_hw / 1000) / 1e12
        tf_gen = flops / (ms_gen / 1000) / 1e12
        ratio = tf_gen / tf_hw
        print(
            f"({Z},{H},{N_CTX:<5}){'':<3}{ms_hw:<10.4f}{tf_hw:<8.1f}{ms_gen:<10.4f}{tf_gen:<8.1f}{ratio:<8.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
