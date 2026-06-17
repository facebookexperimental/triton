"""Single (K,) runner with a PINNED config — reproducible Phase 3 perf check.

Identical kernel to `_one_run_envcompare.py` but without `triton.autotune`.
The config is hard-coded to BM=BN=256, BK=64, num_warps=8, num_stages=2,
which is the config that delivered the +4.6 % pinned win at K=8192 in
Phase 4's coverage matrix.

Without pinning, the autotuner can pick BK=32,W=4 etc. on different
runs, which makes baseline vs APPLY non-comparable.

Usage:
    python _pinned_run.py <K>

Prints:
    OK <tflops>
or:
    FAIL <reason>

⚠ DO NOT pair this with TRITON_ALWAYS_COMPILE=1 — that flag forces a
fresh JIT compile per kernel launch, so do_bench's measured time
becomes dominated by ~30 ms of compile time and you'll see ~30 TF
instead of ~1000 TF. The env-var-gated codegen path is already
selected correctly because this script runs as a fresh process and
the JIT cache is keyed on env-var state.

Correct invocation:
    HIP_VISIBLE_DEVICES=0 python _pinned_run.py 8192
    # → OK ~1055  (baseline)
    TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 \\
      HIP_VISIBLE_DEVICES=0 python _pinned_run.py 8192
    # → OK ~1104  (Phase 3 default, +4.6 %)
"""
import sys, os, torch, triton, triton.language as tl

if os.environ.get("TRITON_ALWAYS_COMPILE"):
    print("FAIL TRITON_ALWAYS_COMPILE=1 is set — that forces a fresh JIT "
          "compile per kernel launch and turns do_bench into a ~30 ms "
          "compile-cost benchmark (you'd see ~30 TF instead of ~1000 TF). "
          "Unset it: `unset TRITON_ALWAYS_COMPILE` and rerun.")
    sys.exit(1)

K = int(sys.argv[1])
M = N = 4096

# Pinned config — the one that gives the +11 % win at K=8192.
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
GROUP_M = 8
NUM_WARPS = 8
NUM_STAGES = 2


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    pid_m = group_id * GROUP_M + (pid % GROUP_M)
    pid_n = (pid % width) // GROUP_M

    off_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    off_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    off_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + off_am[:, None] * sam + off_k[None, :] * sak
    b_ptrs = B + off_k[:, None] * sbk + off_bn[None, :] * sbn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=off_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=off_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk

    c = acc.to(tl.float16)
    off_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + off_cm[:, None] * scm + off_cn[None, :] * scn
    mask = (off_cm[:, None] < M) & (off_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def run():
    a = torch.randn((M, K), device="cuda:0", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda:0", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda:0", dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    def fn():
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

    fn()  # warmup / JIT
    ms = triton.testing.do_bench(fn)
    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)

    # Numerical sanity: compare to PyTorch reference (relaxed tol; tl.dot
    # on AMD MFMA accumulates in fp32 but stores fp16, ~1e-2 tolerance).
    ref = a @ b
    if not torch.allclose(c, ref, atol=1e-1, rtol=1e-2):
        diff = (c.float() - ref.float()).abs().max().item()
        print(f"FAIL numerical mismatch, max abs diff = {diff}")
        return

    print(f"OK {tflops:.2f} BM={BLOCK_M} BN={BLOCK_N} BK={BLOCK_K} W={NUM_WARPS} S={NUM_STAGES}")


try:
    run()
except Exception as e:
    print(f"FAIL {type(e).__name__}: {str(e)[:200]}")
