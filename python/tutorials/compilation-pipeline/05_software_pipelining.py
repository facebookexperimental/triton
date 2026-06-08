"""05 — Software pipelining: overlap loads with compute via num_stages.

Pipeline: **TTGIR** ``AssignLatencies -> ScheduleLoops -> Pipeline``
(``lib/Dialect/TritonGPU/Transforms/Pipeliner/``). With ``num_stages > 1`` the
pipeliner multi-buffers a loop's loads and issues them **asynchronously**
(``ttg.async_copy_global_to_local`` / ``cp.async`` / TMA) so iteration *k+1*'s data
is fetched while iteration *k* computes.

What to notice: in a matmul K-loop, ``num_stages=1`` has no async ops; ``num_stages=3``
introduces many async copies + shared-memory buffers (``!ttg.memdesc``). The loop's
*accumulation order over k is unchanged* — pipelining only moves *when loads happen*.

Bit-neutral mechanic (a NEGATIVE example for the equivalence checker): reordering
loads does not reassociate the FP accumulation, so results are bitwise-identical
across ``num_stages``. We assert that.

Run:  python python/tutorials/compilation-pipeline/05_software_pipelining.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, is_cuda, show


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b, input_precision="ieee")  # ieee => deterministic FMA path
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def main():
    M = N = K = 256
    BM = BN = 64
    BK = 32
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    args = (a, b, torch.empty(M, N, device="cuda"), M, N, K, BM, BN, BK)

    ck1 = compile_only(matmul_kernel, *args, num_stages=1, grid=(1, ))
    ck3 = compile_only(matmul_kernel, *args, num_stages=3, grid=(1, ))

    banner("05 — num_stages multi-buffers the K-loop loads (async copies appear)")
    for label, ck in (("num_stages=1", ck1), ("num_stages=3", ck3)):
        print(f"    {label}:  async-copy ops = {count(ck, 'ttgir', 'async_copy'):>2}   "
              f"smem buffers (!ttg.memdesc) = {count(ck, 'ttgir', '!ttg.memdesc'):>2}")
    show(ck3, "ttgir", grep="async_copy", limit=3, label="\nsample async copies (num_stages=3):")

    # Bit-neutral: pipelining preserves the k-accumulation order exactly.
    c1 = torch.empty(M, N, device="cuda")
    c3 = torch.empty(M, N, device="cuda")
    matmul_kernel[(1, )](a, b, c1, M, N, K, BM, BN, BK, num_stages=1)
    matmul_kernel[(1, )](a, b, c3, M, N, K, BM, BN, BK, num_stages=3)
    torch.cuda.synchronize()
    assert torch.equal(c1, c3), "software pipelining must be bitwise-neutral"
    print("\n[OK] num_stages=1 and num_stages=3 give bitwise-identical results"
          " (pipelining does not reassociate the accumulation).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
