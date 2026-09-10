"""TLX storage alias tests -- Blackwell-only."""
import subprocess
import sys
import textwrap
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell (2-CTA MMA)")
class TestStorageAlias2CTA:
    """Tests for storage_alias_spec with 2-CTA (TwoCTA_RHS) TMEM tiles.

    A 64-row accumulator under a collaborative ``two_ctas=True`` MMA uses the
    ``TwoCTA_RHS`` layout (blockM=64): M=64 stays full in both CTAs and the split
    is along N, so its real per-CTA footprint is 128x64. Storage-alias lowering
    runs after layout propagation, so it knows this footprint and can place such
    a tile at a non-zero TMEM column offset within a shared region.
    """

    @pytest.mark.parametrize("use_alias", [False, True])
    def test_gemm_64x128_2cta(self, use_alias):
        """64x128 2-CTA GEMM: the accumulator is the full (BLOCK_M, BLOCK_N) tile
        (blockM=64 -> TwoCTA_RHS); the B operand is sliced along N per CTA. Run
        with both a plain TMEM accumulator and a storage_alias_spec accumulator
        (the latter checks the lowering produces a valid 2-CTA TMEM encoding)."""

        @triton.jit
        def gemm_2cta(a_ptr, stride_am, stride_ak, b_ptr, stride_bk, stride_bn, c_ptr, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT_DTYPE: tl.constexpr,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, USE_ALIAS: tl.constexpr):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            cluster_cta_rank = tlx.cluster_cta_rank()
            pred_leader_cta = cluster_cta_rank % 2 == 0

            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N + (cluster_cta_rank % 2) * (BLOCK_N // 2)  # 2-CTA: B split along N
            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            desc_a = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
                                               block_shape=[BLOCK_M, BLOCK_K])
            desc_b = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                                               block_shape=[BLOCK_K, BLOCK_N // 2])

            buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), tl.constexpr(1))
            buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N // 2), tlx.dtype_of(b_ptr), tl.constexpr(1))
            a_smem = tlx.local_view(buf_alloc_a, 0)
            b_smem = tlx.local_view(buf_alloc_b, 0)

            bars = tlx.alloc_barriers(tl.constexpr(3))
            bar_a = tlx.local_view(bars, 0)
            bar_b = tlx.local_view(bars, 1)
            bar_cta = tlx.alloc_barriers(1, arrive_count=2)
            bar_leader_cta = tlx.local_view(bar_cta, 0)

            # 64-row accumulator: full (BLOCK_M, BLOCK_N), blockM=64 -> TwoCTA_RHS.
            if USE_ALIAS:
                spec = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
                buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem,
                                          reuse=spec)
            else:
                buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
            acc_tmem = tlx.local_view(buffers, 0)
            tlx.local_store(acc_tmem, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
            dot_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

            phase = 0
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                offs_k = k * BLOCK_K
                tlx.barrier_expect_bytes(bar_a, BLOCK_M * BLOCK_K * 2)
                tlx.barrier_expect_bytes(bar_b, BLOCK_K * (BLOCK_N // 2) * 2)
                tlx.async_descriptor_load(desc_a, a_smem, [offs_am, offs_k], bar_a)
                tlx.async_descriptor_load(desc_b, b_smem, [offs_k, offs_bn], bar_b)
                tlx.barrier_wait(bar_a, phase)
                tlx.barrier_wait(bar_b, phase)
                # CTA0 waits for CTA1's data before issuing the collaborative MMA.
                tlx.barrier_arrive(bar_leader_cta, 1, remote_cta_rank=cluster_cta_rank & ~1)
                tlx.barrier_wait(bar_leader_cta, phase=k % 2, pred=pred_leader_cta)
                tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=True, mBarriers=[dot_bars[0]], two_ctas=True,
                              out_dtype=OUT_DTYPE)
                tlx.barrier_wait(dot_bars[0], phase)
                phase = phase ^ 1

            result = tlx.local_load(acc_tmem)
            c = result.to(tlx.dtype_of(c_ptr))
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            tl.store(c_ptrs, c)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        torch.manual_seed(0)
        triton.set_allocator(alloc_fn)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 128
        # ctas_per_cga=(4,2,1): grid_x must be a multiple of 4, grid_y a multiple of 2.
        M, N, K = BLOCK_M * 4, BLOCK_N * 2, 256
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        c = torch.empty((M, N), device="cuda", dtype=torch.float16)

        gemm_2cta[(M // BLOCK_M, N // BLOCK_N)](
            a, a.stride(0), a.stride(1), b, b.stride(0), b.stride(1), c, c.stride(0), c.stride(1),  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, OUT_DTYPE=tl.float32,  #
            M=M, N=N, K=K, USE_ALIAS=use_alias, num_stages=1, ctas_per_cga=(4, 2, 1))
        torch.cuda.synchronize()

        ref = (a.float() @ b.float()).to(torch.float16)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=5e-2)

    def test_fa_bwd_64x128_2cta(self):
        """FA backward 2-CTA: dQ is a 64x128 TwoCTA_RHS storage-alias tile placed
        at a non-zero TMEM column offset via set_buffer_overlap. This only works
        when storage-alias lowering runs after layout propagation (so it knows
        dQ's real 128x64 footprint). Run the compile+launch in a subprocess so a
        compile-time abort() surfaces as a clean failure instead of crashing
        pytest."""
        driver = textwrap.dedent(r"""
            import torch
            import triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent as fa

            # Avoid unrelated forward autotuning in this backward regression test.
            fa._attn_fwd_ws.configs = [fa.configs[0]]

            # Force the backward autotuner to only the 2-CTA config (BLOCK_M1=128).
            # Its dq tile is (BLOCK_M1//NUM_CTAS, HEAD_DIM) = (64,128): a blockM=64
            # TwoCTA_RHS storage-alias tile placed at a non-zero column offset via
            # set_buffer_overlap -- the "64x128 2-CTA" storage-alias case.
            fa._attn_bwd_ws.configs = [fa.configs_bwd_2cta[0]]

            torch.manual_seed(0)
            BATCH, N_HEAD, N_CTX, HEAD_DIM = 1, 1, 512, 128
            q = torch.randn((BATCH, N_HEAD, N_CTX, HEAD_DIM), device="cuda",
                            dtype=torch.float16, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)
            sm_scale = 1.0 / (HEAD_DIM ** 0.5)

            out = fa.attention(q, k, v, sm_scale, False)
            do = torch.randn_like(out)
            out.backward(do)
            tdq, tdk, tdv = q.grad.float(), k.grad.float(), v.grad.float()

            # fp32 reference
            qf = q.detach().float().requires_grad_()
            kf = k.detach().float().requires_grad_()
            vf = v.detach().float().requires_grad_()
            p = torch.softmax((qf @ kf.transpose(-2, -1)) * sm_scale, dim=-1)
            (p @ vf).backward(do.float())

            for name, a, b in [("dq", tdq, qf.grad), ("dk", tdk, kf.grad), ("dv", tdv, vf.grad)]:
                err = (a - b).abs().max().item()
                assert err < 5e-2, f"{name} max_abs_err={err}"
            print("OK")
        """)
        proc = subprocess.run([sys.executable, "-c", driver], capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0 and "OK" in proc.stdout, (
            f"FA backward 64x128 2-CTA failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
