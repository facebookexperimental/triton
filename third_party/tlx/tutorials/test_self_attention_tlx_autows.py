"""Direct packed-QKV accuracy comparison between TLX and AutoWS HSTU backward."""

import os
import subprocess
import sys
import tempfile

import pytest
import torch
from triton._internal_testing import is_blackwell

_HSTU_DIR = os.path.join(os.path.dirname(__file__), "hstu_self_attn")
H = 4
D = 128


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.norm(a.float() - b.float()) / (torch.norm(b.float()) + 1e-12)).item()


def _make_inputs(L: int, Z: int, target_count: int):
    torch.manual_seed(0)
    total = L * Z
    qkv = torch.randn(total, H, 3 * D, device="cuda", dtype=torch.bfloat16)
    q, k, v = (x.detach().requires_grad_(True) for x in torch.split(qkv, [D, D, D], dim=-1))
    assert all(not x.is_contiguous() and x.stride(-1) == 1 for x in (q, k, v))
    dout = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
    seq_offsets = torch.arange(Z + 1, device="cuda", dtype=torch.int64) * L
    attn_scale = torch.tensor(1.0 / L, device="cuda", dtype=torch.float32)
    num_targets = None
    if target_count:
        num_targets = torch.full((Z, ), target_count, device="cuda", dtype=torch.int64)
    return q, k, v, dout, seq_offsets, attn_scale, num_targets


def _run_backend(backend: str, output_path: str, L: int, Z: int, target_count: int, dq_fp32: bool) -> None:
    sys.path.insert(0, _HSTU_DIR)
    q, k, v, dout, seq_offsets, attn_scale, num_targets = _make_inputs(L, Z, target_count)

    if backend == "autows":
        import hstu_autows_config as config

        config.set_config(
            autows=True,
            dq_reduce=True,
            dq_fp32=dq_fp32,
            dq_reuse=True,
            clc=True,
            clc_smem_algo=1,
            dkdv_subtile=2,
            dp=1,
            bwd_bm=64,
            bwd_bn=128,
            bwd_stages=1 if dq_fp32 else 2,
            warps=8 if dq_fp32 else 4,
            dq_iters=4,
            pin=True,
            split_causal_loops=False,
            dq_transposed=True,
        )
        import triton_hstu_attention as kernel

        out = kernel.triton_hstu_mha(
            max_seq_len=L,
            alpha=1.0 / D,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            sort_by_length=False,
            enable_tma=True,
            num_targets=num_targets,
        )
    else:
        import tlx_bw_hstu_attention as kernel

        early_release = 2 if target_count == 0 else 1
        best = next(cfg for cfg in kernel.get_hstu_bwd_configs() if cfg.kwargs["BLOCK_M1"] == 64
                    and cfg.kwargs["BLOCK_N1"] == 128 and cfg.kwargs["EARLY_RELEASE_SUBTILES"] == early_release)
        kernel._attn_fwd_ws.configs = [kernel.get_fwd_persistent_configs()[0]]
        kernel._attn_fwd_ws.cache.clear()
        kernel._hstu_attn_bwd_ws.configs = [best]
        kernel._hstu_attn_bwd_ws.cache.clear()
        kernel._hstu_attn_bwd_ws_non_persistent.configs = [best]
        kernel._hstu_attn_bwd_ws_non_persistent.cache.clear()
        out = kernel.tlx_bw_hstu_mha(
            max_seq_len=L,
            alpha=1.0 / D,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            num_softmax_heads=0,
            num_targets=num_targets,
            causal=True,
        )

    out.backward(dout)
    torch.save({"dq": q.grad.cpu(), "dk": k.grad.cpu(), "dv": v.grad.cpu()}, output_path)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU for CLC")
@pytest.mark.parametrize("dq_fp32", [False, True], ids=["dq-bf16", "dq-fp32"])
@pytest.mark.parametrize("target_count", [0, 20], ids=["causal", "target-20"])
def test_packed_qkv_tlx_matches_autows(target_count, dq_fp32):
    """Pinned TLX and both AutoWS dQ modes agree on packed QKV views."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = {}
        for backend in ("tlx", "autows"):
            output_path = os.path.join(tmpdir, f"{backend}-{dq_fp32}.pt")
            env = os.environ.copy()
            env["TRITON_ALWAYS_COMPILE"] = "1"
            env["TRITON_DISABLE_WSBARRIER_REORDER"] = "1"
            if backend == "autows":
                env["TRITON_USE_META_WS"] = "1"
                env["TRITON_WS_SMEM_PLAN_SEARCH"] = "1"
                if dq_fp32:
                    env["TRITON_WS_TMA_REDUCE_STAGING_COPIES"] = "2"
                else:
                    env.pop("TRITON_WS_TMA_REDUCE_STAGING_COPIES", None)
            else:
                env.pop("TRITON_USE_META_WS", None)
                env.pop("TRITON_WS_SMEM_PLAN_SEARCH", None)
                env.pop("TRITON_WS_TMA_REDUCE_STAGING_COPIES", None)
            result = subprocess.run(
                [
                    sys.executable, __file__, "--compare-child", backend, output_path, "256", "2",
                    str(target_count),
                    str(int(dq_fp32))
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=900,
            )
            assert result.returncode == 0, f"{backend} failed:\n{result.stdout}\n{result.stderr}"
            outputs[backend] = torch.load(output_path, weights_only=True)

    for name in ("dq", "dk", "dv"):
        error = _rel_l2(outputs["autows"][name], outputs["tlx"][name])
        print(f"target_count={target_count} dq_fp32={dq_fp32} {name} AutoWS-vs-TLX rel-L2: {error:.3e}")
        assert error < 1e-2, f"AutoWS-vs-TLX {name} rel-L2 {error:.2e}"


if __name__ == "__main__" and len(sys.argv) == 8 and sys.argv[1] == "--compare-child":
    _run_backend(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), bool(int(sys.argv[7])))
