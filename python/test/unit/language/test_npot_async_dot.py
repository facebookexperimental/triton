"""Test async_dot (warp-specialized) with NPOT K dimensions.

This tests the minimal warp-specialized kernel with async_dot to isolate
whether the NPOT K=192 hang is in the WGMMA descriptor generation or
barrier synchronization.
"""
import os
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor


@pytest.fixture(autouse=True)
def enable_npot(monkeypatch):
    monkeypatch.setenv("TRITON_ALLOW_NPOT", "1")


def _is_sm100():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and props.minor == 0


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="SM100 only")


def _host_pre_hook(nargs):
    M = nargs["BLOCK_M"]
    K = nargs["BLOCK_K"]
    N = nargs["BLOCK_N"]
    if not isinstance(nargs["desc_a"], TensorDescriptor):
        return
    nargs["desc_a"].block_shape = [M, K]
    nargs["desc_b"].block_shape = [K, N]
    nargs["desc_c"].block_shape = [M, N]


@triton.jit
def _ws_dot_kernel(desc_a, desc_b, desc_c, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    """Minimal warp-specialized async_dot: C = A @ B."""
    a_tiles = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(desc_a), 1)
    b_tiles = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(desc_b), 1)

    a_fulls = tlx.alloc_barriers(num_barriers=1)
    b_fulls = tlx.alloc_barriers(num_barriers=1)

    c_tiles = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, 1, tlx.storage_kind.tmem)
    c_fulls = tlx.alloc_barriers(num_barriers=1)

    with tlx.async_tasks():
        # MMA group
        with tlx.async_task(num_warps=1, registers=24):
            tlx.barrier_wait(a_fulls[0], 0)
            tlx.barrier_wait(b_fulls[0], 0)
            tlx.async_dot(
                a_tiles[0],
                b_tiles[0],
                c_tiles[0],
                use_acc=False,
                mBarriers=[c_fulls[0]],
            )

        # Store group
        with tlx.async_task("default"):
            tlx.barrier_wait(c_fulls[0], 0)
            c = tlx.local_load(c_tiles[0])
            desc_c.store([0, 0], c.to(tlx.dtype_of(desc_c)))

        # Load group
        with tlx.async_task(num_warps=1, registers=24):
            tlx.barrier_expect_bytes(a_fulls[0], 2 * BLOCK_M * BLOCK_K)
            tlx.async_descriptor_load(desc_a, a_tiles[0], [0, 0], a_fulls[0])
            tlx.barrier_expect_bytes(b_fulls[0], 2 * BLOCK_K * BLOCK_N)
            tlx.async_descriptor_load(desc_b, b_tiles[0], [0, 0], b_fulls[0])


def _test_ws_dot(M, K, N, timeout_sec=60):
    """Test warp-specialized async_dot with given dimensions."""
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    dummy_block = [1, 1]
    desc_a = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=dummy_block)
    desc_b = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=dummy_block)
    desc_c = TensorDescriptor(c, shape=[M, N], strides=[N, 1], block_shape=dummy_block)

    nargs = {
        "BLOCK_M": M,
        "BLOCK_K": K,
        "BLOCK_N": N,
        "desc_a": desc_a,
        "desc_b": desc_b,
        "desc_c": desc_c,
    }
    _host_pre_hook(nargs)

    def alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    _ws_dot_kernel[(1, )](
        desc_a,
        desc_b,
        desc_c,
        BLOCK_M=M,
        BLOCK_K=K,
        BLOCK_N=N,
        num_stages=1,
        num_warps=4,
    )
    torch.cuda.synchronize()

    ref = torch.matmul(a.float(), b.float())
    max_diff = (c.float() - ref).abs().max().item()
    rel_err = max_diff / ref.abs().max().item() if ref.abs().max().item() > 0 else 0
    return rel_err


def test_ws_async_dot_k128():
    """Baseline: K=128 (pow2) in warp-specialized async_dot."""
    err = _test_ws_dot(128, 128, 128)
    print(f"\nWS async_dot K=128: rel_err={err:.6f}")
    assert err < 0.05, f"K=128 failed: rel_err={err}"


def test_ws_async_dot_k64():
    """Baseline: K=64 (pow2) in warp-specialized async_dot."""
    err = _test_ws_dot(128, 64, 128)
    print(f"\nWS async_dot K=64: rel_err={err:.6f}")
    assert err < 0.05, f"K=64 failed: rel_err={err}"


def test_ws_async_dot_k96():
    """NPOT K=96 in warp-specialized async_dot."""
    err = _test_ws_dot(128, 96, 128)
    print(f"\nWS async_dot K=96: rel_err={err:.6f}")
    assert err < 0.05, f"K=96 failed: rel_err={err}"


def test_ws_async_dot_k48():
    """NPOT K=48 in warp-specialized async_dot."""
    err = _test_ws_dot(128, 48, 128)
    print(f"\nWS async_dot K=48: rel_err={err:.6f}")
    assert err < 0.05, f"K=48 failed: rel_err={err}"


def test_ws_async_dot_k192():
    """NPOT K=192 in warp-specialized async_dot -- the DSv3 case."""
    err = _test_ws_dot(128, 192, 128)
    print(f"\nWS async_dot K=192: rel_err={err:.6f}")
    assert err < 0.05, f"K=192 failed: rel_err={err}"
