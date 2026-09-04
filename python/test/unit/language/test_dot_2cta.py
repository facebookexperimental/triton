from typing import Optional

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell (sm100)")


@triton.jit
def _tl_dot_2cta_data_partition_kernel(
    a_desc,
    b_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    grid_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    grid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    num_tiles: tl.constexpr = grid_m * grid_n
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    start_pid = tl.program_id(0) // 2
    persistent_step: tl.constexpr = NUM_SMS // 2

    for tile_id in tl.range(
            start_pid,
            num_tiles,
            persistent_step,
            warp_specialize=True,
            data_partition_factor=DATA_PARTITION_FACTOR,
            smem_alloc_algo=1,
    ):
        pid_m = tile_id % grid_m
        pid_n = tile_id // grid_m
        offs_m = pid_m * BLOCK_M
        offs_n = pid_n * BLOCK_N
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_m, offs_k])
            b = b_desc.load([offs_k, offs_n])
            acc = tl.dot(a, b, acc, two_ctas=True)

        c_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))


@triton.jit
def _tl_dot_2cta_sliced_dependent_chain_kernel(q_ptr, k_ptr, v_ptr, o_ptr):
    q_desc = tl.make_tensor_descriptor(q_ptr, [256, 128], [128, 1], [256, 128])
    k_desc = tl.make_tensor_descriptor(k_ptr, [128, 128], [128, 1], [128, 128])
    v_desc = tl.make_tensor_descriptor(v_ptr, [128, 128], [128, 1], [64, 128])
    o_desc = tl.make_tensor_descriptor(o_ptr, [256, 128], [128, 1], [256, 128])

    q = q_desc.load([0, 0])
    k = k_desc.load([0, 0])
    v0 = v_desc.load([0, 0])
    v1 = v_desc.load([64, 0])
    qk = tl.dot(q, k.T, two_ctas=True).to(tl.bfloat16)

    p0, p1 = qk.reshape([256, 2, 64]).permute(0, 2, 1).split()
    acc = tl.zeros([256, 128], tl.float32)
    acc = tl.dot(p0, v0, acc, two_ctas=True)
    acc = tl.dot(p1, v1, acc, two_ctas=True)
    o_desc.store([0, 0], acc.to(tl.bfloat16))


@triton.jit
def _tl_dot_2cta_sliced_dependent_chain_ws_kernel(q_ptr, k_ptr, v_ptr, o_ptr):
    q_desc = tl.make_tensor_descriptor(q_ptr, [256, 128], [128, 1], [256, 128])
    k_desc = tl.make_tensor_descriptor(k_ptr, [256, 128], [128, 1], [128, 128])
    v_desc = tl.make_tensor_descriptor(v_ptr, [256, 128], [128, 1], [64, 128])
    o_desc = tl.make_tensor_descriptor(o_ptr, [256, 128], [128, 1], [256, 128])

    start_pid = tl.program_id(0) // 2
    for _ in tl.range(
            start_pid,
            1,
            1,
            warp_specialize=True,
            data_partition_factor=1,
            smem_alloc_algo=1,
    ):
        q = q_desc.load([0, 0])
        acc = tl.zeros([256, 128], tl.float32)
        for ki in tl.range(0, 256, 128):
            k = k_desc.load([ki, 0])
            v0 = v_desc.load([ki, 0])
            v1 = v_desc.load([ki + 64, 0])
            qk = tl.dot(q, k.T, two_ctas=True).to(tl.bfloat16)

            p0, p1 = qk.reshape([256, 2, 64]).permute(0, 2, 1).split()
            acc = acc * 0.5
            acc = tl.dot(p0, v0, acc, two_ctas=True)
            acc = tl.dot(p1, v1, acc, two_ctas=True)
        o_desc.store([0, 0], acc.to(tl.bfloat16))


@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [2])
def test_tl_dot_2cta_data_partition(DATA_PARTITION_FACTOR, device):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m = 512
    n = 256
    k = 128
    block_m = 256
    block_n = 256
    block_k = 64

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    c = torch.empty((m, n), device=device, dtype=dtype)

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [block_m, block_k])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [block_k, block_n])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [block_m, block_n])

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True

        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n) * 2, )
        kernel = _tl_dot_2cta_data_partition_kernel[grid](
            a_desc,
            b_desc,
            c_desc,
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            NUM_SMS=num_sms,
            num_warps=4,
            num_stages=3,
            ctas_per_cga=(2, 1, 1),
        )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(dtype)
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)

    ttgir = kernel.asm["ttgir"]
    assert "ttg.warp_specialize" in ttgir
    assert ttgir.count("ttng.tc_gen5_mma") >= 2
    assert ttgir.count("two_ctas") >= 2


@triton.jit
def _tl_dot_2cta_persistent_meta_ws_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    # Meta-WS persistent 2-CTA loop. The grid is indexed in CTAs and stepped by
    # NUM_SMS -- program_id is NOT divided by 2 -- because ctas_per_cga regroups
    # the existing grid CTAs into clusters and auto-warp-specialization pairs
    # them. What makes that pairing well-formed is the even M-tile count below:
    # a cluster covers two adjacent M tiles, so an odd grid_m would leave the
    # last cluster half-populated.
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_m = (grid_m + 1) // 2 * 2
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n
    k_tiles = tl.cdiv(K, BLOCK_K)

    for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            warp_specialize=True,
            data_partition_factor=DATA_PARTITION_FACTOR,
    ):
        pid_m = tile_id % grid_m
        pid_n = tile_id // grid_m
        offs_m = pid_m * BLOCK_M
        offs_n = pid_n * BLOCK_N
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_m, offs_k])
            b = b_desc.load([offs_k, offs_n])
            acc = tl.dot(a, b, acc, two_ctas=True)

        c_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))


@triton.jit
def _tl_dot_2cta_truncated_b_kernel(a_desc, b_desc, c_desc):
    """Exercise a descriptor-fed B producer before a 2CTA dot."""
    start_pid = tl.program_id(0)
    for tile_id in tl.range(
            start_pid,
            2,
            2,
            warp_specialize=True,
            data_partition_factor=1,
    ):
        offs_m = tile_id * 128
        acc = tl.zeros([128, 256], dtype=tl.float32)
        for offs_k in range(0, 128, 64):
            a = a_desc.load([offs_m, offs_k])
            # Preserve the explicit producer rounding point.  The 2CTA load
            # transform must split the descriptor before recreating this cast.
            b = b_desc.load([offs_k, 0]).to(tl.bfloat16)
            acc = tl.dot(a, b, acc, two_ctas=True)
        c_desc.store([offs_m, 0], acc.to(tl.bfloat16))


@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("m", [512, 384])
def test_tl_dot_2cta_persistent_meta_ws(DATA_PARTITION_FACTOR, m, device):
    """2-CTA under a persistent meta-WS loop, the convention production kernels use.

    This is deliberately a second, distinct convention from
    ``test_tl_dot_2cta_data_partition`` above, which pairs CTAs explicitly with
    ``program_id(0) // 2`` and a ``NUM_SMS // 2`` step. Both are correct; mixing
    them is not. Keeping only the explicit-pairing example made the persistent
    form look like a missing ``// 2``, so both are pinned here.

    m=384 gives an odd BLOCK_M tile count, exercising the even-rounding path.
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16
    n = 256
    k = 128
    block_m = 128
    block_n = 128
    block_k = 64

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    c = torch.zeros((m, n), device=device, dtype=dtype)

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [block_m, block_k])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [block_k, block_n])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [block_m, block_n])

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    # Round the M-tile count up to even to match the kernel, then round the CTA
    # grid down to even so every launched CTA has a cluster partner.
    grid_m = (triton.cdiv(m, block_m) + 1) // 2 * 2
    num_tiles = grid_m * triton.cdiv(n, block_n)
    grid_size = (min(num_sms, num_tiles) // 2) * 2

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        kernel = _tl_dot_2cta_persistent_meta_ws_kernel[(grid_size, )](
            a_desc,
            b_desc,
            c_desc,
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            NUM_SMS=num_sms,
            num_warps=4,
            num_stages=2,
            ctas_per_cga=(2, 1, 1),
        )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(dtype)
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)

    ttgir = kernel.asm["ttgir"]
    assert "ttg.warp_specialize" in ttgir
    assert "two_ctas" in ttgir


def test_tl_dot_2cta_truncated_b(device):
    torch.manual_seed(0)
    a = torch.randn((256, 128), device=device, dtype=torch.bfloat16)
    b = torch.randn((128, 256), device=device, dtype=torch.float32)
    c = torch.empty((256, 256), device=device, dtype=torch.bfloat16)

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [128, 64])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [64, 256])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [128, 256])

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        compiled = _tl_dot_2cta_truncated_b_kernel[(2, )](
            a_desc,
            b_desc,
            c_desc,
            num_warps=8,
            num_stages=4,
            ctas_per_cga=(2, 1, 1),
        )

    reference = torch.matmul(a.float(), b.to(torch.bfloat16).float()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(c, reference, atol=1.0, rtol=2e-2)

    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    assert "ttg.warp_specialize" in ttgir
    assert "partition0(" in ttgir
    assert (
        '"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, '
        '"ttg.cluster-dim-z" = 1 : i32'
    ) in ttgir
    assert "tcgen05.mma.cta_group::2" in ptx
    assert "!tt.tensordesc<64x128xf32" in ttgir


def test_tl_dot_2cta_sliced_dependent_chain(device):
    torch.manual_seed(0)
    q = torch.randn((256, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((128, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn((128, 128), device=device, dtype=torch.bfloat16)
    o = torch.empty((256, 128), device=device, dtype=torch.bfloat16)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)
    compiled = _tl_dot_2cta_sliced_dependent_chain_kernel[(2, )](
        q,
        k,
        v,
        o,
        num_warps=4,
        num_stages=2,
        ctas_per_cga=(2, 1, 1),
        allowDependentTwoCTA=True,
    )
    qk = torch.matmul(q.float(), k.float().T).to(torch.bfloat16)
    reference = torch.matmul(qk.float(), v.float()).to(torch.bfloat16)
    torch.testing.assert_close(o, reference, atol=1.0, rtol=2e-2)

    ttgir = compiled.asm["ttgir"]
    assert ttgir.count('ttng.two_cta_dependency = "collective_contraction"') == 2
    assert 'ttng.two_cta_dependency = "requires_peer_gather"' not in ttgir
def test_tl_dot_2cta_sliced_dependent_chain_ws(device):
    torch.manual_seed(0)
    q = torch.randn((256, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((256, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn((256, 128), device=device, dtype=torch.bfloat16)
    o = torch.empty((256, 128), device=device, dtype=torch.bfloat16)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True
        compiled = _tl_dot_2cta_sliced_dependent_chain_ws_kernel[(2, )](
            q,
            k,
            v,
            o,
            num_warps=4,
            num_stages=2,
            ctas_per_cga=(2, 1, 1),
            allowDependentTwoCTA=True,
        )
    qk0 = torch.matmul(q.float(), k[:128].float().T).to(torch.bfloat16)
    qk1 = torch.matmul(q.float(), k[128:].float().T).to(torch.bfloat16)
    reference = 0.5 * torch.matmul(qk0.float(), v[:128].float())
    reference += torch.matmul(qk1.float(), v[128:].float())
    reference = reference.to(torch.bfloat16)
    torch.testing.assert_close(o, reference, atol=1.0, rtol=2e-2)

    ttgir = compiled.asm["ttgir"]
    assert "ttg.warp_specialize" in ttgir
    assert ttgir.count('ttng.two_cta_dependency = "collective_contraction"') >= 4
    assert 'ttng.two_cta_dependency = "requires_peer_gather"' not in ttgir
