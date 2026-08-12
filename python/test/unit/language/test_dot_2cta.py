from typing import Optional

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor


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
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
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


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
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


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
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
