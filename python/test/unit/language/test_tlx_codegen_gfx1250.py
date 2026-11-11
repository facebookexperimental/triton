"""Compile-only TLX checks for gfx1250 (MI450): no device required.

Run by the gfx1250 lane without `mirage`; runtime tests remain in
test_tlx_amd_gfx1250.py. The filename matches the `test_tlx_*_gfx1250.py`
exclusions in the h100/b200/mi350 path filters, so only gfx1250 selects it.
"""
import re

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile
from triton.compiler.errors import CompilationError
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.tutorials import amd_tdm_gemm_pipelined as _gfx1250_gemm
from triton.language.extra.tlx.tutorials import amd_mxfp_gemm_tdm_pipelined as _gfx1250_mxfp
from triton.language.extra.tlx.tutorials import amd_fa_tdm_pipelined as _gfx1250_attention
from triton.language.extra.tlx.tutorials.amd_grouped_gemm_gfx1250 import (
    amd_grouped_gemm_gfx1250_test as _gfx1250_grouped, )
from triton.language.extra.tlx.tutorials.amd_mxfp_gemm_tdm_pipelined import (
    mxgemm_tdm_pipelined_kernel as _amd_mxfp_gemm_kernel, )
from triton.language.extra.tlx.tutorials.amd_tdm_gemm_pipelined import (
    matmul_tdm_pipelined_kernel as _amd_tdm_gemm_kernel, )

GFX950 = GPUTarget("hip", "gfx950", 64)
GFX942 = GPUTarget("hip", "gfx942", 64)
GFX1250 = GPUTarget("hip", "gfx1250", 32)


def compile_for_target(fn, signature, constexprs, target, attrs=None):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    return triton_compile(src, target=target)


def compile_for_gfx950(fn, signature, constexprs, attrs=None):
    return compile_for_target(fn, signature, constexprs, GFX950, attrs=attrs)


def compile_for_gfx1250(fn, signature, constexprs):
    """Compile a TLX kernel for gfx1250 and return the compiled object."""
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=GFX1250)


@triton.jit
def _buffer_load_to_local_contiguity_subslice_kernel(x_ptr):
    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 16)],
        [32, 16],
        order=[1, 0],
    )
    k = tl.arange(0, 32)
    m = tl.arange(0, 16)
    offsets = (k[:, None] * 16 + m[None, :]).to(tl.int32)
    buffer = tlx.local_alloc((32, 16), tl.bfloat16, 1, layout=shared_layout)
    destination = tlx.local_slice(
        tlx.local_view(buffer, 0),
        [0, 0],
        [32, 16],
    )
    tlx.buffer_load_to_local(
        destination,
        x_ptr,
        offsets,
        contiguity=2,
    )


@triton.jit
def _amd_scheduled_mfma_gfx942_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    PERSISTENT: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    INSTR_K: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[16, 16, INSTR_K],
        transposed=TRANSPOSED,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    rows = tl.arange(0, 32)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tl.full((32, 128), 7.0, tl.float32)
    acc = tlx.require_layout(acc, mma, pin=False)
    if PERSISTENT:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="persistent",
            # On CDNA3 the compiler-generated AGPR read is not ordered
            # against the MFMA drain, so the accumulator stays in VGPRs.
            accumulator_register_class="vgpr",
            initialize=True,
        )
    else:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="transient",
            initialize=True,
        )
        result, _ = tlx.amd_mfma_commit(result, b)
    offsets = output_ptr + rows[:, None] * 128 + cols[None, :]
    offsets = tlx.require_layout(offsets, mma, pin=False)
    tl.store(offsets, result)


@triton.jit
def _invalid_update_tensor_descriptor_kernel(x_ptr, MODE: tl.constexpr):
    desc = tl.make_tensor_descriptor(x_ptr, [16, 16], [16, 1], [16, 16])
    if MODE == 0:
        desc = tlx.update_tensor_descriptor(desc)
    elif MODE == 1:
        desc = tlx.update_tensor_descriptor(desc, pred=True, clamp_bounds=True)
    elif MODE == 2:
        desc = tlx.update_tensor_descriptor(
            desc,
            add_offsets=[0, 0],
            set_bounds=[16, 16],
            clamp_bounds=True,
        )
    elif MODE == 3:
        desc = tlx.update_tensor_descriptor(desc, add_offsets=[0])
    else:
        desc = tlx.update_tensor_descriptor(desc, add_offsets=[0, 0])


@triton.jit
def _local_reshape_kernel(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    offsets = tl.arange(0, ROWS * COLS)
    values = tl.load(input_ptr + offsets)

    flat_buffers = tlx.local_alloc((ROWS * COLS, ), tl.float32, 1)
    flat = tlx.local_view(flat_buffers, 0)
    tlx.local_store(flat, values)

    reshaped = tlx.local_reshape(flat, [ROWS, COLS])
    result = tlx.local_load(reshaped)

    offs_m = tl.arange(0, ROWS)
    offs_n = tl.arange(0, COLS)
    output_offsets = offs_m[:, None] * COLS + offs_n[None, :]
    tl.store(output_ptr + output_offsets, result)


@triton.jit
def _dot_scaled_tiles_per_warp_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    TILES_PER_WARP: tl.constexpr,
):
    block_k_scale: tl.constexpr = BLOCK_K // SCALE_BLOCK
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_ks = tl.arange(0, block_k_scale)

    a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    a_scale = tl.load(a_scale_ptr + offs_m[:, None] * block_k_scale + offs_ks[None, :])
    b_scale = tl.load(b_scale_ptr + offs_n[:, None] * block_k_scale + offs_ks[None, :])

    acc = tlx.dot_scaled(a, a_scale, "e5m2", b, b_scale, "e5m2", tiles_per_warp=TILES_PER_WARP)
    tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc)


def _compile_dot_scaled_tiles_per_warp(tiles_per_warp):
    src = ASTSource(
        fn=_dot_scaled_tiles_per_warp_kernel,
        signature={
            "a_ptr": "*fp8e5",
            "b_ptr": "*fp8e5",
            "a_scale_ptr": "*i8",
            "b_scale_ptr": "*i8",
            "c_ptr": "*fp32",
        },
        constexprs={
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "SCALE_BLOCK": 32,
            "TILES_PER_WARP": tiles_per_warp,
        },
    )
    return triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))


@triton.jit
def _require_amd_wmma_layout_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offsets = offs_m[:, None] * BLOCK + offs_n[None, :]
    values = tl.load(x_ptr + offsets)
    values = tlx.require_amd_wmma_layout(
        values,
        version=3,
        transposed=True,
        warp_bases=((0, 2), (2, 0)),
        reg_bases=((0, 1), (1, 0)),
        instr_shape=(16, 16, 128),
    )
    offsets = tlx.require_amd_wmma_layout(
        offsets,
        version=3,
        transposed=True,
        warp_bases=((0, 2), (2, 0)),
        reg_bases=((0, 1), (1, 0)),
        instr_shape=(16, 16, 128),
    )
    tl.store(y_ptr + offsets, values)


@triton.jit
def _async_amd_desc_load_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _async_amd_desc_load_fused_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    MULTICAST_MASKS: tl.constexpr = None,
    SHARED_LAYOUT: tl.constexpr = None,
    MASK_SHIFT_BITS: tl.constexpr = 0,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, N], [N, 1], [M, N])
    b_desc = tl.make_tensor_descriptor(b_ptr, [M, N], [N, 1], [M, N])
    if SHARED_LAYOUT is None:
        a_buf = tlx.local_alloc((M, N), tl.float16, 1)
        b_buf = tlx.local_alloc((M, N), tl.float16, 1)
    else:
        a_buf = tlx.local_alloc((M, N), tl.float16, 1, layout=SHARED_LAYOUT)
        b_buf = tlx.local_alloc((M, N), tl.float16, 1, layout=SHARED_LAYOUT)
    a_smem = tlx.local_view(a_buf, 0)
    b_smem = tlx.local_view(b_buf, 0)
    # A missing member load must not pass by reading a previous launch's LDS.
    tlx.local_store(a_smem, tl.full((M, N), float("nan"), tl.float16))
    tlx.local_store(b_smem, tl.full((M, N), float("nan"), tl.float16))
    tlx.workgroup_barrier()
    a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    if MASK_SHIFT_BITS and MULTICAST_MASKS is not None:
        shift = tlx.cluster_cta_rank() & MASK_SHIFT_BITS
        token = tlx.async_amd_descriptor_load_fused([(a_desc, a_smem, 0b0011), (b_desc, b_smem, 0b1100)],
                                                    multicast_masks=(MULTICAST_MASKS[0] << shift,
                                                                     MULTICAST_MASKS[1] << shift))
    else:
        token = tlx.async_amd_descriptor_load_fused([(a_desc, a_smem, 0b0011), (b_desc, b_smem, 0b1100)],
                                                    multicast_masks=MULTICAST_MASKS)
    tlx.async_amd_descriptor_wait(tokens=[token])
    # TDM completion is per wave. Consumers only read their own CTA's LDS,
    # so the handoff requires workgroup synchronization, not a cluster barrier.
    tlx.workgroup_barrier()
    a = tlx.local_load(a_smem)
    b = tlx.local_load(b_smem)
    offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    # Check each member separately: A+B would also pass if A and B were swapped.
    tl.store(output_ptr + offsets, a)
    tl.store(output_ptr + M * N + offsets, b)


@triton.jit
def _async_amd_desc_load_with_token_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(tokens=[tok])
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _async_amd_desc_load_pred_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    pred = tl.program_id(0) == 0
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0], pred=pred)
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _async_amd_desc_store_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc_in = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    desc_out = tl.make_tensor_descriptor(y_ptr, [M, N], [N, 1], [M, N])
    # Separate buffers for load vs store — they get different encodings
    # (padded for load, swizzled for store) and can't share a buffer
    # until alignTDMDescriptorEncodings is ported.
    load_buf = tlx.local_alloc((M, N), tl.float16, 1)
    store_buf = tlx.local_alloc((M, N), tl.float16, 1)
    load_view = tlx.local_view(load_buf, 0)
    store_view = tlx.local_view(store_buf, 0)
    tlx.async_amd_descriptor_load(desc_in, load_view, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    data = tlx.local_load(load_view)
    tlx.local_store(store_view, data)
    tlx.async_amd_descriptor_store(desc_out, store_view, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)


@triton.jit
def _update_tensor_descriptor_store_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc_in = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    desc_out = tl.make_tensor_descriptor(y_ptr, [M, N], [N, 1], [M, N])
    load_buf = tlx.local_alloc((M, N), tl.float16, 1)
    store_buf = tlx.local_alloc((M, N), tl.float16, 1)
    load_view = tlx.local_view(load_buf, 0)
    store_view = tlx.local_view(store_buf, 0)

    pred = tl.program_id(0) == 0
    desc_in = tlx.update_tensor_descriptor(desc_in, set_bounds=[M, N], pred=pred)
    offset_m = desc_in.shape[0] - M
    offset_n = (desc_in.strides[1] - 1).to(tl.int32)
    desc_in = tlx.update_tensor_descriptor(desc_in, add_offsets=[offset_m, offset_n])
    tlx.async_amd_descriptor_load(desc_in, load_view)
    tlx.async_amd_descriptor_wait(0)
    tlx.local_store(store_view, tlx.local_load(load_view))

    desc_out = tlx.update_tensor_descriptor(desc_out, add_offsets=[0, 0], clamp_bounds=True)
    tlx.async_amd_descriptor_store(desc_out, store_view)
    tlx.async_amd_descriptor_wait(0)


@triton.jit
def _amd_desc_prefetch_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    tlx.amd_descriptor_prefetch_tensor(desc, [0, 0])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _amd_desc_prefetch_speculative_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    pred = tl.program_id(0) == 0
    tlx.amd_descriptor_prefetch_tensor(desc, [0, 0], pred=pred, speculative=True)
    # A TDM load on the same descriptor so it gets a valid encoding
    # during lowering (prefetch alone doesn't assign one).
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _pinned_tdm_memdesc_view_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    VIEW: tl.constexpr,
):
    smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(N, 128 // 16)], [M, N])
    desc = tl.make_tensor_descriptor(input_ptr, [M, N], [N, 1], [M, N])
    buffers = tlx.local_alloc((M, N), tl.float16, 1, layout=smem_layout)
    full = tlx.local_view(buffers, 0)

    token = tlx.async_amd_descriptor_load(desc, full, [0, 0])
    tlx.async_amd_descriptor_wait(tokens=[token])

    if VIEW == 0:
        view = tlx.local_slice(full, [0, 32], [M, 32])
        rows = tl.arange(0, M)
        cols = tl.arange(0, 32)
        width: tl.constexpr = 32
    else:
        view = tlx.local_reshape(full, [N, M])
        rows = tl.arange(0, N)
        cols = tl.arange(0, M)
        width: tl.constexpr = M

    values = tlx.local_load(view)
    tl.store(output_ptr + rows[:, None] * width + cols[None, :], values)


@triton.jit
def _tdm_copy_view_kernel(input_ptr, other_ptr, output_ptr, row, VIEW: tl.constexpr, MODE: tl.constexpr,
                          PADDED: tl.constexpr = True):
    narrow_store: tl.constexpr = MODE == "store" and (VIEW == "transpose" or VIEW == "compatible_transpose"
                                                      or VIEW == "reshape" or VIEW == "inner_slice")
    interval: tl.constexpr = 64 if narrow_store else 128
    order: tl.constexpr = [0, 1] if VIEW == "compatible_transpose" else [1, 0]
    if PADDED:
        layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(interval, 8)], [64, 128], order)
    else:
        layout: tl.constexpr = tlx.swizzled_layout(0, 0, 0, order=order)
    buffers = tlx.local_alloc((64, 128), tl.float16, 1, layout=layout)
    full = tlx.local_view(buffers, 0)
    if MODE == "store":
        offsets = tl.arange(0, 64)[:, None] * 128 + tl.arange(0, 128)[None, :]
        tlx.local_store(full, tl.load(input_ptr + offsets))
    if VIEW == "transpose" or VIEW == "compatible_transpose":
        view = tlx.local_trans(full)
        M: tl.constexpr = 128
        N: tl.constexpr = 64
    elif VIEW == "reshape":
        view = tlx.local_reshape(full, [128, 64])
        M: tl.constexpr = 128
        N: tl.constexpr = 64
    elif VIEW == "slice":
        view = tlx.local_slice(full, [32, 0], [32, 128])
        M: tl.constexpr = 32
        N: tl.constexpr = 128
    elif VIEW == "dynamic_slice":
        view = tlx.local_slice(full, [row, 0], [32, 128])
        M: tl.constexpr = 32
        N: tl.constexpr = 128
    elif VIEW == "inner_slice":
        view = tlx.local_slice(full, [0, 64], [64, 64])
        M: tl.constexpr = 64
        N: tl.constexpr = 64
    else:
        view = full
        M: tl.constexpr = 64
        N: tl.constexpr = 128
    if MODE == "store":
        desc = tl.make_tensor_descriptor(output_ptr, [M, N], [N, 1], [M, N])
        tlx.async_amd_descriptor_store(desc, view, [0, 0])
        tlx.async_amd_descriptor_wait(pendings=0)
    else:
        desc = tl.make_tensor_descriptor(input_ptr, [M, N], [N, 1], [M, N])
        if MODE == "fused":
            other_buffers = tlx.local_alloc((M, N), tl.float16, 1)
            other = tlx.local_view(other_buffers, 0)
            other_desc = tl.make_tensor_descriptor(other_ptr, [M, N], [N, 1], [M, N])
            other_desc = tlx.update_tensor_descriptor(other_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
            desc = tlx.update_tensor_descriptor(desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
            token = tlx.async_amd_descriptor_load_fused([(desc, view, 3), (other_desc, other, 12)])
        else:
            token = tlx.async_amd_descriptor_load(desc, view, [0, 0])
        tlx.async_amd_descriptor_wait(tokens=[token])
        values = tlx.local_load(view)
        tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], values)


@triton.jit
def _amd_wmma_queuing_kernel():
    pass


def _tdm_fused_distributed_constants(num_ctas, split_m, mask_mode):
    replicas = num_ctas // split_m
    shared = tlx.swizzled_shared_layout_encoding(
        vectorSize=1,
        perPhase=1,
        maxPhase=1,
        order=[1, 0],
        numCTAs=num_ctas,
        numCTAsPerCGA=[split_m, replicas],
        numCTASplit=[split_m, 1],
        numCTAOrder=[1, 0],
    )
    # Adjacent CTAs share a row partition, matching the default register layout
    # without requiring cross-CTA shared loads. Explicit masks disable one
    # member's multicast despite its replicated layout; test both member orders.
    mask = (1 << replicas) - 1
    masks = {"inferred": None, "explicit": (mask, 0), "explicit_b": (0, mask), "disabled": (0, 0)}[mask_mode]
    return dict(M=64, N=64, SHARED_LAYOUT=shared, MULTICAST_MASKS=masks, MASK_SHIFT_BITS=num_ctas - replicas)


def _check_tdm_fused_multicast_codegen(llvm_ir, num_ctas, split_m, mask_mode):
    from triton._filecheck import run_filecheck

    # Numerical correctness cannot distinguish multicast from separate loads.
    # Follow descriptor group 1, whose low 16 bits select recipients, all the
    # way to the TDM intrinsic. Also check which producer waves select each mask.
    enabled = (mask_mode in ("inferred", "explicit"), mask_mode in ("inferred", "explicit_b"))
    checks = ["CHECK: [[WAVE:%[^ ]+]] = {{.*}}call i32 @llvm.amdgcn.wave.id()"]
    dynamic_mask = split_m > 1 and any(enabled)
    if dynamic_mask:
        # With four CTAs and two row partitions, recipient groups are {0,1}
        # and {2,3}. Layout inference clears the replica bit; explicit masks
        # select the partition bit. Both must reach the descriptor's mask.
        partition_bits = -2 if mask_mode == "inferred" else 2
        checks += [
            "CHECK: [[CTA:%[^ ]+]] = {{.*}}call i32 @llvm.amdgcn.cluster.workgroup.id.x()",
            f"CHECK: [[SHIFT:%[^ ]+]] = and i32 [[CTA]], {partition_bits}",
            "CHECK: [[MASK:%[^ ]+]] = shl {{.*}}i32 3, [[SHIFT]]",
        ]
        mask_value = "[[MASK]]"
        if mask_mode == "inferred":
            checks += ["CHECK: [[CLEAN_MASK:%[^ ]+]] = and i32 [[MASK]], -327681"]
            mask_value = "[[CLEAN_MASK]]"
        checks += [f"CHECK-DAG: [[WORD:%[^ ]+]] = or {{{{.*}}}}i32 {mask_value}, 65536"]

    for multicast in dict.fromkeys(enabled):
        group = "MULTICAST" if multicast else "UNICAST"
        if multicast and dynamic_mask:
            checks += [
                "CHECK-DAG: [[PARTIAL:%[^ ]+]] = insertelement <8 x i32> {{.*}}, i32 [[WORD]], i64 0",
                f"CHECK-DAG: [[{group}:%[^ ]+]] = insertelement <8 x i32> [[PARTIAL]], i32 {{{{.*}}}}, i64 2",
            ]
        else:
            # data_size=1 (f16) occupies bit 16; all other control bits are zero.
            word = 65536 | ((1 << num_ctas) - 1 if multicast else 0)
            checks += [
                f"CHECK-DAG: [[{group}:%[^ ]+]] = insertelement <8 x i32> "
                f"<i32 {word}, {{{{.*}}}}>, i32 {{{{.*}}}}, i64 2"
            ]

    groups = ["[[MULTICAST]]" if multicast else "[[UNICAST]]" for multicast in enabled]
    if enabled[0] != enabled[1]:
        checks += [
            "CHECK: [[IS_B:%[^ ]+]] = icmp ugt i32 [[WAVE]], 1",
            f"CHECK: [[GROUP1:%[^ ]+]] = select i1 [[IS_B]], <8 x i32> {groups[1]}, <8 x i32> {groups[0]}",
        ]
        group1 = "[[GROUP1]]"
    else:
        group1 = groups[0]
    checks += [
        "CHECK: call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> {{[^,]+}}, <8 x i32> " + group1 + ",",
        "CHECK-NOT: = load {{.*}}ptr addrspace(3)",
        "CHECK: call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)",
        "CHECK-NOT: = load {{.*}}ptr addrspace(3)",
        "CHECK: call void @llvm.amdgcn.s.barrier()",
        "CHECK: = load {{.*}}ptr addrspace(3)",
    ]
    run_filecheck("tdm_fused_multicast", llvm_ir, "\n".join(checks))


@triton.jit
def _amd_independent_cluster_ids(output, SPLIT: tl.constexpr = False):
    pid = tl.program_id(0)
    rank = tlx.cluster_cta_rank()
    if SPLIT:
        tlx.amd_cluster_arrive()
        tlx.amd_cluster_wait()
    else:
        tlx.cluster_barrier()
    tl.store(output + pid, pid * 16 + rank)


def test_availability_buffer_load_to_local_rejects_gfx1250():
    with pytest.raises(RuntimeError, match="BufferLoadToLocal unsupported on target architecture"):
        compile_for_gfx1250(
            _buffer_load_to_local_contiguity_subslice_kernel,
            signature={"x_ptr": "*bf16"},
            constexprs={},
        )


def test_availability_scheduled_mfma_rejects_gfx1250():
    with pytest.raises(RuntimeError, match="supported only on CDNA3.*CDNA4"):
        compile_for_gfx1250(
            _amd_scheduled_mfma_gfx942_kernel,
            signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "output_ptr": "*fp32"},
            constexprs={"PERSISTENT": False, "TRANSPOSED": True, "INSTR_K": 16},
        )


@pytest.mark.parametrize(
    "mode, error",
    [
        (0, "requires at least one"),
        (1, "clamp_bounds requires add_offsets"),
        (2, "clamp_bounds and set_bounds are mutually exclusive"),
        (3, "add_offsets must have length 2"),
    ],
)
def test_update_tensor_descriptor_rejects_invalid_gfx1250(mode, error):
    with pytest.raises(CompilationError, match=error):
        compile_for_gfx1250(
            _invalid_update_tensor_descriptor_kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"MODE": mode},
        )


def test_local_reshape_compiles_gfx1250(device):
    """tlx.local_reshape should lower to ttg.memdesc_reshape and compile."""
    compiled = compile_for_gfx1250(
        _local_reshape_kernel,
        signature={"input_ptr": "*fp32", "output_ptr": "*fp32"},
        constexprs={"ROWS": 8, "COLS": 8},
    )
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_reshape" in ttgir, ("expected memdesc_reshape in TTGIR, got:\n" + ttgir)
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


@pytest.mark.parametrize("tiles_per_warp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_dot_scaled_tiles_per_warp_attr_gfx1250(tiles_per_warp):
    compiled = _compile_dot_scaled_tiles_per_warp(tiles_per_warp)
    m_tiles, n_tiles = tiles_per_warp
    assert f"amdg.wmma_tiles_per_warp = array<i32: {m_tiles}, {n_tiles}>" in compiled.asm["ttir"]
    # WMMA encodes tiles-per-warp in the CTA layout's register bases, not as
    # a separate tilesPerWarp field. Check that lowering used the requested hint.
    expected_cta_layout = {
        (1, 1): "ctaLayout = {warp =",
        (2, 1): "ctaLayout = {register = [[1, 0]], warp =",
        (1, 2): "ctaLayout = {register = [[0, 1]], warp =",
        (2, 2): "ctaLayout = {register = [[0, 1], [1, 0]], warp =",
    }[tiles_per_warp]
    wmma_encodings = [line for line in compiled.asm["ttgir"].splitlines() if " = #ttg.amd_wmma<" in line]
    assert wmma_encodings, compiled.asm["ttgir"]
    assert all(expected_cta_layout in encoding for encoding in wmma_encodings), wmma_encodings


@pytest.mark.parametrize(
    "tiles_per_warp, error",
    [
        ((1, ), "tiles_per_warp requires 2 entries"),
        ((0, 1), "tiles_per_warp entries must be positive"),
        ((3, 1), "tiles_per_warp entries must be powers of two"),
    ],
)
def test_dot_scaled_tiles_per_warp_rejects_invalid_gfx1250(tiles_per_warp, error):
    with pytest.raises(CompilationError, match=error):
        _compile_dot_scaled_tiles_per_warp(tiles_per_warp)


def test_require_amd_wmma_layout_compiles_gfx1250():
    compiled = compile_for_gfx1250(
        _require_amd_wmma_layout_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={"BLOCK": 256},
    )
    assert "#ttg.amd_wmma" in compiled.asm["ttgir"]


def test_mxgemm_tdm_pipelined_compiles_gfx1250(device):
    """The mxfp GEMM tutorial kernel should lower to TDM + dot_scaled + WMMA."""
    compiled = compile_for_gfx1250(
        _amd_mxfp_gemm_kernel,
        signature={
            "a_ptr": "*fp8e5",
            "b_ptr": "*fp8e5",
            "c_ptr": "*fp32",
            "a_scale": "*i8",
            "b_scale": "*i8",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i64",
            "stride_ak": "i64",
            "stride_bk": "i64",
            "stride_bn": "i64",
            "stride_cm": "i64",
            "stride_cn": "i64",
            "stride_scale": "i64",
        },
        constexprs={
            "DTYPE_A": "e5m2",
            "DTYPE_B": "e5m2",
            "SCALE_BLOCK": 32,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "TRANSPOSE_B": True,
            "NUM_BUFFERS": 2,
            "SCALE_PRESHUFFLE": True,
            "WITH_A_SCALE": True,
            "SCHEDULE": "baseline",
            "TDM_FUSION": "none",
            "L2_PREFETCH_DISTANCE": -1,
            "TDM_SPLIT": False,
            "TILES_PER_WARP": (1, 2),
        },
    )
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "tt.dot_scaled" in ttgir
    assert "amdg.wmma_tiles_per_warp = array<i32: 1, 2>" in compiled.asm["ttir"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "wmma" in amdgcn


def test_async_amd_desc_load_compiles_gfx1250(device):
    """async_amd_descriptor_load should produce TDM ops in TTGIR."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir
    assert "clamp_bounds" in ttgir
    assert "async_tdm_wait" in ttgir
    assert "local_load" in ttgir
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_async_amd_desc_load_fused_compiles_gfx1250(device):
    """Two positioned TLX descriptors lower to one fused TDM instruction."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_fused_kernel,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 16, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_fused_copy_global_to_local" in ttgir
    assert "warp_used_hints = array<i32: 3, 12>" in ttgir
    assert len(re.findall(r"tensor_load_to_lds|tensor\.load\.to\.lds", compiled.asm["amdgcn"])) == 1


def test_async_amd_desc_load_with_token_compiles_gfx1250(device):
    """async_amd_descriptor_load with token-threaded wait compiles."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_with_token_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir
    assert "async_tdm_wait" in ttgir


def test_async_amd_desc_load_pred_compiles_gfx1250(device):
    """async_amd_descriptor_load with i1 pred extends to i32."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_pred_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir


def test_async_amd_desc_store_compiles_gfx1250(device):
    """async_amd_descriptor_store produces TDM store ops in TTGIR."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_store_kernel,
        signature={"x_ptr": "*fp16", "y_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir
    assert "async_tdm_copy_local_to_global" in ttgir
    assert ttgir.count("clamp_bounds") == 2


def test_update_tensor_descriptor_store_compiles_gfx1250(device):
    compiled = compile_for_gfx1250(
        _update_tensor_descriptor_store_kernel,
        signature={"x_ptr": "*fp16", "y_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "amdg.update_tensor_descriptor" in ttgir
    assert "set_bounds =" in ttgir
    assert "pred =" in ttgir
    assert "clamp_bounds" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert ttgir.count("clamp_bounds") == 1
    # Two explicit input updates and one output update are the only descriptor
    # mutations. Neither pre-positioned copy may add a no-op update.
    assert ttgir.count("amdg.update_tensor_descriptor") == 3


def test_amd_desc_prefetch_compiles_gfx1250(device):
    """amd_descriptor_prefetch_tensor produces tdm_prefetch in TTGIR."""
    compiled = compile_for_gfx1250(
        _amd_desc_prefetch_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "tdm_prefetch" in ttgir


def test_amd_desc_prefetch_speculative_compiles_gfx1250(device):
    """amd_descriptor_prefetch_tensor with speculative=True compiles."""
    compiled = compile_for_gfx1250(
        _amd_desc_prefetch_speculative_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "tdm_prefetch" in ttgir


def test_padded_layout_local_alloc_compiles_gfx1250(device):
    """local_alloc with an explicit padded_shared_layout_encoding compiles."""

    @triton.jit
    def _kernel(x_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr):
        layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(N, 128 // 16)], [M, N])
        buf = tlx.local_alloc((M, N), tl.float16, 1, layout=layout)
        buf0 = tlx.local_view(buf, 0)
        x = tlx.local_load(buf0)
        tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)

    compiled = compile_for_gfx1250(
        _kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "padded_shared" in ttgir


def test_async_amd_desc_load_auto_propagates_padded_layout_gfx1250(device):
    """Default local_alloc + async_amd_descriptor_load auto-propagates padded encoding."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "padded_shared" in ttgir


@pytest.mark.parametrize(
    "view, expected_op",
    [(0, "ttg.memdesc_subslice"), (1, "ttg.memdesc_reshape")],
    ids=["slice", "reshape"],
)
def test_pinned_tdm_memdesc_views_compile_gfx1250(device, view, expected_op):
    compiled = compile_for_gfx1250(
        _pinned_tdm_memdesc_view_kernel,
        signature={"input_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 128, "VIEW": view},
    )
    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("amdg.async_tdm_copy_global_to_local") == 1
    assert expected_op in ttgir
    assert "#ttg.padded_shared<[128:+8]" in ttgir
    assert "#tlx.user_layout" not in ttgir
    assert "#tlx.no_verify_layout" not in ttgir
    assert "tlx.require_layout" not in ttgir
    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn


def test_amd_tdm_gemm_pipelined_compiles_gfx1250(device):
    """Compile-only: validates TDM GEMM tutorial produces TDM ops + padded encoding."""
    compiled = compile_for_gfx1250(
        _amd_tdm_gemm_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
        },
        constexprs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "amdg.tdm_prefetch" in ttgir
    assert "ttg.padded_shared" in ttgir, "expected propagated padded encoding"
    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn


@pytest.mark.parametrize("view", ["transpose", "inner_slice"])
@pytest.mark.parametrize("mode", ["load", "fused", "store"])
def test_tdm_copy_view_incompatible_gfx1250(view, mode, capfd):
    with pytest.raises(RuntimeError, match="shared layout of the tensor descriptor .* is inconsistent"):
        compile_for_gfx1250(
            _tdm_copy_view_kernel,
            signature={"input_ptr": "*fp16", "other_ptr": "*fp16", "output_ptr": "*fp16", "row": "i32"},
            constexprs={"VIEW": view, "MODE": mode, "PADDED": True},
        )
    assert "is inconsistent with the shared memory allocation layout" in capfd.readouterr().err


def test_gfx1250_matmul_tdm_pipelined_compiles():
    """Compile-only check: runs everywhere, validates the kernel still
    lowers cleanly to TDM intrinsics + a propagated padded encoding."""
    from triton.compiler.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget

    src = ASTSource(
        fn=_gfx1250_gemm.matmul_tdm_pipelined_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
        },
        constexprs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))

    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir, ("expected TDM store of C in TTGIR, got:\n" + ttgir)
    assert "amdg.tdm_prefetch" in ttgir, ("expected TDM prefetch in TTGIR, got:\n" + ttgir)
    assert ("amdg.async_tdm_wait" in ttgir) or ("amdg.async_tdm_intrinsic_wait" in ttgir)
    # Auto-propagation gives:
    #   A: [128, 32] fp16 opIdx=0 -> WMMA-tuned `[128:+8]`
    #   B: [32, 128] fp16 opIdx=1 transposed -> WMMA-tuned `[128:+16]`
    #   C: [128, 128] fp16 -> default `[128:+8]` (innermost = 128)
    # So three distinct encoding strings should be present.
    assert "ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 32]}" in ttgir, (
        "expected WMMA-tuned encoding for A, got:\n" + ttgir)
    assert "ttg.padded_shared<[128:+16] {order = [1, 0], shape = [32, 128]}" in ttgir, (
        "expected WMMA-tuned encoding for B, got:\n" + ttgir)
    assert "ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 128]}" in ttgir, (
        "expected default encoding for C, got:\n" + ttgir)

    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "tensor_store_from_lds" in amdgcn or "tensor.store.from.lds" in amdgcn, (
        "expected tensor_store_from_lds intrinsic in AMDGCN, got:\n" + amdgcn)


@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
def test_gfx1250_matmul_tdm_pipelined_single_warp_per_simd_schedule_compiles(TRANSPOSE_B):
    """Compile-only check for the TLX port of the Gluon single-warp-per-SIMD schedule."""
    from triton.compiler.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget

    src = ASTSource(
        fn=_gfx1250_gemm.matmul_tdm_pipelined_single_warp_per_simd_schedule_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*bf16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i64",
            "stride_ak": "i64",
            "stride_bk": "i64",
            "stride_bn": "i64",
            "stride_cm": "i64",
            "stride_cn": "i64",
        },
        constexprs={
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_K": 128,
            "NUM_BUFFERS": 2,
            "TRANSPOSE_B": TRANSPOSE_B,
            "L2_PREFETCH_DISTANCE": 2,
        },
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))

    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_fused_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert "amdg.tdm_prefetch" in ttgir
    assert ("amdg.async_tdm_wait" in ttgir) or ("amdg.async_tdm_intrinsic_wait" in ttgir)
    assert "tt.dot" in ttgir

    amdgcn = compiled.asm["amdgcn"]
    tensor_load_count = amdgcn.count("tensor_load_to_lds") + amdgcn.count("tensor.load.to.lds")
    tensor_store_count = amdgcn.count("tensor_store_from_lds") + amdgcn.count("tensor.store.from.lds")
    assert tensor_load_count == 3, ("expected grouped full-tile TDM loads with LDS subtile slicing, got:\n" + amdgcn)
    assert tensor_store_count == 1, ("expected one TDM store of C, got:\n" + amdgcn)


@pytest.mark.parametrize("TDM_FUSION", ["none", "2way", "4way", "partial"])
def test_gfx1250_mxgemm_tdm_pipelined_compiles(TDM_FUSION):
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    src = ASTSource(
        fn=_gfx1250_mxfp.mxgemm_tdm_pipelined_kernel,
        signature={
            "a_ptr": "*fp8e5",
            "b_ptr": "*fp8e5",
            "c_ptr": "*fp32",
            "a_scale": "*i8",
            "b_scale": "*i8",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i64",
            "stride_ak": "i64",
            "stride_bk": "i64",
            "stride_bn": "i64",
            "stride_cm": "i64",
            "stride_cn": "i64",
            "stride_scale": "i64",
        },
        constexprs={
            "DTYPE_A": "e5m2",
            "DTYPE_B": "e5m2",
            "SCALE_BLOCK": 32,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "TRANSPOSE_B": True,
            "NUM_BUFFERS": 2,
            "SCALE_PRESHUFFLE": True,
            "WITH_A_SCALE": True,
            "SCHEDULE": "baseline",
            "TDM_FUSION": TDM_FUSION,
            "L2_PREFETCH_DISTANCE": 2,
            "TDM_SPLIT": False,
        },
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    if TDM_FUSION == "none":
        assert "amdg.async_tdm_copy_global_to_local" in ttgir
        assert "amdg.async_tdm_fused_copy_global_to_local" not in ttgir
    else:
        assert "amdg.async_tdm_fused_copy_global_to_local" in ttgir
        assert "amdg.async_tdm_copy_global_to_local" not in ttgir
    if TDM_FUSION == "2way":
        assert "warp_used_hints = array<i32: 3, 12>" in ttgir
    elif TDM_FUSION == "4way":
        assert "warp_used_hints = array<i32: 1, 2, 4, 8>" in ttgir
    elif TDM_FUSION == "partial":
        assert "warp_used_hints = array<i32: 5, 10>" in ttgir
    assert "amdg.tdm_prefetch" in ttgir
    assert "tt.dot_scaled" in ttgir
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "wmma" in amdgcn


def test_gfx1250_mxgemm_tdm_split_compiles():
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    src = ASTSource(
        fn=_gfx1250_mxfp.mxgemm_tdm_pipelined_kernel,
        signature={
            "a_ptr": "*fp8e4nv",
            "b_ptr": "*u8",
            "c_ptr": "*fp32",
            "a_scale": "*i8",
            "b_scale": "*i8",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i64",
            "stride_ak": "i64",
            "stride_bk": "i64",
            "stride_bn": "i64",
            "stride_cm": "i64",
            "stride_cn": "i64",
            "stride_scale": "i64",
        },
        constexprs={
            "DTYPE_A": "e4m3",
            "DTYPE_B": "e2m1",
            "SCALE_BLOCK": 32,
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 256,
            "GROUP_SIZE_M": 8,
            "TRANSPOSE_B": True,
            "NUM_BUFFERS": 3,
            "SCALE_PRESHUFFLE": True,
            "WITH_A_SCALE": True,
            "SCHEDULE": "sliceMNK",
            "TDM_FUSION": "partial",
            "L2_PREFETCH_DISTANCE": -1,
            "TDM_SPLIT": True,
        },
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert "amdg.async_tdm_fused_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert "warp_used_hints = array<i32: 5, 10>" in ttgir
    assert "tt.dot_scaled" in ttgir
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "wmma" in amdgcn


def test_gfx1250_attn_fwd_tdm_pipelined_compiles():
    """Compile-only check: lowers cleanly to TDM intrinsics + WMMA."""
    from triton.compiler.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget

    src = ASTSource(
        fn=_gfx1250_attention.attn_fwd_tdm_pipelined_kernel,
        signature={
            "q_ptr": "*bf16",
            "k_ptr": "*bf16",
            "v_ptr": "*bf16",
            "o_ptr": "*fp32",
            "stride_qz": "i64",
            "stride_qh": "i64",
            "stride_qm": "i64",
            "stride_qk": "i64",
            "stride_kz": "i64",
            "stride_kh": "i64",
            "stride_kn": "i64",
            "stride_kk": "i64",
            "stride_vz": "i64",
            "stride_vh": "i64",
            "stride_vn": "i64",
            "stride_vk": "i64",
            "stride_oz": "i64",
            "stride_oh": "i64",
            "stride_om": "i64",
            "stride_on": "i64",
            "SM_SCALE": "constexpr",
            "SEQLEN_Q": "constexpr",
            "SEQLEN_K": "constexpr",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "HEAD_SZ": "constexpr",
        },
        constexprs={
            "SM_SCALE": 1.0 / (128**0.5),
            "SEQLEN_Q": 1024,
            "SEQLEN_K": 1024,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "HEAD_SZ": 128,
        },
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32), options={"num_warps": 4})
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir  # TDM store of O
    assert "tt.dot" in ttgir
    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "tensor_store_from_lds" in amdgcn or "tensor.store.from.lds" in amdgcn


test_gfx1250_grouped_gemm_xcd_remap_is_permutation = _gfx1250_grouped.test_grouped_gemm_xcd_remap_is_permutation

test_gfx1250_grouped_gemm_cost_model_selects_large_saturated_tile = _gfx1250_grouped.test_grouped_gemm_cost_model_selects_large_saturated_tile

test_gfx1250_grouped_gemm_cost_model_selects_small_m_tile = _gfx1250_grouped.test_grouped_gemm_cost_model_selects_small_m_tile

test_gfx1250_grouped_gemm_tdm_compiles = _gfx1250_grouped.test_grouped_gemm_tdm_compiles_gfx1250

test_gfx1250_grouped_gemm_tdm_asymmetric_alias_compiles = _gfx1250_grouped.test_grouped_gemm_tdm_asymmetric_alias_compiles_gfx1250

test_gfx1250_grouped_gemm_tdm_asymmetric_dedicated_c_compiles = _gfx1250_grouped.test_grouped_gemm_tdm_asymmetric_dedicated_c_compiles_gfx1250

test_gfx1250_grouped_gemm_tdm_cross_tile_prefetch_compiles = _gfx1250_grouped.test_grouped_gemm_tdm_cross_tile_prefetch_compiles_gfx1250


@pytest.mark.parametrize("enabled", [None, False, True])
def test_amd_wmma_queuing_compiles(enabled):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = ASTSource(_amd_wmma_queuing_kernel, signature={})
    options = {} if enabled is None else dict(enable_wmma_queuing=enabled)
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx1250", 32), options=options)
    assert compiled.metadata.enable_wmma_queuing == bool(enabled)
    assert "llvm.amdgcn.s.setreg" not in compiled.asm["ttir"]
    # A result-free intrinsic must survive optimization even in an empty kernel.
    assert compiled.asm["llir"].count("call void @llvm.amdgcn.s.setreg(i32 154, i32 1)") == int(bool(enabled))
    assert "asm sideeffect" not in compiled.asm["llir"]
    instruction = "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1"
    assert compiled.asm["amdgcn"].count(instruction) == int(bool(enabled))
    assert "s_mov_b32" not in compiled.asm["amdgcn"]


def test_amd_wmma_queuing_cache_key():
    from triton.backends.amd.compiler import HIPBackend
    from triton.backends.compiler import GPUTarget

    backend = HIPBackend(GPUTarget("hip", "gfx1250", 32))
    default = backend.parse_options({})
    disabled = backend.parse_options(dict(enable_wmma_queuing=False))
    enabled = backend.parse_options(dict(enable_wmma_queuing=True))
    assert default.hash() == disabled.hash()
    assert enabled.hash() != disabled.hash()


@pytest.mark.parametrize("value", [0, 1, -1, 1.0, "true"])
def test_amd_wmma_queuing_invalid_option(value):
    from triton.backends.amd.compiler import HIPOptions

    with pytest.raises(ValueError, match="enable_wmma_queuing must be a bool"):
        HIPOptions(arch="gfx1250", enable_wmma_queuing=value)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1200", "gfx1201"])
def test_amd_wmma_queuing_unsupported_target(arch):
    from triton.backends.amd.compiler import HIPOptions

    assert not HIPOptions(arch=arch).enable_wmma_queuing
    with pytest.raises(ValueError, match="requires an AMD gfx1250 target"):
        HIPOptions(arch=arch, enable_wmma_queuing=True)


@pytest.mark.parametrize("independent_ctas", [False, True])
@pytest.mark.parametrize("cluster_size,masks,message", [
    (4, (-1, 3), "outside the cluster"),
    (4, (16, 3), "outside the cluster"),
    (4, (3, ), "one multicast mask per member"),
    (4, (1.5, 3), "scalar integers"),
    (8, (63, 3), "at most 5 recipients"),
])
def test_tdm_fused_multicast_invalid_masks(cluster_size, masks, message, independent_ctas):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.errors import CompilationError
    source = ASTSource(_async_amd_desc_load_fused_kernel,
                       signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "output_ptr":
                                  "*fp16"}, constexprs={"M": 64, "N": 64, "MULTICAST_MASKS": masks})
    options = dict(ctas_per_cga=(cluster_size, 1, 1)) if independent_ctas else dict(num_ctas=cluster_size)
    with pytest.raises(CompilationError, match=message):
        triton.compile(source, target=GPUTarget("hip", "gfx1250", 32), options=dict(num_warps=4, **options))


@pytest.mark.parametrize("num_ctas,split_m", [(2, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize("mask_mode", ["inferred", "explicit", "explicit_b", "disabled"])
def test_tdm_fused_multicast_distributed_compiles(num_ctas, split_m, mask_mode):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = ASTSource(_async_amd_desc_load_fused_kernel,
                       signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "output_ptr":
                                  "*fp16"}, constexprs=_tdm_fused_distributed_constants(num_ctas, split_m, mask_mode))
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx1250", 32),
                              options=dict(num_warps=4, num_ctas=num_ctas))
    assert compiled.metadata.num_ctas == num_ctas
    assert (" multicast " in compiled.asm["ttgir"]) == (mask_mode != "inferred")
    assert compiled.asm["amdgcn"].count("tensor_load_to_lds") == 1
    if split_m > 1:
        assert "CGALayout = [[0, 0], [1, 0]]" in compiled.asm["ttgir"]
    assert "amdg.cluster_barrier" not in compiled.asm["ttgir"]
    assert "s_barrier_signal -3" not in compiled.asm["amdgcn"]
    assert "s_barrier_wait -3" not in compiled.asm["amdgcn"]
    _check_tdm_fused_multicast_codegen(compiled.asm["llir"], num_ctas, split_m, mask_mode)


@pytest.mark.parametrize("cluster_size,multicast", [(1, False), (2, True), (4, True), (4, False)])
@pytest.mark.parametrize("group_size", [1, 2])
@pytest.mark.parametrize("interface", ["packed", "list"])
@pytest.mark.parametrize("enable_wmma_queuing", [False, True])
def test_grouped_gemm_multicast_compiles(cluster_size, multicast, group_size, interface, enable_wmma_queuing):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton import knobs

    constants = dict(NUM_PROGRAMS=16, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=4, NUM_BUFFERS=2,
                     L2_PREFETCH_DISTANCE=0, C_STAGING_MODE=0, CROSS_TILE_PREFETCH=True, XCD_REMAP_MODE=2, NUM_XCDS=8,
                     XCD_CHUNK=2, K=512, group_size=group_size, CLUSTER_SIZE=cluster_size, CLUSTER_MULTICAST=multicast)
    if interface == "packed":
        kernel = _gfx1250_grouped.grouped_gemm_tdm_kernel
        signature = _gfx1250_grouped._grouped_gemm_tdm_compile_signature()
        attrs = _gfx1250_grouped._grouped_gemm_tdm_compile_attrs()
    else:
        kernel = _gfx1250_grouped.grouped_gemm_tdm_list_kernel
        constants["ALLOW_MN_TAILS"] = False
        signature = _gfx1250_grouped._grouped_gemm_tdm_list_compile_signature()
        attrs = _gfx1250_grouped._grouped_gemm_tdm_list_compile_attrs()
    source = ASTSource(kernel, signature={k: v
                                          for k, v in signature.items()
                                          if k not in constants}, constexprs=constants, attrs=attrs)
    assert knobs.runtime.add_stages_inspection_hook is None
    compiled = triton.compile(
        source, target=GPUTarget("hip", "gfx1250", 32),
        options=dict(num_warps=4, waves_per_eu=1, ctas_per_cga=(cluster_size, 1, 1),
                     enable_wmma_queuing=enable_wmma_queuing))
    assert compiled.metadata.num_ctas == 1
    assert compiled.metadata.ctas_per_cga == (cluster_size, 1, 1)
    assert "llvm.amdgcn.s.setreg" not in compiled.asm["ttir"]
    assert compiled.asm["llir"].count("call void @llvm.amdgcn.s.setreg(i32 154, i32 1)") == int(enable_wmma_queuing)
    instruction = "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1"
    assert compiled.asm["amdgcn"].count(instruction) == int(enable_wmma_queuing)
    if enable_wmma_queuing:
        assert compiled.asm["amdgcn"].index(instruction) < compiled.asm["amdgcn"].index("v_wmma_")
    assert f'"amdgpu-cluster-dims"="{cluster_size},1,1"' in compiled.asm["llir"]
    assert compiled.metadata.global_scratch_size == 0
    assert compiled.metadata.shared <= 320 * 1024
    assert ("amdg.cluster_barrier_arrive" in compiled.asm["ttgir"]) == (cluster_size > 1)
    assert ("amdg.cluster_barrier_wait" in compiled.asm["ttgir"]) == (cluster_size > 1)
    assert (" multicast " in compiled.asm["ttgir"]) == (multicast and cluster_size > 1)


@pytest.mark.parametrize("dims,num_ctas,arch", [
    ((3, 1, 1), 1, "gfx1250"),
    ((4, 2, 1), 1, "gfx1250"),
    ((4, 1), 1, "gfx1250"),
    ((4, 1, 1), 2, "gfx1250"),
    ((4, 1, 1), 1, "gfx950"),
])
def test_amd_independent_cluster_invalid_options(dims, num_ctas, arch):
    from triton.backends.amd.compiler import HIPOptions
    with pytest.raises(ValueError, match="ctas_per_cga"):
        HIPOptions(arch=arch, num_ctas=num_ctas, ctas_per_cga=dims)


@pytest.mark.parametrize("m_list,n,num_programs,valid", [
    ([4096, 4096], 4096, 32, True),
    ([4096, 4096], 4096, 256, True),
    ([4096, 2048], 4096, 32, False),
    ([0, 4096], 4096, 32, False),
    ([4096, 4096], 768, 32, False),
    ([4096, 4096], 4096, 24, False),
    ([4096, 4096], 4096, 48, False),
])
def test_grouped_gemm_multicast_config_validation(m_list, n, num_programs, valid):

    def validate():
        _gfx1250_grouped._validate_grouped_gemm_cluster_config(m_list, n, num_programs, block_m=256, block_n=256,
                                                               block_k=128, group_m=4, tdm_pipeline_depth=2,
                                                               l2_prefetch_distance=0, c_staging_mode=0,
                                                               cross_tile_prefetch=True, auto_config=False,
                                                               xcd_remap_mode="chunked", num_xcds=8, xcd_chunk=2)

    if valid:
        validate()
    else:
        with pytest.raises(ValueError):
            validate()


@pytest.mark.parametrize("arch,cluster_size", [
    ("gfx1250", None),
    ("gfx1250", 1),
    ("gfx1250", 2),
    ("gfx1250", 4),
    ("gfx950", None),
    ("gfx950", 1),
])
@pytest.mark.parametrize("split", [False, True])
def test_amd_cluster_barrier_compiles(arch, cluster_size, split):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    options = dict(num_warps=4)
    if cluster_size is not None:
        options["ctas_per_cga"] = (cluster_size, 1, 1)
    source = ASTSource(_amd_independent_cluster_ids, signature={"output": "*i32"}, constexprs={"SPLIT": split})
    compiled = triton.compile(source, target=GPUTarget("hip", arch, 32 if arch == "gfx1250" else 64), options=options)
    ttgir = compiled.asm["ttgir"]
    assert ("ttg.barrier local" in ttgir) == (not split)
    assert "amdg.cluster_barrier_arrive" in ttgir
    assert "amdg.cluster_barrier_wait" in ttgir
    assert ("s_barrier" in compiled.asm["amdgcn"]) == (not split or (cluster_size is not None and cluster_size > 1))
    multi_cta = cluster_size is not None and cluster_size > 1
    assert ("s_barrier_signal -3" in compiled.asm["amdgcn"]) == multi_cta
    assert ("s_barrier_wait -3" in compiled.asm["amdgcn"]) == multi_cta


@pytest.mark.parametrize("staging,cross,tdm_stores", [(0, False, 1), (0, True, 8), (1, True, 0)])
def test_grouped_gemm_list_tails_preserve_output_config(staging, cross, tdm_stores):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    constants = dict(K=512, group_size=3, NUM_PROGRAMS=2, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=4,
                     NUM_BUFFERS=2, L2_PREFETCH_DISTANCE=0, C_STAGING_MODE=staging, CROSS_TILE_PREFETCH=cross,
                     XCD_REMAP_MODE=0, NUM_XCDS=8, XCD_CHUNK=2, CLUSTER_SIZE=1, CLUSTER_MULTICAST=True,
                     ALLOW_MN_TAILS=True)
    signature = {
        k: v
        for k, v in _gfx1250_grouped._grouped_gemm_tdm_list_compile_signature().items()
        if k not in constants
    }
    source = ASTSource(_gfx1250_grouped.grouped_gemm_tdm_list_kernel, signature=signature, constexprs=constants,
                       attrs=_gfx1250_grouped._grouped_gemm_tdm_list_compile_attrs())
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx1250", 32),
                              options=dict(num_warps=4, waves_per_eu=1, enable_wmma_queuing=True))
    asm = compiled.asm["amdgcn"]
    assert asm.count("tensor_store_from_lds") == tdm_stores
    assert ("global_store" in asm) == (tdm_stores == 0)
    assert compiled.metadata.global_scratch_size == 0


@pytest.mark.parametrize("dimension", ["N", "K"])
def test_grouped_gemm_list_rejects_ragged_nk(dimension):
    shapes = [(128, 256, 256), (256, 384 if dimension == "N" else 256, 384 if dimension == "K" else 256)]
    group_a = [torch.empty((m, k), dtype=torch.float16) for m, n, k in shapes]
    group_b = [torch.empty((n, k), dtype=torch.float16).T for m, n, k in shapes]
    with pytest.raises(ValueError, match="same N and K"):
        _gfx1250_grouped.grouped_gemm_tdm_list(group_a, group_b)


@pytest.mark.parametrize("config,k,match", [
    ({"TILE_MODE": 0}, 512, "unsupported.*config keys"),
    ({"HAS_K_TAIL": True}, 512, "unsupported.*config keys"),
    ({"BLOCK_SIZE_M": 64}, 512, "BLOCK_SIZE_M"),
    ({"BLOCK_SIZE_N": 192}, 512, "BLOCK_SIZE_N"),
    ({"BLOCK_SIZE_K": 64}, 512, "BLOCK_SIZE_K"),
    ({"num_warps": 8}, 512, "num_warps"),
    ({"waves_per_eu": 2}, 512, "waves_per_eu"),
    ({"CLUSTER_SIZE": 3}, 512, "CLUSTER_SIZE"),
    ({"NUM_BUFFERS": 1}, 512, "NUM_BUFFERS"),
    ({"NUM_BUFFERS": 3}, 512, "cross-tile prefetch requires depth 2"),
    ({"NUM_BUFFERS": 3, "C_STAGING_MODE": 1, "CROSS_TILE_PREFETCH": False}, 512, "full C buffer"),
    ({"BLOCK_SIZE_M": 128}, 512, "alias-C cross-tile"),
    ({"C_STAGING_MODE": 2}, 512, "C_STAGING_MODE"),
    ({"XCD_REMAP_MODE": 3}, 512, "XCD_REMAP_MODE"),
    ({"GROUP_SIZE_M": 0}, 512, "GROUP_SIZE_M"),
    ({"NUM_XCDS": 0}, 512, "NUM_XCDS"),
    ({"XCD_CHUNK": 0}, 512, "XCD_CHUNK"),
    ({"NUM_SM": 0}, 512, "NUM_SM"),
    ({"NUM_SM": 1.5}, 512, "NUM_SM must be an int"),
    ({"NUM_SM": True}, 512, "NUM_SM must be an int"),
    ({"CROSS_TILE_PREFETCH": 1}, 512, "must be a bool"),
    ({"L2_PREFETCH_DISTANCE": -1}, 512, "nonnegative"),
    ({"L2_PREFETCH_DISTANCE": 5}, 512, "number of K blocks"),
    ({"XCD_CHUNK": 1 << 30}, 512, "NUM_XCDS \\* XCD_CHUNK"),
    ({"GROUP_SIZE_M": 1 << 30}, 512, "GROUP_SIZE_M \\* number of N tiles"),
    ({"NUM_BUFFERS": 4, "CROSS_TILE_PREFETCH": False}, 512, "unsupported gfx1250 TDM schedule"),
    ({}, 264, "full K tiles"),
    ({}, 128, "fill the TDM pipeline"),
    ({}, 384, "even number of K blocks"),
    ([], 512, "config must be a dict"),
])
def test_grouped_gemm_interface_rejects_config(config, k, match, monkeypatch):
    a = torch.empty((1, k), dtype=torch.float16)
    b = torch.empty((257, k), dtype=torch.float16).T

    def no_allocation(*args, **kwargs):
        pytest.fail("invalid configuration reached device allocation")

    monkeypatch.setattr(torch, "empty", no_allocation)
    monkeypatch.setattr(torch, "tensor", no_allocation)
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped._make_grouped_gemm_args([a], [b], config)


@pytest.mark.parametrize("condition,match", [
    ("ragged_n", "same N and K"),
    ("ragged_k", "same N and K"),
    ("dtype", "FP16"),
    ("row_major_b", "canonical strides"),
    ("padded_a", "canonical strides"),
    ("unaligned_a", "16-byte aligned"),
    ("grad", "forward-only"),
    ("empty_groups", "same nonzero length"),
    ("length", "same nonzero length"),
    ("rank", "rank-2"),
    ("non_tensor", "rank-2"),
    ("k_mismatch", "mismatched reduction"),
    ("empty_n", "positive"),
    ("cpu", "requires an AMD gfx1250"),
])
def test_grouped_gemm_interface_rejects_inputs(condition, match, monkeypatch):
    a = [torch.empty((1, 512), dtype=torch.float16) for _ in range(2)]
    b = [torch.empty((257, 512), dtype=torch.float16).T for _ in range(2)]
    if condition == "ragged_n":
        b[1] = torch.empty((256, 512), dtype=torch.float16).T
    elif condition == "ragged_k":
        a[1] = torch.empty((1, 256), dtype=torch.float16)
        b[1] = torch.empty((257, 256), dtype=torch.float16).T
    elif condition == "dtype":
        a[0] = a[0].float()
    elif condition == "row_major_b":
        b[0] = b[0].contiguous()
    elif condition == "padded_a":
        a[0] = torch.empty((1, 520), dtype=torch.float16)[:, :512]
    elif condition == "unaligned_a":
        a[0] = torch.empty(513, dtype=torch.float16)[1:].view(1, 512)
    elif condition == "grad":
        a[0].requires_grad_()
    elif condition == "empty_groups":
        a, b = [], []
    elif condition == "length":
        b.pop()
    elif condition == "rank":
        a[0] = a[0].view(-1)
    elif condition == "non_tensor":
        a[0] = None
    elif condition == "k_mismatch":
        b[0] = b[0][:-1]
    elif condition == "empty_n":
        b[0] = b[0][:, :0]

    def no_allocation(*args, **kwargs):
        pytest.fail("invalid input reached output/metadata allocation")

    monkeypatch.setattr(torch, "empty", no_allocation)
    monkeypatch.setattr(torch, "tensor", no_allocation)
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped.grouped_gemm(a, b)


def test_grouped_gemm_interface_rejects_other_arch(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(gcnArchName="gfx950:sramecc+"))
    with pytest.raises(ValueError, match="requires gfx1250; got gfx950"):
        _gfx1250_grouped._require_grouped_gemm_gfx1250(torch.device("cuda"))


def test_grouped_gemm_interface_defaults_match_bench(monkeypatch):
    import sys
    from triton.language.extra.tlx.tutorials.amd_grouped_gemm_gfx1250 import bench

    checked = []

    def check_case(args, case, kernel_variant, index, total):
        cfg = _gfx1250_grouped._grouped_gemm_interface_launch_config(
            _gfx1250_grouped._grouped_gemm_interface_config(None, case.k), [case.m] * case.groups, case.n, case.k, 256,
            auto_cluster=True)
        for key, value in dict(
                BLOCK_SIZE_M=args.block_m,
                BLOCK_SIZE_N=args.block_n,
                BLOCK_SIZE_K=args.block_k,
                GROUP_SIZE_M=args.group_m,
                NUM_BUFFERS=args.tdm_pipeline_depth,
                L2_PREFETCH_DISTANCE=args.l2_prefetch_distance,
                C_STAGING_MODE=int(args.dedicated_c_buffer),
                CROSS_TILE_PREFETCH=args.cross_tile_prefetch,
                XCD_REMAP_MODE=_gfx1250_grouped._XCD_REMAP_MODES[args.xcd_remap],
                NUM_XCDS=args.num_xcds,
                XCD_CHUNK=args.xcd_chunk,
                CLUSTER_SIZE=args.cluster_size,
                CLUSTER_MULTICAST=args.cluster_multicast,
        ).items():
            assert cfg[key] == value, (case, key)
        assert cfg["ALLOW_MN_TAILS"] is False
        assert cfg["NUM_SM"] == 256
        checked.append(case)
        return bench.BenchResult(case, kernel_variant, case.tensor_gib, 0, None, None)

    monkeypatch.setattr(sys, "argv", ["bench.py", "--dry-run"])
    monkeypatch.setattr(bench, "_run_case", check_case)
    monkeypatch.setattr(bench, "_print_summary", lambda results: None)
    assert bench.main() == 0
    assert len(checked) == 2 * len(bench.DEFAULT_CASES)


@pytest.mark.parametrize("ms,n,num_sm,config,expected", [
    ([2048, 2048], 1024, 16, {}, 4),
    ([2048, 2048], 1024, 16, {"CLUSTER_SIZE": 1}, 1),
    ([2048, 2048], 1024, 16, {"CLUSTER_SIZE": 2}, 2),
    ([2048, 2048], 1024, 16, {"CLUSTER_SIZE": 4}, 4),
    ([2048, 2048], 1024, 16, {"CROSS_TILE_PREFETCH": False}, 1),
    ([2048, 2048], 1024, 16, {"C_STAGING_MODE": 1}, 1),
    ([2048, 2048], 1024, 16, {"XCD_REMAP_MODE": 0}, 1),
    ([2048, 2048], 1024, 24, {}, 1),
    ([3072, 3072], 1024, 32, {}, 1),
    ([2048, 1024], 1024, 16, {}, 1),
    ([0, 2048], 1024, 16, {}, 1),
    ([0, 0], 1024, 16, {}, 1),
    ([257, 513], 257, 16, {}, 1),
    ([1024], 512, 32, {}, 1),
])
def test_grouped_gemm_interface_selects_cluster(ms, n, num_sm, config, expected):
    cfg = _gfx1250_grouped._grouped_gemm_interface_launch_config(
        _gfx1250_grouped._grouped_gemm_interface_config(config, 512), ms, n, 512, num_sm, auto_cluster="CLUSTER_SIZE"
        not in config)
    assert cfg["CLUSTER_SIZE"] == expected
    if expected == 1 and "CLUSTER_SIZE" not in config:
        explicit = dict(config, CLUSTER_SIZE=4)
        with pytest.raises(ValueError, match="clustering requires"):
            _gfx1250_grouped._grouped_gemm_interface_launch_config(
                _gfx1250_grouped._grouped_gemm_interface_config(explicit, 512), ms, n, 512, num_sm, auto_cluster=False)


@pytest.mark.parametrize("condition,match", [
    ("dtype", "metadata table 0"),
    ("shape", "metadata table 3"),
    ("strides", "metadata table 4"),
    ("count", "positive int32 group count"),
    ("config", "cfg must include"),
    ("tail_flag", "ALLOW_MN_TAILS must be a bool"),
    ("cluster_tails", "clustered launches require ALLOW_MN_TAILS=False"),
    ("cluster_grid", "positive and divisible by 16"),
    ("device", "requires an AMD gfx1250"),
])
def test_grouped_gemm_interface_rejects_metadata(condition, match):
    tables = [torch.empty(2, dtype=torch.int64) for _ in range(3)]
    tables += [torch.empty(6, dtype=torch.int32) for _ in range(2)]
    g, cfg = 2, dict(K=512, NUM_SM=2, ALLOW_MN_TAILS=True, CLUSTER_SIZE=1)
    if condition == "dtype":
        tables[0] = tables[0].int()
    elif condition == "shape":
        tables[3] = tables[3][:-1]
    elif condition == "strides":
        tables[4] = torch.empty(12, dtype=torch.int32)[::2]
    elif condition == "count":
        g = 0
    elif condition == "config":
        cfg = {}
    elif condition == "tail_flag":
        cfg["ALLOW_MN_TAILS"] = 1
    elif condition == "cluster_tails":
        cfg.update(CLUSTER_SIZE=4, NUM_SM=16)
    elif condition == "cluster_grid":
        cfg.update(CLUSTER_SIZE=4, ALLOW_MN_TAILS=False)
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped._perf_fn(*tables, g, cfg)


@pytest.mark.parametrize("m,n,k,match", [
    (128, 256, 264, "full K tiles"),
    (128, 256, 128, "fill the TDM pipeline"),
])
def test_grouped_gemm_list_rejects_unsupported_tiles(m, n, k, match):
    a = torch.empty((m, k), dtype=torch.float16)
    b = torch.empty((n, k), dtype=torch.float16).T
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped.grouped_gemm_tdm_list([a], [b], num_programs=2)


def test_grouped_gemm_list_rejects_clustered_tails():
    a = torch.empty((257, 256), dtype=torch.float16)
    b = torch.empty((257, 256), dtype=torch.float16).T
    with pytest.raises(ValueError, match="tails require cluster_size=1"):
        _gfx1250_grouped.grouped_gemm_tdm_list([a], [b], block_m=256, block_n=256, num_programs=2, c_staging_mode=0,
                                               cross_tile_prefetch=True, cluster_size=2)


def test_grouped_gemm_list_tail_config_selection():
    cfg = _gfx1250_grouped._pick_grouped_gemm_config([1, 257], 257, 512, 2, allow_mn_tails=True)
    assert cfg["block_k"] == 128
    with pytest.raises(ValueError, match="no validated"):
        _gfx1250_grouped._pick_grouped_gemm_config([1, 257], 257, 512, 2)


@pytest.mark.parametrize("condition", ["row_major_b", "padded_a", "unaligned_a", "empty_k"])
def test_grouped_gemm_list_rejects_layout(condition):
    a = torch.empty((128, 256), dtype=torch.float16)
    b = torch.empty((256, 256), dtype=torch.float16).T
    match = "canonical strides"
    if condition == "row_major_b":
        b = b.T
    elif condition == "padded_a":
        a = torch.empty((128, 264), dtype=torch.float16)[:, :256]
    elif condition == "unaligned_a":
        a = torch.empty(128 * 256 + 1, dtype=torch.float16)[1:].view(128, 256)
        match = "16-byte aligned"
    else:
        a = a[:, :0]
        b = b[:0]
        match = "positive"
    with pytest.raises(ValueError, match=match):
        _gfx1250_grouped.grouped_gemm_tdm_list([a], [b])


@pytest.mark.parametrize('partitions,groups,dimension,message', [
    (3, 1, 0, 'num_partitions'),
    (2, 0, 0, 'num_groups'),
    (2, 3, 0, 'num_groups'),
    (2, 1, -1, 'partition_dim'),
])
def test_partitioned_shared_invalid_parameters(partitions, groups, dimension, message):
    inner = tlx.padded_shared_layout_encoding.with_identity_for([(128, 8)], [64, 128])
    with pytest.raises(ValueError, match=message):
        tlx.partitioned_shared_layout_encoding(partitions, groups, dimension, inner)


def test_partitioned_shared_permute_preserves_bases():
    inner = tlx.padded_shared_layout_encoding.with_bases([(8, 2)], [[1, 0], [0, 1], [2, 0], [0, 2]], [4, 4])
    original = tlx.partitioned_shared_layout_encoding(2, 1, 0, inner)
    transposed = original.make_permute([1, 0])
    assert transposed.partition_dim == 1
    assert transposed.partition_layout.offset_bases == [[0, 1], [1, 0], [0, 2], [2, 0]]
    assert transposed.make_permute([1, 0]).partition_layout.offset_bases == inner.offset_bases
