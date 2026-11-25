"""Rubin (SM107) TLX compile-only tests: PTX is checked, nothing is launched."""
import re

import pytest
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend


def _compile_to_ptx(kernel, signature, constexprs, capability):
    # Stop at PTX: Rubin-only PTX needs a PTX 9.4 assembler for sm_107a.
    target = GPUTarget("cuda", capability, 32)
    src = ASTSource(
        fn=kernel,
        signature=signature,
        constexprs=constexprs,
        attrs={(i, ): [("tt.divisibility", 16)]
               for i in range(len(signature))},
    )
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4, **src.parse_options()})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                         context)
    metadata = {"target": target, **options.__dict__}
    stages = {}
    backend.add_stages(stages, options, src.language)
    for stage_name, compile_stage in stages.items():
        module = compile_stage(module, metadata)
        if stage_name == "ptx":
            return module
    raise AssertionError("no ptx stage")


@triton.jit
def _shared_b_dots_kernel(a_ptr, b_ptr, c_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a0_smem = tlx.local_view(tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1)), 0)
    a1_smem = tlx.local_view(tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1)), 0)
    b_smem = tlx.local_view(tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, tl.constexpr(1)), 0)
    a_ptrs = a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :]
    tlx.async_load(a_ptrs, a0_smem)
    tlx.async_load(a_ptrs + BLOCK_M * BLOCK_K, a1_smem)
    tlx.async_load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :], b_smem)
    tlx.async_load_commit_group()
    tlx.async_load_wait_group(tl.constexpr(0))

    acc = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(2), tlx.storage_kind.tmem)
    acc0 = tlx.local_view(acc, 0)
    acc1 = tlx.local_view(acc, 1)
    bar = tlx.local_view(tlx.alloc_barriers(tl.constexpr(1)), 0)
    # Rows [0, M) and [M, 2M) of one 2M x N product: different A, same B.
    tlx.async_dot(a0_smem, b_smem, acc0, use_acc=False, mBarriers=[], force_async=True)
    tlx.async_dot(a1_smem, b_smem, acc1, use_acc=False, mBarriers=[bar])
    tlx.barrier_wait(bar, tl.constexpr(0))

    c_ptrs = c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    tl.store(c_ptrs, tlx.local_load(acc0))
    tl.store(c_ptrs + BLOCK_M * BLOCK_N, tlx.local_load(acc1))


def _compile_shared_b_dots(capability):
    return _compile_to_ptx(
        _shared_b_dots_kernel,
        {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
        capability,
    )


def _collectors(ptx):
    # (A action, B action) per f16 tcgen05.mma, whatever order the modifiers come in.
    def action(modifiers, operand):
        match = re.search(rf"\.collector::{operand}::(\w+)", modifiers)
        return match.group(1) if match else ""

    return [(action(m, "a"), action(m, "b")) for m in re.findall(r"tcgen05\.mma\.cta_group::1\.kind::f16(\S*)\s", ptx)]


def test_shared_b_dots_use_b_collector_on_sm107():
    ptx = _compile_shared_b_dots(107)
    assert ".target sm_107a" in ptx
    # Each 16-wide K slice: fill B for the first dot, last-use it for the second.
    assert _collectors(ptx) == [("", "fill"), ("", "lastuse")] * 2


@pytest.mark.parametrize("capability", [100, 103])
def test_shared_b_dots_skip_b_collector_before_sm107(capability):
    ptx = _compile_shared_b_dots(capability)
    assert ".collector::b" not in ptx
    assert _collectors(ptx) == [("", "")] * 4
