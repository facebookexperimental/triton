import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent_mxfp8 import (
    attention as _attention_ws_pipelined_persistent_mxfp8,
    attention_bwd as _attention_bwd_mxfp8,
    generate_attention_inputs,
    _attn_fwd_mxf8_ws,
    _mxf8_host_descriptor_pre_hook,
)
from triton.language.extra.tlx.tutorials.testing.test_correctness import (
    _quantize_mxfp8_bwd_operand,
    FlashAttention,
)
from triton.tools.tensor_descriptor import TensorDescriptor

from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()
"""
Benchmarks for the TLX MXFP8 flash attention forward and backward kernels.
Run with: third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_fa_mxfp8_perf.py --mode fwd
          third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_fa_mxfp8_perf.py --mode bwd

Facebook: If you are developing in fbsource, use tritonbench instead to collect perf numbers.
"""


def _setup_bwd_inputs(shape, sm_scale, dtype):
    Z, H, N_CTX, HEAD_DIM = shape
    (q, q_scale, q_ref), (k, k_scale, k_ref), (v, v_scale, v_ref) = (generate_attention_inputs(shape, DEVICE, dtype))

    q_dk, q_scale_dk = _quantize_mxfp8_bwd_operand(q_ref, dtype, transpose_for_reduction=True)
    k_dq, k_scale_dq = _quantize_mxfp8_bwd_operand(k_ref, dtype, transpose_for_reduction=True)
    v_bwd, v_scale_bwd = _quantize_mxfp8_bwd_operand(v_ref, dtype)
    do_bf16 = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    do_fp8, do_scale = _quantize_mxfp8_bwd_operand(do_bf16, dtype)
    do_fp8_dv, do_scale_dv = _quantize_mxfp8_bwd_operand(do_bf16, dtype, transpose_for_reduction=True)

    fwd_config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_mxfp8"]
    y_dim = Z * H * N_CTX
    o = torch.empty(shape, device=DEVICE, dtype=torch.bfloat16)
    M = torch.empty((Z, H, N_CTX), device=DEVICE, dtype=torch.float32)
    dummy_block = [1, 1]
    dummy_5d = [1, 1, 1, 1, 1]

    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_q_scale = TensorDescriptor.from_tensor(q_scale, block_shape=dummy_5d)
    desc_k_scale = TensorDescriptor.from_tensor(k_scale, block_shape=dummy_5d)
    desc_v_scale = TensorDescriptor.from_tensor(v_scale, block_shape=dummy_5d)
    nargs = {
        **fwd_config,
        "HEAD_DIM": HEAD_DIM,
        "desc_q": desc_q,
        "desc_k": desc_k,
        "desc_v": desc_v,
        "desc_o": desc_o,
        "desc_q_scale": desc_q_scale,
        "desc_k_scale": desc_k_scale,
        "desc_v_scale": desc_v_scale,
    }
    _mxf8_host_descriptor_pre_hook(nargs)

    def alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (min(num_sms, triton.cdiv(N_CTX, fwd_config["BLOCK_M"]) * Z * H), 1, 1)
    _attn_fwd_mxf8_ws.fn[grid](
        sm_scale,
        M,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        desc_q_scale,
        desc_k_scale,
        desc_v_scale,
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM,
        STAGE=1,
        num_stages=1,
        num_warps=4,
        **fwd_config,
    )

    return {
        "do_fp8": do_fp8,
        "do_fp8_dv": do_fp8_dv,
        "do_bf16": do_bf16,
        "q": q,
        "q_dk": q_dk,
        "k": k,
        "k_dq": k_dq,
        "v_bwd": v_bwd,
        "o": o,
        "M": M,
        "q_scale": q_scale,
        "q_scale_dk": q_scale_dk,
        "k_scale": k_scale,
        "k_scale_dq": k_scale_dq,
        "v_scale_bwd": v_scale_bwd,
        "do_scale": do_scale,
        "do_scale_dv": do_scale_dv,
    }


def create_benchmark(mode="fwd"):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=["ws_pipelined_persistent_mxfp8"],
            line_names=["ws_pipelined_persistent_mxfp8"],
            ylabel="TFLOPS",
            plot_name=f"flash-attention-{mode}-performance-mxfp8-d128",
            args={"BATCH": 4, "H": 32, "HEAD_DIM": 128, "causal": False},
        ))
    def benchmark(BATCH, H, N_CTX, HEAD_DIM, causal, provider):
        shape = (BATCH, H, N_CTX, HEAD_DIM)
        sm_scale = 1.3
        quantiles = [0.5, 0.2, 0.8]
        dtype = torch.float8_e4m3fn

        if mode == "bwd":
            bwd = _setup_bwd_inputs(shape, sm_scale, dtype)
            fn = lambda: _attention_bwd_mxfp8(
                bwd["do_fp8"],
                bwd["do_fp8_dv"],
                bwd["q"],
                bwd["q_dk"],
                bwd["k"],
                bwd["k_dq"],
                bwd["v_bwd"],
                bwd["o"],
                bwd["M"],
                bwd["q_scale"],
                bwd["q_scale_dk"],
                bwd["k_scale"],
                bwd["k_scale_dq"],
                bwd["v_scale_bwd"],
                bwd["do_scale"],
                bwd["do_scale_dv"],
                sm_scale,
                do_bf16=bwd["do_bf16"],
            )
        else:
            (q, q_scale, _), (k, k_scale, _), (v, v_scale, _) = (generate_attention_inputs(shape, DEVICE, dtype))
            fn = lambda: _attention_ws_pipelined_persistent_mxfp8(q, k, v, q_scale, k_scale, v_scale, sm_scale, causal)

        ms, min_ms, max_ms = triton.testing.do_bench(
            fn,
            quantiles=quantiles,
            warmup=500,
            rep=500,
        )

        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        # fwd: 2 matmuls (QK, PV). bwd: 5 matmuls (QK^T, V·dO^T, P^T·dO, dS^T·Q, dS_trans·K)
        total_flops = 2 * flops_per_matmul if mode == "fwd" else 5 * flops_per_matmul
        perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX Blackwell MXFP8 Flash Attention")
    parser.add_argument(
        "--mode",
        type=str,
        default="fwd",
        choices=["fwd", "bwd"],
        help="Benchmark forward or backward pass (default: fwd)",
    )
    args = parser.parse_args()

    if is_blackwell():
        print(f"Running MXFP8 flash attention {args.mode} benchmark")
        benchmark = create_benchmark(mode=args.mode)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
