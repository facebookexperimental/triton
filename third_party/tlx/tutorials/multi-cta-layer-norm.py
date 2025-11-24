# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._inductor.runtime.triton_compat import libdevice

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float32


########################## HELION #############################
@triton.autotune(
    configs=[triton.Config({"_BLOCK_SIZE_0": 1, "_REDUCTION_BLOCK_1": 4096}, num_warps=16, num_stages=3)],
    key=["x"],
    
)
@triton.jit
def _helion_layer_norm_fwd(
    x, mean, rstd, weight, bias, out, eps, _BLOCK_SIZE_0: tl.constexpr, _REDUCTION_BLOCK_1: tl.constexpr
):
    # src[layer_norm.py:297]: for tile_m in hl.tile(m):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    # src[layer_norm.py:300]: mean_val = torch.sum(acc, dim=-1) / n
    sum_1_acc = tl.full([_BLOCK_SIZE_0, _REDUCTION_BLOCK_1], 0, tl.float32)
    # src[layer_norm.py:298]: acc = x[tile_m, :].to(torch.float32)
    for roffset_1 in tl.range(0, 16384, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        load = tl.load(x + (indices_0[:, None] * 16384 + rindex_1[None, :] * 1), None, eviction_policy="evict_first")
        v_0 = tl.cast(load, tl.float32)
        # src[layer_norm.py:300]: mean_val = torch.sum(acc, dim=-1) / n
        v_1 = sum_1_acc + v_0
        sum_1_acc = v_1
    sum_1 = tl.cast(tl.sum(sum_1_acc, 1), tl.float32)
    v_2 = 6.103515625e-05
    v_3 = sum_1 * v_2
    # src[layer_norm.py:302]: centered = acc - mean_val[:, None]
    subscript = v_3[:, None]
    # src[layer_norm.py:303]: var_val = torch.sum(centered * centered, dim=-1) / n
    sum_2_acc = tl.full([_BLOCK_SIZE_0, _REDUCTION_BLOCK_1], 0, tl.float32)
    # src[layer_norm.py:298]: acc = x[tile_m, :].to(torch.float32)
    for roffset_1 in tl.range(0, 16384, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        subscript_copy = subscript
        load_1 = tl.load(x + (indices_0[:, None] * 16384 + rindex_1[None, :] * 1), None)
        v_4 = tl.cast(load_1, tl.float32)
        # src[layer_norm.py:302]: centered = acc - mean_val[:, None]
        v_5 = v_4 - subscript_copy
        # src[layer_norm.py:303]: var_val = torch.sum(centered * centered, dim=-1) / n
        v_6 = v_5 * v_5
        v_7 = sum_2_acc + v_6
        sum_2_acc = v_7
    sum_2 = tl.cast(tl.sum(sum_2_acc, 1), tl.float32)
    v_8 = 6.103515625e-05
    v_9 = sum_2 * v_8
    # src[layer_norm.py:305]: rstd_val = torch.rsqrt(var_val + eps)
    v_10 = v_9 + eps
    v_11 = libdevice.rsqrt(v_10)
    # src[layer_norm.py:307]: normalized = centered * rstd_val[:, None]
    subscript_1 = v_11[:, None]
    # src[layer_norm.py:316]: mean[tile_m] = mean_val
    tl.store(mean + indices_0 * 1, v_3, None)
    # src[layer_norm.py:317]: rstd[tile_m] = rstd_val
    tl.store(rstd + indices_0 * 1, v_11, None)
    # src[layer_norm.py:298]: acc = x[tile_m, :].to(torch.float32)
    for roffset_1 in tl.range(0, 16384, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        v_3_copy = v_3
        subscript_1_copy = subscript_1
        load_2 = tl.load(x + (indices_0[:, None] * 16384 + rindex_1[None, :] * 1), None)
        v_12 = tl.cast(load_2, tl.float32)
        # src[layer_norm.py:302]: centered = acc - mean_val[:, None]
        subscript_2 = v_3_copy[:, None]
        v_13 = v_12 - subscript_2
        # src[layer_norm.py:307]: normalized = centered * rstd_val[:, None]
        v_14 = v_13 * subscript_1_copy
        # src[layer_norm.py:310]: acc = normalized * (weight[:].to(torch.float32)) + (
        load_3 = tl.load(weight + rindex_1 * 1, None, eviction_policy="evict_first")
        v_15 = tl.cast(load_3, tl.float32)
        v_16 = v_15[None, :]
        v_17 = v_14 * v_16
        # src[layer_norm.py:311]: bias[:].to(torch.float32)
        load_4 = tl.load(bias + rindex_1 * 1, None)
        v_18 = tl.cast(load_4, tl.float32)
        # src[layer_norm.py:310]: acc = normalized * (weight[:].to(torch.float32)) + (
        # src[layer_norm.py:311]:     bias[:].to(torch.float32)
        # src[layer_norm.py:312]: )
        v_19 = v_18[None, :]
        v_20 = v_17 + v_19
        # src[layer_norm.py:315]: out[tile_m, :] = acc.to(x.dtype)
        v_21 = tl.cast(v_20, tl.bfloat16)
        tl.store(out + (indices_0[:, None] * 16384 + rindex_1[None, :] * 1), v_21, None)


def layer_norm_fwd(x, mean, rstd, weight, bias, out, eps, provider:str="helion"):   
    def grid_1d(meta):
        return (x.size(0),)
    m = x.size(0)
    n = x.size(1)
    if provider == "helion":
        _helion_layer_norm_fwd[grid_1d](x, mean, rstd, weight, bias, out, eps)
    else:
        assert provider == "tlx", "type must be either helion or tlx"
        # X, # pointer to the input
        # Y, # pointer to the output
        # W, # pointer to the weights
        # B, # pointer to the biases
        # Mean, # pointer to the mean
        # Rstd, # pointer to the 1/std
        # row_stride, # input row stride
        # M, # number of rows in X
        # N, # number of columns in X
        # eps, # epsilon to avoid division by zero
        # BLOCK_SIZE_N: tl.constexpr
        kernel_norm_multi_cta[grid_1d](x, out, weight, bias, mean, rstd, x.stride(0), m, n, eps, n)

    return (out, mean, rstd)

def normalize_and_create_outs(x_orig, weight, bias, eps):
    """
    Performs 1D layer normalization on the input tensor using Helion.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim], expected to be FP16.
        normalized_shape (list[int]): List containing the dimension to normalize over (should be length 1).
        weight (torch.Tensor): Learnable scale parameter of shape [dim].
        bias (torch.Tensor | None): Optional learnable bias parameter of shape [dim].
        eps (float, optional): Small value added to variance for numerical stability. Default is 1e-5.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - The layer-normalized output tensor of shape [batch_size, dim], in FP16.
            - Mean tensor of shape [batch_size], in FP32.
            - Reciprocal standard deviation tensor of shape [batch_size], in FP32.
    """

    normalized_shape = (x_orig.shape[-1],)
    #x.shape = (B * M, N)
    x = x_orig.reshape(-1, x_orig.shape[-1])

    # src[layer_norm.py:283]: m, n = x.size()
    m, n = x.size()
    # src[layer_norm.py:284]: assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    # src[layer_norm.py:285]: if bias is not None:
    # src[layer_norm.py:286]:     assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
    if bias is not None:
        # src[layer_norm.py:286]: assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
        assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
    # src[layer_norm.py:287]: assert (
    # src[layer_norm.py:288]:     len(normalized_shape) == 1
    # src[layer_norm.py:289]: ), "Helion layer norm only supports 1D layer norm currently"
    assert len(normalized_shape) == 1, "Helion layer norm only supports 1D layer norm currently"
    # src[layer_norm.py:290]: assert (
    # src[layer_norm.py:291]:     normalized_shape[0] == n
    # src[layer_norm.py:292]: ), f"normalized shape mismatch {normalized_shape[0]} != {n}"
    assert normalized_shape[0] == n, f"normalized shape mismatch {normalized_shape[0]} != {n}"
    # src[layer_norm.py:293]: out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    # src[layer_norm.py:294]: mean = torch.empty([m], dtype=torch.float32, device=x.device)
    mean = torch.empty([m], dtype=torch.float32, device=x.device)
    # src[layer_norm.py:295]: rstd = torch.empty([m], dtype=torch.float32, device=x.device)
    rstd = torch.empty([m], dtype=torch.float32, device=x.device)
    # src[layer_norm.py:298]: acc = x[tile_m, :].to(torch.float32)
    # src[layer_norm.py:297]: for tile_m in hl.tile(m):
    # src[layer_norm.py:298]:     acc = x[tile_m, :].to(torch.float32)
    # src[layer_norm.py:299]:     # Compute mean
    # src[layer_norm.py:297-317]: ...
    # _launcher(_helion_layer_norm_fwd, (1152,), x, mean, rstd, weight, bias, out, eps, _BLOCK_SIZE_0, _REDUCTION_BLOCK_1, num_warps=16, num_stages=3)
  
    
    # src[layer_norm.py:318]: return out, mean, rstd
    return (x, mean, rstd, weight, bias, out, eps)
########################## END- HELION #############################

################## TLX #################
@triton.autotune(
    configs=[triton.Config({}, num_warps=16, num_stages=3)],
    key=["x"],
    
)
@triton.jit
def kernel_norm_multi_cta(
    X, # pointer to the input
    Y, # pointer to the output
    W, # pointer to the weights
    B, # pointer to the biases
    Mean_out, # pointer to the mean
    Rstd_out, # pointer to the 1/std
    row_stride, # input row stride
    M, # number of rows in X
    N, # number of columns in X
    eps, # epsilon to avoid division by zero
    BLOCK_SIZE_N: tl.constexpr
):
    row_offsets = tl.program_id(0)
    # Partition reduction axes over multiple CTAs
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # mask_row = row_offsets < M
    mask_col = col_offsets < N
    read_write_offsets = (row_offsets * row_stride) + col_offsets
    read_write_mask = mask_col

    X_ptr = X + read_write_offsets
    Y_ptr = Y + read_write_offsets

    x = tl.load(X_ptr, mask=read_write_mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    x_minus_mean = tl.where(read_write_mask, x - mean, 0.0)
    x_minus_mean_sq = x_minus_mean * x_minus_mean
    var = tl.sum(x_minus_mean_sq, axis=0) / N
    rstd = libdevice.rsqrt(var + eps)
    tl.store(Mean_out + row_offsets, mean)
    tl.store(Rstd_out + row_offsets, rstd)
    w = tl.load(W + col_offsets, mask=mask_col)
    b = tl.load(B + col_offsets, mask=mask_col)
    x_hat = (x - mean) * rstd
    y = x_hat * w + b
    tl.store(Y_ptr, y, mask=read_write_mask)


################### END-TLX ###############
def detailed_tensor_comparison(a, b, rtol=1e-5, atol=1e-8, max_print=20):
    """Comprehensive tensor comparison with detailed output."""

    print("\n" + "=" * 80)
    print("TENSOR COMPARISON")
    print("=" * 80)

    # Shape check
    print(f"\nShapes: {a.shape} vs {b.shape}")
    if a.shape != b.shape:
        print("❌ Shape mismatch!")
        return False

    # Dtype check
    print(f"Dtypes: {a.dtype} vs {b.dtype}")
    if a.dtype != b.dtype:
        print("⚠️  Warning: Different dtypes")

    # Find mismatches
    mismatch_mask = ~torch.isclose(a, b, rtol=rtol, atol=atol)
    num_mismatches = mismatch_mask.sum().item()
    total_elements = a.numel()

    print(f"\nTotal elements: {total_elements}")
    print(f"Matching elements: {total_elements - num_mismatches}")
    print(f"Mismatching elements: {num_mismatches}")
    print(f"Match rate: {100 * (1 - num_mismatches / total_elements):.4f}%")

    if num_mismatches == 0:
        print("\n✅ TENSORS MATCH!")
        return True

    # Difference statistics
    diff = (a - b).abs()
    print(f"\nDifference Statistics:")
    print(f"  Max abs diff: {diff.max().item():.6e}")
    print(f"  Mean abs diff: {diff.mean().item():.6e}")
    print(f"  Std abs diff: {diff.std().item():.6e}")
    print(f"  Median abs diff: {diff.median().item():.6e}")

    # Print mismatches
    print(f"\n{'=' * 80}")
    print(f"First {min(max_print, num_mismatches)} Mismatches:")
    print(f"{'=' * 80}")
    print(f"{'Index':<25} {'Tensor A':<20} {'Tensor B':<20} {'Abs Diff':<15} {'Rel Diff':<15}")
    print("-" * 80)

    mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)
    for idx in mismatch_indices[:max_print]:
        idx_tuple = tuple(idx.tolist())
        val_a = a[idx_tuple].item()
        val_b = b[idx_tuple].item()
        abs_diff = abs(val_a - val_b)
        rel_diff = abs_diff / (abs(val_a) + 1e-10)

        print(f"{str(idx_tuple):<25} {val_a:<20.8e} {val_b:<20.8e} {abs_diff:<15.8e} {rel_diff:<15.8e}")

    if num_mismatches > max_print:
        print(f"\n... and {num_mismatches - max_print} more mismatches")

    print("=" * 80 + "\n")
    return False


def _create_input_tensors(B, M, N, elementwise_affine=True, dtype=DTYPE, device=DEVICE):
    assert B * M == 1152, f"B * M is hardcoded to 1152 in this kernel, got {B * M}"
    assert N == 16384, f"N is hardcoded to 16384 in this kernel, got {N}"
    x_shape = (B, M, N)
    # x.shape = (B, M, N)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    weight, bias = None, None

    if elementwise_affine:
        w_shape = (x_shape[-1],)
        # weight.shape = (N,)
        weight = torch.randn(w_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
        # weight.shape = (N,)
        bias = torch.randn(w_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)

    x.requires_grad_(True)

    return (x, dy, weight, bias)

def check_correctness_elementwise_affine_true(x, dy, weight, bias, eps=1e-5):
    _x = x.detach().clone().requires_grad_(True)
    _weight = weight.detach().clone().requires_grad_(True)
    _bias = bias.detach().clone().requires_grad_(True)

    _x_ref = x.detach().clone().requires_grad_(True)
    _weight_ref = weight.detach().clone().requires_grad_(True)
    _bias_ref = bias.detach().clone().requires_grad_(True)

    w_shape = (x.shape[-1],)
    _x_ref = x.reshape(-1, x.shape[-1])
    y_ref = torch.nn.functional.layer_norm(_x_ref, w_shape, _weight_ref, _bias_ref, eps).to(DTYPE)

    x, mean, rstd, weight, bias, out, eps = normalize_and_create_outs(_x, _weight, _bias, eps)
    for p in ["helion", "tlx"]:
        y_tri, mean, rsts = layer_norm_fwd(x, mean, rstd, weight, bias, out, eps, provider=p)
        if not (torch.allclose(y_ref, y_tri, rtol=1e-5, atol=1e-5)):
            print(f"FAIL! Tensor comparison failed for {p}!")
            detailed_tensor_comparison(y_ref, y_tri, atol=1e-5, rtol=1e-5)
        else:
            print(f"✅ Tensor comparison passed for {p}!")

    # y_ref.backward(dy, retain_graph=True)
    # y_tri.backward(dy, retain_graph=True)

    # torch.testing.assert_close(_x.grad, _x_ref.grad, rtol=1e-4, atol=1e-4)
    # torch.testing.assert_close(_weight.grad, _weight_ref.grad, rtol=1e-4, atol=1e-4)
    # torch.testing.assert_close(_bias.grad, _bias_ref.grad, rtol=1e-4, atol=1e-4)


check_correctness_elementwise_affine_true(*_create_input_tensors(1, 1152, 16384, elementwise_affine=True))

shapes = [(1152, 1, 16384)]
impls = ["helion", "tlx"]
benchmark_configs = [
    triton.testing.Benchmark(
        x_names=["shape"],
        x_vals=[f"{s[0]},{s[1]}, {s[2]}" for s in shapes],
        args={"dtype": torch.float32},
        line_arg="provider",
        # line_vals=["triton-1-cta"],
        # line_names=["triton-1-cta"],
        line_vals=impls,
        line_names=impls,
        plot_name="layer-norm",
    )
]
quantiles = [0.5, 0.2, 0.8]

@triton.testing.perf_report(benchmark_configs)
def benchmark(shape:str, provider, dtype):
    B, M, N = shape.split(",")
    B = int(B)
    M = int(M)
    N = int(N)
    x, dy, weight, bias = _create_input_tensors(B, M, N)
    x, mean, rstd, weight, bias, out, eps = normalize_and_create_outs(x, weight, bias, eps=1e-5)
    x_size = x.numel() * x.element_size()
    w_size = weight.numel() * weight.element_size()
    mem_size = x_size * 2 + w_size

    ms, min_ms, max_ms = triton.testing.do_bench(
            #def layer_norm_fwd(x_orig, weight, bias, eps, provider:str="helion")
            lambda: layer_norm_fwd(x, mean, rstd, weight, bias, out, eps, provider=provider), quantiles=quantiles, rep=1000, warmup=200
        )   
    
    gbps = lambda ms: mem_size * 1e-9 / (ms * 1e-3)
    
    print("latency", shape, provider, ms, max_ms, min_ms)
    print("b/w", shape, provider, gbps(ms), gbps(max_ms), gbps(min_ms))
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(save_path=".", print_data=True)
