import importlib.util
import math
import sys

import torch
import triton

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("gen", path)
gen = importlib.util.module_from_spec(spec)
sys.modules["gen"] = gen
spec.loader.exec_module(gen)

triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device="cuda"))
D = 128
shapes = [(1, 2, 512)] if "--small" in sys.argv else [(1, 2, 512), (1, 32, 8192)]
for (Z, H, N_CTX) in shapes:
    torch.manual_seed(0)
    q = torch.randn(Z, H, N_CTX, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Z, H, N_CTX, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Z, H, N_CTX, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    m_lse = torch.full((Z * H * N_CTX,), float("nan"), device="cuda", dtype=torch.float32)
    sm = 1.0 / (D ** 0.5)
    grid = (triton.cdiv(N_CTX, 256), Z * H)
    call = lambda: gen.fa_fwd_kernel_nows_subtiled[grid](
        q.view(-1, D), k.view(-1, D), v.view(-1, D), out.view(-1, D), m_lse,
        sm, Z * H, N_CTX, num_warps=4)
    call()
    torch.cuda.synchronize()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    ok = rel < 1e-2 and not torch.isnan(out).any()
    msg = f"({Z},{H},{N_CTX}) out rel={rel:.2e} {'PASS' if ok else 'FAIL'}"
    if N_CTX <= 1024:
        qk = (q.float() @ k.float().transpose(-1, -2)) * sm
        ref_m = torch.logsumexp(qk, dim=-1) / math.log(2)
        got_m = m_lse.view(Z, H, N_CTX)
        mrel = (got_m - ref_m).abs().max().item() / ref_m.abs().max().item()
        mok = mrel < 1e-2 and not torch.isnan(got_m).any()
        msg += f" | m_lse rel={mrel:.2e} {'PASS' if mok else 'FAIL'}"
    print(msg)
    if "--bench" in sys.argv and N_CTX >= 8192:
        ms = triton.testing.do_bench(call, warmup=50, rep=200, quantiles=[0.5])
        tf = 4 * Z * H * N_CTX * N_CTX * D / (ms * 1e-3) / 1e12
        print(f"({Z},{H},{N_CTX}): {ms:.3f} ms = {tf:.1f} TFLOPS")
