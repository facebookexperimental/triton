"""Run the `triton.tlx.ops` ops implemented for the current GPU.

A small, self-contained workload for profilers (rocprofv3/ATT, Proton, ...):
each op runs a few times on a representative shape and is checked against a
torch reference where one is cheap, otherwise for finite output. With no op
names it runs every op the catalog implements for the current arch. AMD only
for now.

    python -m triton.tlx.ops.profiling.driver [--list] [OP ...]
"""

import argparse
import itertools
import sys
import traceback

import torch
import torch.nn.functional as F
from triton.language.extra.tlx.hw.target import current_target
from triton.tlx import ops
from triton.tlx.ops._catalog import has_impl
from triton.tlx.ops.kernels.flash_attn_varlen._shapes import inputs as varlen_inputs
from triton.tlx.ops.kernels.hstu_attn._shapes import inputs as hstu_inputs
from triton.tlx.ops.kernels.moe._shapes import MoeShape
from triton.tlx.ops.kernels.moe._shapes import inputs as moe_inputs
from triton.tlx.ops.kernels.paged_attn import gfx950_decode, gfx950_prefill

DEVICE = "cuda"


def _randn(*shape, dtype=torch.bfloat16, scale=1.0):
    return torch.randn(shape, device=DEVICE, dtype=torch.float32).mul_(scale).to(dtype)


def _offsets(lengths, dtype=torch.int32):
    return torch.tensor([0, *itertools.accumulate(lengths)], device=DEVICE, dtype=dtype)


def _e4m3(*shape):
    return torch.randint(-2, 3, shape, device=DEVICE).to(torch.float8_e4m3fn)


def _e8m0(*shape):
    return torch.randint(125, 128, shape, device=DEVICE, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def _dequant_mxfp8(data, scale):
    return data.float() * torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(32, dim=-1)


def _attention(q, k, v, causal, scale):
    """fp32 reference over (..., S, D); causal is bottom-right aligned."""
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        sq, sk = scores.shape[-2:]
        mask = torch.ones(sq, sk, device=q.device, dtype=torch.bool).tril(sk - sq)
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v.float(), torch.logsumexp(scores, dim=-1)


def _packed_attention(q, k, v, q_lengths, kv_lengths, causal, scale):
    """Reference for packed (tokens, heads, D) sequences: output and (heads, tokens) LSE."""
    outs, lses = [], []
    for q_seq, k_seq, v_seq in zip(q.split(q_lengths), k.split(kv_lengths), v.split(kv_lengths)):
        out, lse = _attention(q_seq.transpose(0, 1), k_seq.transpose(0, 1), v_seq.transpose(0, 1), causal, scale)
        outs.append(out.transpose(0, 1))
        lses.append(lse)
    return torch.cat(outs), torch.cat(lses, dim=-1)


# Each case returns (run, reference): `run()` calls the op and returns its
# output; `reference()` returns the expected output, or is None when the op
# has no cheap reference and only finite output is checked.


def case_mm(space):
    a, b = _randn(4096, 4096), _randn(4096, 4096)
    return lambda: ops.mm(a, b, space=space), lambda: a.float() @ b.float()


def case_addmm(space):
    bias, a, b = _randn(4096), _randn(4096, 4096), _randn(4096, 4096)
    return lambda: ops.addmm(bias, a, b, space=space), lambda: bias.float() + a.float() @ b.float()


def case_mm_mxfp(space):
    m = n = k = 4096
    a, b, a_scale, b_scale = _e4m3(m, k), _e4m3(n, k), _e8m0(m, k // 32), _e8m0(n, k // 32)
    return (lambda: ops.mm_mxfp(a, a_scale, b, b_scale, space=space),
            lambda: _dequant_mxfp8(a, a_scale) @ _dequant_mxfp8(b, b_scale).T)


def case_grouped_gemm(space):
    shapes = ((4096, 4096, 4096), (2048, 4096, 4096), (1000, 4096, 4096), (333, 4096, 4096))
    group_a = [_randn(m, k, dtype=torch.float16) for m, _, k in shapes]
    group_b = [_randn(n, k, dtype=torch.float16).T for _, n, k in shapes]
    return (lambda: ops.grouped_gemm(group_a, group_b, space=space),
            lambda: [a.float() @ b.float() for a, b in zip(group_a, group_b)])


def case_grouped_gemm_mxfp8(space):
    split_sizes, n, k = (1024, 512, 128, 2048), 2048, 2048
    gm, groups = sum(split_sizes), len(split_sizes)
    x, x_scale = _e4m3(gm, k), _e8m0(gm, k // 32)
    w, w_scale = _e4m3(groups, n, k), _e8m0(groups, n, k // 32)
    splits = torch.tensor(split_sizes, device=DEVICE, dtype=torch.int32)

    def reference():
        rows = _dequant_mxfp8(x, x_scale).split(split_sizes)
        return torch.cat([r @ _dequant_mxfp8(w[g], w_scale[g]).T for g, r in enumerate(rows)])

    return lambda: ops.grouped_gemm_mxfp8(x, x_scale, w, w_scale, splits, space=space), reference


def case_moe(space):
    shape = MoeShape(tokens=4096, K=1024, I=1024, H=1024, local_experts=8, num_experts=32, topk=4, expert_start=8,
                     dtype="bf16")
    x, weights, ids, w13, w2, _ = moe_inputs(shape)
    return lambda: ops.moe(x, weights, ids, w13, w2, expert_start=shape.expert_start, space=space), None


def _bhsd(dtype=torch.bfloat16):
    return [_randn(1, 16, 4096, 128, dtype=dtype, scale=0.5).clamp_(-1, 1) for _ in range(3)]


def case_flash_attn(space):
    q, k, v = _bhsd()
    return lambda: ops.flash_attn(q, k, v, space=space), lambda: _attention(q, k, v, False, 128**-0.5)[0]


def case_flash_attn_mxfp8(space):
    q, k, v = _bhsd()
    kwargs = {} if space is None else {"space": space}
    return lambda: ops.flash_attn_mxfp8(q, k, v, **kwargs), lambda: _attention(q, k, v, False, 128**-0.5)[0]


def case_flash_attn_varlen(space):
    lengths = [512, 1024, 2048, 777]
    q, k, v, cu = varlen_inputs(lengths, 16, 128, torch.bfloat16)
    return (lambda: ops.flash_attn_varlen(q, k, v, cu, cu, max(lengths), max(lengths), space=space),
            lambda: _packed_attention(q, k, v, lengths, lengths, False, 128**-0.5)[0])


def case_flash_attn_varlen_backward(space):
    lengths, scale = [128, 512, 1000], 128**-0.5
    q, k, v, cu = varlen_inputs(lengths, 8, 128, torch.bfloat16)
    out, lse = _packed_attention(q, k, v, lengths, lengths, False, scale)
    out, do = out.to(torch.bfloat16), torch.randn_like(q)
    plan = ops.prepare_flash_attn_varlen_backward(cu, cu)

    def reference():
        leaves = [t.float().requires_grad_() for t in (q, k, v)]
        _packed_attention(*leaves, lengths, lengths, False, scale)[0].backward(do.float())
        return [t.grad for t in leaves]

    return lambda: ops.flash_attn_varlen_backward(q, k, v, out, do, lse.contiguous(), plan, scale), reference


def case_hstu_attn_dev(space):
    max_seq_len = 1024
    q, k, v, offsets, attn_scale = hstu_inputs(4, max_seq_len, 4, 128, torch.bfloat16)
    kwargs = {} if space is None else {"space": space}
    return lambda: ops.hstu_attn_dev(q, k, v, offsets, max_seq_len, attn_scale, **kwargs), None


def case_hstu_cross_attn(space):
    lengths_kv, lengths_q, heads, dim = [1024, 512, 2048], [64, 128, 32], 2, 128
    seq_offsets, seq_offsets_q = _offsets(lengths_kv, torch.int64), _offsets(lengths_q, torch.int64)
    q = _randn(sum(lengths_q), heads, dim, scale=0.05)
    k, v = (_randn(sum(lengths_kv), heads, dim, scale=0.05) for _ in range(2))
    attn_scale = torch.tensor(1.0 / (max(lengths_kv) + max(lengths_q)), device=DEVICE)
    kwargs = {} if space is None else {"space": space}

    def run():
        return ops.hstu_cross_attn(q, k, v, seq_offsets, max(lengths_kv), attn_scale, alpha=dim**-0.5,
                                   seq_offsets_q=seq_offsets_q, max_q_len=max(lengths_q), num_softmax_heads=heads,
                                   **kwargs)

    return run, None


def _kda_inputs(tokens, heads, dim):
    shape = (1, tokens, heads, dim)
    q, k = (F.normalize(_randn(*shape, dtype=torch.float32), dim=-1).to(torch.bfloat16) for _ in range(2))
    v = _randn(*shape)
    g = -F.softplus(_randn(*shape, dtype=torch.float32))
    beta = torch.sigmoid(_randn(1, tokens, heads, dtype=torch.float32))
    return q, k, v, g, beta


def case_kda_paged_prefill(space):
    lengths, heads = [512, 1024, 333], 8
    q, k, v, g, beta = _kda_inputs(sum(lengths), heads, 128)
    state = 0.1 * _randn(len(lengths), heads, 128, 128, dtype=torch.float32)
    cu = _offsets(lengths, torch.int64)
    return lambda: ops.kda_paged_prefill(q, k, v, g, beta, initial_state=state, cu_seqlens=cu, space=space), None


def case_kda_recurrent_decode(space):
    batch, heads = 64, 8
    q, k, v, g, beta = _kda_inputs(batch, heads, 128)
    pool = _randn(2 * batch, heads, 128, 128, dtype=torch.float32)
    read = torch.arange(batch, device=DEVICE, dtype=torch.int32)
    cu = torch.arange(batch + 1, device=DEVICE, dtype=torch.int64)
    write = read + batch

    def run():
        return ops.kda_recurrent_decode(q, k, v, g, beta, state_pool=pool, read_indices=read, write_indices=write,
                                        cu_seqlens=cu, space=space)

    return run, None


def case_paged_attention_decode(space):
    ctx_lens, kv_heads, q_heads, head_dim = [3, 40, 300, 2100, 4133, 8000, 1024, 512], 4, 32, 128
    inputs = gfx950_decode.build_inputs(len(ctx_lens), ctx_lens, q_heads, kv_heads, head_dim, 16, device=DEVICE)
    return (lambda: ops.paged_attention_decode(*inputs, space=space),
            lambda: gfx950_decode.ref_decode(*inputs, head_dim**-0.5, q_heads, kv_heads, 1))


def case_paged_attention_prefill(space):
    q_lens, ctx_lens, q_heads, kv_heads, head_dim = [128, 256, 77], [1024, 2048, 300], 16, 2, 64
    query, key_cache, value_cache, cu, ctx, block_tables = gfx950_prefill.build_inputs(
        q_lens, ctx_lens, q_heads, kv_heads, head_dim, 16, device=DEVICE)
    scale = head_dim**-0.5

    def run():
        return ops.paged_attention_prefill(query, key_cache, value_cache, torch.empty_like(query), cu, max(q_lens), ctx,
                                           block_tables.shape[1] * key_cache.shape[1], scale, True, (-1, -1),
                                           block_tables, 0.0, None, None, None, space=space)

    return run, lambda: gfx950_prefill.ref_prefill(query, key_cache, value_cache, cu, ctx, block_tables, scale)


CASES = {
    "mm": case_mm,
    "addmm": case_addmm,
    "mm_mxfp": case_mm_mxfp,
    "grouped_gemm": case_grouped_gemm,
    "grouped_gemm_mxfp8": case_grouped_gemm_mxfp8,
    "moe": case_moe,
    "flash_attn": case_flash_attn,
    "flash_attn_mxfp8": case_flash_attn_mxfp8,
    "flash_attn_varlen": case_flash_attn_varlen,
    "flash_attn_varlen_backward": case_flash_attn_varlen_backward,
    "hstu_attn_dev": case_hstu_attn_dev,
    "hstu_cross_attn": case_hstu_cross_attn,
    "kda_paged_prefill": case_kda_paged_prefill,
    "kda_recurrent_decode": case_kda_recurrent_decode,
    "paged_attention_decode": case_paged_attention_decode,
    "paged_attention_prefill": case_paged_attention_prefill,
}


def _flatten(value):
    if isinstance(value, torch.Tensor):
        return [value]
    return [t for item in value for t in _flatten(item)]


def _check(out, reference):
    outs = _flatten(out)
    if not all(torch.isfinite(t.float()).all() for t in outs):
        return "non-finite output"
    if reference is None:
        return "finite (no reference)"
    refs = _flatten(reference())
    err = max((o.float() - r.float()).abs().max().item() for o, r in zip(outs, refs, strict=True))
    return f"max_abs_err={err:.4g}"


def _run_case(name, args):
    run, reference = CASES[name](args.space)
    with torch.no_grad():
        out = run()
        for _ in range(args.iters - 1):
            out = run()
        torch.cuda.synchronize()
    return _check(out, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ops", nargs="*", metavar="OP",
                        help=f"ops to run (default: all supported); one of {list(CASES)}")
    parser.add_argument("--list", action="store_true", help="list the ops supported on this GPU and exit")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--space", default=None, help="search space passed to each op; defaults to each op's default")
    args = parser.parse_args()
    unknown = sorted(set(args.ops) - set(CASES))
    if unknown:
        parser.error(f"unknown ops {unknown}; choose from {list(CASES)}")

    arch = current_target().key
    supported = [name for name in CASES if has_impl(name, arch)]
    if args.list:
        print("\n".join(supported))
        return
    torch.manual_seed(0)
    failed = []
    for name in args.ops or supported:
        if name not in supported:
            print(f"{name}: skipped, no {arch} implementation", flush=True)
            continue
        try:
            print(f"{name}: {_run_case(name, args)}", flush=True)
        except Exception:
            print(f"{name}: FAILED\n{traceback.format_exc()}", flush=True)
            failed.append(name)
    if failed:
        sys.exit(f"failed on {arch}: {', '.join(failed)}")


if __name__ == "__main__":
    main()
