"""Numerically gated warm graph benchmark of the public per-head FP8 operator."""
import argparse
import json
import statistics

import torch
import triton
import triton.testing
from triton.tlx.ops import paged_attention_decode_fp8, paged_attention_decode_fp8_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('heads', 16), ('kv-heads', 2), ('dim', 64), ('length', 2304), ('page', 64), ('batch', 1),
                          ('query-length', 1), ('window', 0)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--fp8-q', action='store_true')
    parser.add_argument('--noncausal', action='store_true')
    parser.add_argument('--num-splits', type=int)
    a = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('requires a gfx1250 GPU')
    if (a.dim not in (64, 128) or a.heads < 1 or a.kv_heads < 1 or a.heads % a.kv_heads or a.page < 16 or a.page % 16
            or min(a.length, a.batch, a.query_length) < 1 or a.window < 0 or (a.window and a.noncausal)):
        parser.error('invalid head geometry, page, lengths or window')
    torch.manual_seed(182)
    pages = triton.cdiv(a.length, a.page)
    blocks = a.batch * pages
    ks = torch.linspace(.025, .08, a.kv_heads, device='cuda')
    vs = torch.linspace(.04, .12, a.kv_heads, device='cuda').flip(0).contiguous()
    qs = torch.linspace(.02, .06, a.heads, device='cuda') if a.fp8_q else None
    qf = torch.randn(a.batch * a.query_length, a.heads, a.dim, device='cuda') * .4
    q = (qf / qs[None, :, None]).to(torch.float8_e4m3fn) if a.fp8_q else qf.to(torch.bfloat16)
    k = (torch.randn(blocks, a.kv_heads, a.page, a.dim, device='cuda') * .5 / ks[None, :, None, None]).to(
        torch.float8_e4m3fn)
    v = (torch.randn(blocks, a.kv_heads, a.page, a.dim, device='cuda') * .8 / vs[None, :, None, None]).to(
        torch.float8_e4m3fn)
    kp = k.reshape(blocks, a.kv_heads, a.page, a.dim // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    vp = v.reshape(blocks, a.kv_heads, a.page // 16, 16, a.dim).permute(0, 1, 2, 4, 3).contiguous()
    table = torch.randperm(blocks, device='cuda').reshape(a.batch, pages).to(torch.int32)
    lengths = torch.full((a.batch, ), a.length, device='cuda', dtype=torch.int32)
    plan = dict(max_context_len=a.length, query_length=a.query_length, sliding_window=a.window, num_splits=a.num_splits)
    workspace = paged_attention_decode_fp8_workspace(q, kp, vp, table, **plan)
    out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)

    def call():
        return paged_attention_decode_fp8(q, kp, vp, lengths, table, ks, vs, q_scale=qs, causal=not a.noncausal,
                                          out=out, workspace=workspace, **plan)

    call()
    expected = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    qdeq = q.float() if qs is None else q.float() * qs[None, :, None]
    group = a.heads // a.kv_heads
    positions = torch.arange(a.length, device=q.device)
    for seq in range(a.batch):
        ids = table[seq].long()
        for kvh in range(a.kv_heads):
            kd = k[ids, kvh].float().reshape(-1, a.dim)[:a.length] * ks[kvh]
            vd = v[ids, kvh].float().reshape(-1, a.dim)[:a.length] * vs[kvh]
            for first in range(0, a.query_length, 128):
                last = min(first + 128, a.query_length)
                rows = slice(seq * a.query_length + first, seq * a.query_length + last)
                heads = slice(kvh * group, (kvh + 1) * group)
                qr = qdeq[rows, heads].transpose(0, 1)
                scores = (qr @ kd.T) * a.dim**-.5
                qpos = a.length - a.query_length + torch.arange(first, last, device=q.device)
                mask = torch.ones(last - first, a.length, device=q.device, dtype=torch.bool)
                if not a.noncausal:
                    mask &= positions[None, :] <= qpos[:, None]
                if a.window:
                    mask &= positions[None, :] > qpos[:, None] - a.window
                p = torch.softmax(scores.masked_fill(~mask[None], -float('inf')), -1).nan_to_num()
                expected[rows, heads] = (p @ vd).transpose(0, 1)
    torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-2)
    samples = [triton.testing.do_bench_cudagraph(call, rep=100) * 1000 for _ in range(3)]
    us = statistics.median(samples)
    useful_bytes = 2 * a.batch * min(a.length, a.window or a.length) * a.kv_heads * a.dim
    print(
        json.dumps(
            dict(op='paged_attention_decode_fp8', workload=vars(a), correctness='pass', warm_graph_us=us,
                 rounds_us=samples, useful_kv_tb_per_s=useful_bytes / (us * 1e6)), indent=2))


if __name__ == '__main__':
    main()
