"""Warm graph benchmark of the public per-head FP8 paged-attention operator.

Run as ``python -m triton.tlx.ops.kernels.paged_attn_fp8.bench``. For example::

    python -m triton.tlx.ops.kernels.paged_attn_fp8.bench --heads 16 --kv-heads 2 --dim 64 --length 2304 --page-size 64

``--page-size`` counts cached tokens per physical block, not bytes; ``--page``
is an alias. To expose scheduling overhead with mostly short active requests::

    python -m triton.tlx.ops.kernels.paged_attn_fp8.bench --batch 32 --length 128 --length-profile ragged-short --max-context-len 262144 --compare-bounds --split-sweep 1 4 8 16 32

``--length`` is the largest active length; ``--max-context-len`` independently
sets the graph/scheduling bound and block-table capacity. ``--compare-bounds``
compares tight and configured bounds on identical Q/K/V and block tables.
``--split-sweep`` adds explicit split controls alongside automatic selection.
``one-long`` keeps all but the last request short; ``--context-lengths`` accepts
an exact comma-separated vector instead of a generated profile. Automatic batched decode can use fewer splits per request from device lengths;
reported selected splits describe workspace/grid capacity. Empty workgroup counts
use that runtime policy. Timings remain warm-cache graph measurements, including
inactive partition dispatch and any merge.
Input generation, FP8 quantization, cache packing and the independent
CPU float64 correctness gate run outside timing. The measured call includes the public
operator's attention and split merge. Useful KV TB/s counts unique active K/V
bytes per call; warm-cache throughput is not measured physical HBM bandwidth.
"""
import argparse
import json
import statistics

import torch
import triton
import triton.testing
from triton.tlx.ops import paged_attention_decode_fp8, paged_attention_decode_fp8_workspace
from triton.tlx.ops.kernels.paged_attn_fp8.gfx1250_decode import _select_decode_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('heads', 16), ('kv-heads', 2), ('dim', 64), ('length', 2304), ('batch', 1),
                          ('query-length', 1), ('window', 0)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--page-size', '--page', dest='page_size', type=int, default=64,
                        help='cached tokens per physical KV block (a positive multiple of 16)')
    parser.add_argument('--fp8-q', action='store_true')
    parser.add_argument('--noncausal', action='store_true')
    parser.add_argument('--num-splits', type=int)
    parser.add_argument('--max-context-len', type=int, help='context bound, independent of active lengths')
    parser.add_argument('--length-profile', choices=['uniform', 'ragged-short', 'one-long'], default='uniform')
    parser.add_argument('--context-lengths', help='explicit nonnegative lengths, one per batch request')
    parser.add_argument('--compare-bounds', action='store_true', help='also measure the tight active-length bound')
    parser.add_argument('--split-sweep', type=int, nargs='+', default=[], help='additional explicit split counts')
    a = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('requires a gfx1250 GPU')
    if (a.dim not in (64, 128) or a.heads < 1 or a.kv_heads < 1 or a.heads % a.kv_heads or a.page_size < 16
            or a.page_size % 16 or min(a.length, a.batch, a.query_length) < 1 or a.window < 0
            or (a.window and a.noncausal)):
        parser.error('invalid head geometry, page, lengths or window')
    if a.context_lengths is not None:
        try:
            active_lengths = [int(n) for n in a.context_lengths.split(',')]
        except ValueError:
            parser.error('--context-lengths must contain comma-separated integers')
        if len(active_lengths) != a.batch or min(active_lengths) < 0:
            parser.error('--context-lengths must contain one nonnegative length per batch request')
    elif a.length_profile == 'uniform':
        active_lengths = [a.length] * a.batch
    else:
        short = a.length if a.length_profile == 'ragged-short' else min(128, a.length)
        pattern = [0, 1, a.page_size - 1, a.page_size, a.page_size + 1, short]
        active_lengths = [min(short, pattern[i % len(pattern)]) for i in range(a.batch)]
        active_lengths[-1] = a.length
    tight_bound = max(1, max(active_lengths))
    bound = a.max_context_len if a.max_context_len is not None else tight_bound
    if bound < tight_bound or min([a.num_splits if a.num_splits is not None else 1] + a.split_sweep) < 1:
        parser.error('context bound must cover active lengths and split counts must be positive')
    torch.manual_seed(182)
    pages = triton.cdiv(bound, a.page_size)
    page_counts = [triton.cdiv(n, a.page_size) for n in active_lengths]
    blocks = max(1, sum(page_counts))
    ks = torch.linspace(.025, .08, a.kv_heads, device='cuda')
    vs = torch.linspace(.04, .12, a.kv_heads, device='cuda').flip(0).contiguous()
    qs = torch.linspace(.02, .06, a.heads, device='cuda') if a.fp8_q else None
    qf = torch.randn(a.batch * a.query_length, a.heads, a.dim, device='cuda') * .4
    q = (qf / qs[None, :, None]).to(torch.float8_e4m3fn) if a.fp8_q else qf.to(torch.bfloat16)
    k = (torch.randn(blocks, a.kv_heads, a.page_size, a.dim, device='cuda') * .5 / ks[None, :, None, None]).to(
        torch.float8_e4m3fn)
    v = (torch.randn(blocks, a.kv_heads, a.page_size, a.dim, device='cuda') * .8 / vs[None, :, None, None]).to(
        torch.float8_e4m3fn)
    kp = k.reshape(blocks, a.kv_heads, a.page_size, a.dim // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    vp = v.reshape(blocks, a.kv_heads, a.page_size // 16, 16, a.dim).permute(0, 1, 2, 4, 3).contiguous()
    table = torch.full((a.batch, pages), -1, device='cuda', dtype=torch.int32)
    shuffled = torch.randperm(blocks, device='cuda').to(torch.int32)
    offset = 0
    for seq, count in enumerate(page_counts):
        table[seq, :count] = shuffled[offset:offset + count]
        offset += count
    lengths = torch.tensor(active_lengths, device='cuda', dtype=torch.int32)
    expected = torch.zeros(q.shape, dtype=torch.float64)
    qdeq = q.detach().cpu().double()
    if qs is not None:
        qdeq *= qs.detach().cpu().double()[None, :, None]
    kr, vr = k.detach().cpu().double(), v.detach().cpu().double()
    ksr, vsr = ks.detach().cpu().double(), vs.detach().cpu().double()
    page_table = table.cpu().long()
    group = a.heads // a.kv_heads
    for seq, length in enumerate(active_lengths):
        if not length:
            continue
        ids = page_table[seq, :page_counts[seq]]
        positions = torch.arange(length)
        for kvh in range(a.kv_heads):
            kd = kr[ids, kvh].reshape(-1, a.dim)[:length] * ksr[kvh]
            vd = vr[ids, kvh].reshape(-1, a.dim)[:length] * vsr[kvh]
            for first in range(0, a.query_length, 128):
                last = min(first + 128, a.query_length)
                rows = slice(seq * a.query_length + first, seq * a.query_length + last)
                heads = slice(kvh * group, (kvh + 1) * group)
                qr = qdeq[rows, heads].transpose(0, 1)
                scores = (qr @ kd.T) * a.dim**-.5
                qpos = length - a.query_length + torch.arange(first, last)
                mask = torch.ones(last - first, length, dtype=torch.bool)
                if not a.noncausal:
                    mask &= positions[None, :] <= qpos[:, None]
                if a.window:
                    mask &= positions[None, :] > qpos[:, None] - a.window
                p = torch.softmax(scores.masked_fill(~mask[None], -float('inf')), -1).nan_to_num()
                expected[rows, heads] = (p @ vd).transpose(0, 1)
    expected = expected.to(device=q.device, dtype=torch.float32)
    bounds = list(dict.fromkeys([tight_bound, bound] if a.compare_bounds else [bound]))
    split_counts = list(dict.fromkeys([a.num_splits] + a.split_sweep))
    measurements = []
    for context_bound in bounds:
        for splits in split_counts:
            plan = dict(max_context_len=context_bound, query_length=a.query_length, sliding_window=a.window,
                        num_splits=splits)
            workspace = paged_attention_decode_fp8_workspace(q, kp, vp, table, **plan)
            out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
            config = _select_decode_config(q, kp, max_context_len=context_bound, query_length=a.query_length,
                                           window_size=a.window, num_splits=splits)

            def call(plan=plan, workspace=workspace, out=out):
                return paged_attention_decode_fp8(q, kp, vp, lengths, table, ks, vs, q_scale=qs, causal=not a.noncausal,
                                                  out=out, workspace=workspace, **plan)

            call()
            torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-2)
            diagnostics = {}
            if a.query_length == 1:
                # Analytical launch/partition counts for the selected config,
                # not profiler counters. Runtime partitions use actual lengths.
                count, bn = config['num_splits'], config['block_n']
                chunk = config.get('split_chunk', 0)
                tasks_per_seq = a.heads if config['variant'].startswith('vector') else (
                    a.kv_heads * triton.cdiv(group, config['block_m']))
                nonempty = 0
                work_items = 0
                sequence_work = []
                for n in active_lengths:
                    begin = max(0, n - a.window) if a.window else 0
                    if config['variant'] in ('matrix_tdm_flat', 'vector_page'):
                        begin = begin // a.page_size * a.page_size
                    active = max(0, n - begin)
                    if config.get('split_target'):
                        group_size = config['sequence_group']
                        first_seq = len(sequence_work) // group_size * group_size
                        live = sum(length > 512 for length in active_lengths[first_seq:first_seq + group_size])
                        budget = max(1, config['split_target'] // max(1, live * tasks_per_seq))
                        active_count = 1 if n <= 512 else min(count, triton.cdiv(n, chunk), budget)
                    else:
                        active_count = (1 if n <= 512 else min(count, triton.cdiv(n, chunk))) if chunk else count
                    work_items += active_count * tasks_per_seq
                    sequence_work.append(active_count * tasks_per_seq)
                    span = triton.cdiv(active, active_count * bn) * bn
                    nonempty += (triton.cdiv(active, span) if span else 0) * tasks_per_seq
                if config.get('partition_grid'):
                    # A bounded block can execute several packed work items;
                    # work-item counts and physical workgroups differ here.
                    group_size = config.get('sequence_group', a.batch)
                    group_grid = config['partition_grid']
                    grid = group_grid * triton.cdiv(a.batch, group_size)
                    idle = sum(
                        max(0, group_grid - sum(sequence_work[i:i + group_size]))
                        for i in range(0, a.batch, group_size))
                    diagnostics = dict(partition_ctas=grid, partition_work_items=work_items,
                                       nonempty_partition_work_items=nonempty, idle_partition_ctas=idle)
                else:
                    total = a.batch * tasks_per_seq * count
                    diagnostics = dict(partition_ctas=total, nonempty_partition_ctas=nonempty,
                                       empty_partition_ctas=total - nonempty)
            if config.get('split_chunk'):
                diagnostics['active_splits_per_sequence'] = [n // tasks_per_seq for n in sequence_work]
            measurements.append(
                dict(
                    call=call, max_context_len=context_bound, requested_splits=splits, selected_config=config,
                    correctness='pass', rounds_us=[],
                    workspace_bytes=sum(t.numel() * t.element_size()
                                        for t in workspace) if workspace is not None else 0, **diagnostics))
    # Compile/capture before sampling, then rotate order to reduce drift bias.
    for item in measurements:
        triton.testing.do_bench_cudagraph(item['call'], rep=100)
    for round_id in range(3):
        order = measurements[round_id % len(measurements):] + measurements[:round_id % len(measurements)]
        for item in order:
            item['rounds_us'].append(triton.testing.do_bench_cudagraph(item['call'], rep=100) * 1000)
    useful_bytes = 2 * sum(min(n, a.window or n) for n in active_lengths) * a.kv_heads * a.dim
    for item in measurements:
        del item['call']
        item['warm_graph_us'] = statistics.median(item['rounds_us'])
        item['useful_kv_tb_per_s'] = useful_bytes / (item['warm_graph_us'] * 1e6)
    result = dict(op='paged_attention_decode_fp8', workload=vars(a), active_lengths=active_lengths,
                  useful_kv_bytes=useful_bytes, allocated_kv_bytes=kp.numel() + vp.numel(),
                  block_table_shape=list(table.shape), results=measurements)
    # Preserve the original single-case JSON fields for benchmark consumers.
    if len(measurements) == 1:
        result.update(measurements[0])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
