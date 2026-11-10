"""Run pinned-source comparisons sequentially on the selected GPU.

Run inside the documented experiment container with its TLX runtime. Every
case has explicit head/mask/scale assumptions; hidden model metadata is not
inferred. Outputs are incremental, so an interrupted run remains inspectable.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent


def cases():
    all_bf16 = ['tlx_auto', 'tlx_baseline', 'aiter_unified', 'vllm_unified', 'sglang', 'vllm_decode']
    full_fp8 = ['tlx_auto', 'aiter_unified', 'aiter_dense', 'tokenspeed', 'vllm_unified', 'sglang']
    result = []

    def add(stage, name, backends, *args):
        result.append(dict(stage=stage, name=name, backends=backends, args=list(args)))

    for name, shape in [('gqa', []), ('mha', ['--heads', '128', '--kv-heads', '128'])]:
        add('controls', name + '_shared', all_bf16, *shape, '--scales', 'shared')
        add('native', name + '_perhead', ['tlx_auto', 'tlx_baseline', 'vllm_unified'], *shape)
        add('fp8', name + '_fp8_unit', full_fp8, *shape, '--fp8-q', '--scales', 'unit')
        add('native', name + '_fp8_perhead', ['tlx_auto', 'tlx_baseline'], *shape, '--fp8-q')
    for t in [8, 32, 128, 512, 2048, 8192]:
        add('queries', f'folded_shared_t{t}', ['tlx_auto', 'vllm_unified'], '--tokens', str(t))
    for t in [8, 32, 128, 512, 2048, 8192]:
        add('dense_queries', f'dense_shared_t{t}', ['tlx_auto', 'vllm_unified'], '--tokens', str(t), '--heads', '128',
            '--kv-heads', '128')
    for length in [1024, 8192, 32768, 131072, 262144]:
        add('dense_controls', f'dense_shared_l{length}', ['tlx_auto', 'aiter_unified'], '--heads', '128', '--kv-heads',
            '128', '--length', str(length), '--scales', 'shared')
    for length in [1024, 8192, 32768, 131072, 262144]:
        add('lengths', f'dense_l{length}', ['tlx_auto', 'tlx_baseline', 'vllm_unified'], '--heads', '128', '--kv-heads',
            '128', '--length', str(length), '--splits', '32')
    for length in [8192, 32768, 131072, 262144]:
        add('long_controls', f'gqa_shared_l{length}', all_bf16, '--length', str(length), '--scales', 'shared',
            '--splits', '32')
    for window in [37, 128]:
        add('windows', f'gqa_swa{window}_perhead', ['tlx_auto', 'tlx_baseline', 'vllm_unified'], '--window',
            str(window))
        add('windows', f'gqa_swa{window}_shared', ['tlx_auto', 'aiter_unified'], '--window', str(window), '--scales',
            'shared')
    add('layout', 'd128_dense_perhead', ['tlx_auto', 'tlx_baseline', 'vllm_unified'], '--heads', '64', '--kv-heads',
        '64', '--dim', '128')
    add('layout', 'page128_gqa_shared', ['tlx_auto', 'aiter_unified', 'vllm_unified'], '--page', '128', '--scales',
        'shared')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--stages', nargs='+', default=[
            'controls', 'native', 'fp8', 'queries', 'dense_queries', 'lengths', 'long_controls', 'dense_controls',
            'windows', 'layout'
        ])
    p.add_argument('--samples', type=int, default=15)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    specs = [c for c in cases() if c['stage'] in a.stages]
    (a.output / 'suite_spec.json').write_text(json.dumps(specs, indent=2) + '\n')
    statuses = []
    for spec in specs:
        path = a.output / (spec['name'] + '.json')
        cmd = [
            sys.executable,
            str(BASE / 'bench_decode.py'), '--backends', *spec['backends'], '--samples',
            str(a.samples), '--repeats',
            str(a.repeats), '--output',
            str(path), *spec['args']
        ]
        with (a.output / (spec['name'] + '.log')).open('w') as log:
            completed = subprocess.run(cmd, stdout=log, stderr=log)
        rows = json.loads(path.read_text()) if path.exists() else []
        status = dict(
            name=spec['name'], exit_code=completed.returncode,
            backends=[{k: r[k]
                       for k in ('backend', 'status', 'median_us', 'selected_config')
                       if k in r}
                      for r in rows])
        statuses.append(status)
        (a.output / 'suite_status.json').write_text(json.dumps(statuses, indent=2) + '\n')
        print(json.dumps(status), flush=True)
    if any(s['exit_code'] for s in statuses):
        raise SystemExit('A benchmark process failed; inspect per-case logs')


if __name__ == '__main__':
    main()
