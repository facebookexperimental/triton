import json
import os
from pathlib import Path
import re
import subprocess

from meta_llvm_assets import (BRANCHES, BUILD_NUMBER, FORK, bundle_complete, github_json, published_assets,
                              release_assets, required_platforms)

MANIFESTS = {'core': 'cmake/meta-llvm-build-info.json', 'amd': 'cmake/meta-amd-llvm-build-info.json'}


def build_requests(paths, root=Path('.')):
    requests = []
    for kind, path in MANIFESTS.items():
        if path not in paths:
            continue
        info = json.loads((root / path).read_text())
        if info['repository'] != FORK:
            raise ValueError(f'{path}: expected the Meta LLVM fork')
        if re.fullmatch(r'[0-9a-f]{40}', info['llvm_hash']) is None:
            raise ValueError(f'{path}: expected a full immutable LLVM SHA')
        if type(info['build_number']) is not int or info['build_number'] < 0:
            raise ValueError(f'{path}: expected a non-negative integer build number')
        requests.append({'kind': kind, 'llvm_sha': info['llvm_hash'], 'build_number': str(info['build_number'])})
    return requests


def pending_builds(repo, candidates, raw_assets, platforms, validate_existing=False):
    from meta_llvm_ready import ready_receipt

    assets = published_assets(raw_assets)
    requests = []
    for candidate in candidates:
        kind, sha, tree, build = candidate['kind'], candidate['llvm_sha'], candidate['llvm_tree'], int(
            candidate['build_number'])
        if re.fullmatch(r'[0-9a-f]{40}', sha) is None:
            raise ValueError(f'{kind}: expected a full immutable LLVM SHA')
        if (validate_existing or not bundle_complete(kind, sha, tree, assets, platforms, build)
                or ready_receipt(repo, kind, sha, tree, build, assets, platforms) is None):
            requests.append({key: candidate[key] for key in ('kind', 'llvm_sha', 'build_number')})
    return requests


def main():

    event = os.environ.get('EVENT_NAME', 'workflow_dispatch')
    kind = os.environ.get('KIND') or 'all'
    kinds = BRANCHES if kind == 'all' else {kind: BRANCHES[kind]}
    manifests = event == 'push' or os.environ.get('REQUEST_SOURCE') == 'manifests'
    if manifests:
        if event == 'push':
            paths = subprocess.check_output(
                ['git', 'diff', '--name-only', os.environ['BASE_SHA'], os.environ['HEAD_SHA']], text=True).splitlines()
        else:
            paths = [MANIFESTS[k] for k in kinds]
        candidates = []
        for request in build_requests(paths):
            commit = github_json(f"repos/{FORK}/commits/{request['llvm_sha']}")
            candidates.append({**request, 'llvm_tree': commit['commit']['tree']['sha']})
    else:
        tips = {k: github_json(f'repos/{FORK}/branches/{branch}')['commit'] for k, branch in kinds.items()}
        candidates = [{
            'kind': k, 'llvm_sha': tip['sha'], 'llvm_tree': tip['commit']['tree']['sha'], 'build_number':
            str(BUILD_NUMBER)
        } for k, tip in tips.items()]
    platforms = required_platforms()
    repo = os.environ['GITHUB_REPOSITORY']
    _, raw = release_assets(repo, 'meta-llvm')
    validate_existing = event == 'workflow_dispatch' and os.environ.get('PUBLISH') == 'false'
    requests = pending_builds(repo, candidates, raw, platforms, validate_existing)
    print(json.dumps(requests))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write('requests=' + json.dumps(requests) + '\n')
        output.write('candidates=' + json.dumps(candidates) + '\n')


if __name__ == '__main__':
    main()
