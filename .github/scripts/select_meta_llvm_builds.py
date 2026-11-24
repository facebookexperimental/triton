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


def branch_requests(tips, assets, platforms):
    assets = published_assets(assets)
    requests = []
    for kind, tip in tips.items():
        sha, tree = tip['sha'], tip['commit']['tree']['sha']
        if re.fullmatch(r'[0-9a-f]{40}', sha) is None:
            raise ValueError(f'{BRANCHES[kind]}: expected a full immutable LLVM SHA')
        if not bundle_complete(kind, sha, tree, assets, platforms):
            requests.append({'kind': kind, 'llvm_sha': sha, 'build_number': str(BUILD_NUMBER)})
    return requests


def main():
    from meta_llvm_ready import ready_receipt

    event = os.environ.get('EVENT_NAME', 'workflow_dispatch')
    kind = os.environ.get('KIND') or 'all'
    kinds = BRANCHES if kind == 'all' else {kind: BRANCHES[kind]}
    if os.environ.get('CREATE_CANDIDATE_PR') == 'true':
        if re.fullmatch(r'[0-9a-f]{40}', os.environ.get('CANDIDATE_BASE_SHA', '')) is None:
            raise ValueError('Candidate PR creation requires a full Triton base SHA')
        if os.environ.get('PUBLISH') == 'false':
            raise ValueError('Candidate PR creation requires published bundles')
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
    assets = published_assets(raw)
    requests = []
    for candidate in candidates:
        k, sha, tree, build = candidate['kind'], candidate['llvm_sha'], candidate['llvm_tree'], int(
            candidate['build_number'])
        if (not bundle_complete(k, sha, tree, assets, platforms, build)
                or ready_receipt(repo, k, sha, tree, build, assets, platforms) is None):
            requests.append({key: candidate[key] for key in ('kind', 'llvm_sha', 'build_number')})
    print(json.dumps(requests))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write('requests=' + json.dumps(requests) + '\n')
        output.write('candidates=' + json.dumps(candidates) + '\n')


if __name__ == '__main__':
    main()
