import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import quote

from meta_llvm_assets import (BRANCHES, FORK, bundle_complete, github_json, published_assets, read_json_asset,
                              recover_starters, release_assets, required_platforms, source_record)
from meta_llvm_bump import candidate_text, readiness_text

REPOSITORY = 'facebookexperimental/triton'
LABEL = 'llvm-bundle-ready'


class BundleNotReady(ValueError):
    pass


def ready_name(kind, sha, build):
    prefix = 'llvm' if kind == 'core' else 'amd-llvm'
    return f'{prefix}-{sha[:8]}-{build}.ready.json'


def validate_sdk_records(source, sums, records):
    if set(records) != set(sums):
        raise ValueError('Ready record lacks SDK validation provenance')
    for platform, digest in sums.items():
        record = records[platform]
        if (record.get('schema_version') != 1 or record.get('source') != source or record.get('platform') != platform
                or record.get('sha256') != digest):
            raise ValueError('SDK validation provenance differs from the bundle')
        if not re.fullmatch(r'https://github\.com/facebookexperimental/triton/actions/runs/[0-9]+', record.get(
                'run', '')):
            raise ValueError('SDK validation has an invalid run')
        if type(record.get('attempt')) is not int or record['attempt'] < 1:
            raise ValueError('SDK validation has an invalid attempt')
        if not re.fullmatch(r'[0-9a-f]{40}', record.get('workflow_sha', '')):
            raise ValueError('SDK validation has an invalid workflow revision')


def ready_receipt(repo, kind, sha, tree, build, assets, platforms):
    asset = assets.get(ready_name(kind, sha, build))
    if asset is None:
        return None
    receipt = read_json_asset(repo, asset)
    expected = source_record(kind, sha, tree, build)
    prefix = 'llvm' if kind == 'core' else 'amd-llvm'
    if receipt.get('schema_version') != 1 or receipt.get('source') != expected:
        raise ValueError('Ready record has mismatched source provenance')
    sums = {
        platform: assets[f'{prefix}-{sha[:8]}-{platform}-{build}.tar.gz']['digest'].removeprefix('sha256:')
        for platform in platforms
    }
    if receipt.get('sha256sum') != sums:
        raise ValueError('Ready record has mismatched platform digests')
    if not re.fullmatch(r'https://github\.com/facebookexperimental/triton/actions/runs/[0-9]+',
                        receipt.get('build_run', '')):
        raise ValueError('Ready record has an invalid build run')
    if not re.fullmatch(r'[0-9a-f]{40}', receipt.get('workflow_sha', '')):
        raise ValueError('Ready record has an invalid workflow revision')
    if type(receipt.get('publication_attempt')) is not int or receipt['publication_attempt'] < 1:
        raise ValueError('Ready record has an invalid publication attempt')
    validate_sdk_records(expected, sums, receipt.get('sdk_validation', {}))
    return receipt


def publish_receipt(repo, source, platforms, validated_dir, run, attempt, workflow_sha):
    _, raw = release_assets(repo, 'meta-llvm')
    assets = published_assets(raw)
    kind, sha, build = source['kind'], source['llvm_sha'], source['build_number']
    if not bundle_complete(kind, sha, source['llvm_tree'], assets, platforms, build):
        raise ValueError('Bundle is incomplete after publication')
    name = ready_name(kind, sha, build)
    if ready_receipt(repo, kind, sha, source['llvm_tree'], build, assets, platforms):
        return
    prefix = 'llvm' if kind == 'core' else 'amd-llvm'
    sums, records = {}, {}
    for platform in platforms:
        archive = f'{prefix}-{sha[:8]}-{platform}-{build}.tar.gz'
        digest = assets[archive]['digest'].removeprefix('sha256:')
        if (validated_dir / f'{archive}.sha256').read_text() != f'{digest}  {archive}\n':
            raise ValueError(f'{archive}: SDK validation did not use the published archive')
        sums[platform] = digest
        records[platform] = json.loads((validated_dir / f'{archive}.validation.json').read_text())
    validate_sdk_records(source, sums, records)
    receipt = {
        'schema_version': 1, 'source': source, 'sha256sum': sums, 'build_run': run, 'publication_attempt': attempt,
        'workflow_sha': workflow_sha, 'sdk_validation': records
    }
    path = validated_dir / name
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    recover_starters(repo, raw, {name})
    subprocess.run(['gh', 'release', 'upload', 'meta-llvm', str(path), '--repo', repo], check=True)


def validated_candidate(repo, candidate, platforms):
    kind, sha, tree, build = candidate['kind'], candidate['llvm_sha'], candidate['llvm_tree'], int(
        candidate['build_number'])
    tip = github_json(f'repos/{FORK}/branches/{BRANCHES[kind]}')['commit']
    if tip['sha'] != sha:
        return None
    if tip['commit']['tree']['sha'] != tree:
        raise ValueError('Candidate tree differs from the LLVM commit')
    _, raw = release_assets(repo, 'meta-llvm')
    assets = published_assets(raw)
    if not bundle_complete(kind, sha, tree, assets, platforms, build):
        raise BundleNotReady('Cannot hand off an incomplete bundle')
    receipt = ready_receipt(repo, kind, sha, tree, build, assets, platforms)
    if receipt is None:
        raise BundleNotReady('Cannot hand off a bundle without successful SDK validation')
    return assets, receipt


def issue_marker(kind, sha, build):
    return f'<!-- llvm-bundle-ready beta/{kind}/{sha}/{build} -->'


def existing_issue(repo, marker):
    pages = github_json(f'repos/{repo}/issues?state=all&per_page=100', paginate=True)
    return next((issue['html_url']
                 for page in pages
                 for issue in page
                 if 'pull_request' not in issue and marker in (issue.get('body') or '')), None)


def claim_issue_creation(repo, kind, sha, build, marker):
    # INVARIANT: the caller holds the readiness-issue workflow lock.
    name = ready_name(kind, sha, build).replace('.ready.json', '.issue-request.json')
    intent = {'schema_version': 1, 'candidate': marker}
    _, raw = release_assets(repo, 'meta-llvm')
    assets = {asset['name']: asset for asset in raw}
    if name in assets:
        if read_json_asset(repo, assets[name]) != intent:
            raise ValueError('Issue submission intent has a different candidate')
        raise RuntimeError('Previous issue-create outcome is unresolved; verify it before retrying creation')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / name
        path.write_text(json.dumps(intent, sort_keys=True) + '\n')
        result = subprocess.run(
            ['gh', 'release', 'upload', 'meta-llvm', str(path), '--repo', repo], capture_output=True, text=True)
        result.check_returncode()
    return name, intent


def clear_rejected_issue_intent(repo, name, intent):
    _, raw = release_assets(repo, 'meta-llvm')
    asset = published_assets(raw).get(name)
    if asset is not None and read_json_asset(repo, asset) == intent:
        github_json(f"repos/{repo}/releases/assets/{asset['id']}", method='DELETE')


def ensure_issue(repo, candidate, platforms):
    validated = validated_candidate(repo, candidate, platforms)
    if validated is None:
        return 'Candidate superseded before readiness notification'
    assets, _ = validated
    kind, sha, build = candidate['kind'], candidate['llvm_sha'], int(candidate['build_number'])
    marker = issue_marker(kind, sha, build)
    existing = existing_issue(repo, marker)
    if existing:
        return existing
    if github_json(f'repos/{repo}/labels/{LABEL}', missing_ok=True) is None:
        github_json(
            f'repos/{repo}/labels', method='POST', payload={
                'name': LABEL, 'color': '5319e7', 'description':
                'Validated LLVM bundle ready for a manifest-update review diff'
            })
    prefix = 'llvm' if kind == 'core' else 'amd-llvm'
    body = readiness_text(kind, sha, candidate['llvm_tree'], build, BRANCHES[kind],
                          assets[f'{prefix}-{sha[:8]}-{build}.source.json']['browser_download_url'],
                          assets[ready_name(kind, sha, build)]['browser_download_url'], marker)
    name, intent = claim_issue_creation(repo, kind, sha, build, marker)
    try:
        return github_json(
            f'repos/{repo}/issues', method='POST',
            payload={'title': f'[LLVM bundle ready] beta {kind} {sha} build {build}', 'body': body, 'labels':
                     [LABEL]})['html_url']
    except subprocess.CalledProcessError as error:
        existing = existing_issue(repo, marker)
        if existing:
            return existing
        if (error.stderr or '').rstrip().endswith(tuple(f'(HTTP {s})' for s in (401, 403, 404, 422))):
            clear_rejected_issue_intent(repo, name, intent)
        raise


def read_manifest(repo, path, ref):
    obj = github_json(f'repos/{repo}/contents/{path}?ref={quote(ref, safe="")}')
    manifest = json.loads(base64.b64decode(obj['content']))
    if type(manifest['build_number']) is not int:
        raise ValueError('Meta LLVM packaging revision must be an integer')
    return manifest


def dispatch_consumer_tests(repo, kind, branch, head, pr):
    # The AMD SDK only builds the AMD codegen library, so NVIDIA GPU tests cannot observe it.
    workflows = {
        'gpu-tests.yml': {'platforms': 'mi350'} if kind == 'amd' else {}, 'tlx-ops-perf-smoke.yml': {'pr': str(pr)}
    }
    for workflow, inputs in workflows.items():
        runs = github_json(f'repos/{repo}/actions/workflows/{workflow}/runs?head_sha={head}&per_page=100')
        if any(run['conclusion'] != 'action_required' for run in runs['workflow_runs']):
            print(f'{workflow} already has a run for {head}', flush=True)
            continue
        github_json(f'repos/{repo}/actions/workflows/{workflow}/dispatches', method='POST',
                    payload={'ref': branch, 'inputs': inputs})
        print(f'Dispatched {workflow} on {branch} at {head}', flush=True)


def ensure_candidate_pr(repo, candidate, platforms, base):
    validated = validated_candidate(repo, candidate, platforms)
    if validated is None:
        return 'Candidate superseded before draft PR creation'
    assets, receipt = validated
    if re.fullmatch(r'[0-9a-f]{40}', base) is None:
        raise ValueError('Candidate PR requires a full Triton base SHA')
    if github_json(f'repos/{repo}/compare/{base}...main')['status'] not in ('ahead', 'identical'):
        raise ValueError('Candidate baseline is not on main history')
    for workflow in ('gpu-tests.yml', 'tlx-ops-perf-smoke.yml'):
        if github_json(f'repos/{repo}/contents/.github/workflows/{workflow}?ref={base}', missing_ok=True) is None:
            raise ValueError(f'Candidate baseline has no {workflow} to test the PR')
    kind, sha, build = candidate['kind'], candidate['llvm_sha'], int(candidate['build_number'])
    path = 'cmake/meta-amd-llvm-info.json' if kind == 'amd' else 'cmake/meta-llvm-info.json'
    old = read_manifest(repo, path, base)
    if old['llvm_hash'] == sha and int(old['build_number']) > build:
        return 'The baseline selects a newer packaging revision of this LLVM source'
    new = dict(old, llvm_hash=sha, build_number=build,
               release_base_url=f'https://github.com/{repo}/releases/download/meta-llvm',
               sha256sum=receipt['sha256sum'])
    new.pop('based_on_oai_llvm_hash', None)
    if new == old:
        return 'The baseline already selects this bundle'
    content = json.dumps(new, indent=2) + '\n'
    branch = f'scratch/llvm-canary/{kind}-{sha[:8]}-{build}-{base[:8]}'
    title, message, body = candidate_text(kind, old, receipt['source'],
                                          assets[ready_name(kind, sha, build)]['browser_download_url'], base)
    marker = f'<!-- llvm-candidate beta/{kind}/{sha}/{build}/{base} -->'
    pages = github_json(f'repos/{repo}/pulls?state=all&head=facebookexperimental:{quote(branch, safe="")}&per_page=100',
                        paginate=True)
    pulls = [pull for page in pages for pull in page]
    for pull in pulls:
        if (pull['state'] != 'open' or not pull['draft'] or pull['base']['ref'] != 'main'
                or marker not in (pull.get('body') or '')):
            raise ValueError('Existing candidate PR was closed or changed; it will not be recreated')
    existing = github_json(f'repos/{repo}/git/ref/heads/{branch}', missing_ok=True)
    if pulls and existing is None:
        raise ValueError('Existing candidate PR has no branch; it will not be recreated')
    if existing:
        head = existing['object']['sha']
        commit = github_json(f'repos/{repo}/git/commits/{head}')
        changed = github_json(f'repos/{repo}/compare/{base}...{head}')['files']
        if ([p['sha'] for p in commit['parents']] != [base] or read_manifest(repo, path, head) != new
                or [f['filename'] for f in changed] != [path]):
            raise ValueError('Existing candidate branch differs from the expected one-file update')
    else:
        parent = github_json(f'repos/{repo}/git/commits/{base}')
        blob = github_json(f'repos/{repo}/git/blobs', method='POST', payload={'content': content, 'encoding': 'utf-8'})
        tree = github_json(
            f'repos/{repo}/git/trees', method='POST', payload={
                'base_tree': parent['tree']['sha'], 'tree':
                [{'path': path, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']}]
            })
        commit = github_json(f'repos/{repo}/git/commits', method='POST',
                             payload={'message': message, 'tree': tree['sha'], 'parents': [base]})
        head = commit['sha']
        github_json(f'repos/{repo}/git/refs', method='POST', payload={'ref': 'refs/heads/' + branch, 'sha': head})
    print(f'Candidate branch: {branch} at {head}. Main was not changed.', flush=True)
    if pulls:
        if any(pull['head']['sha'] != head for pull in pulls):
            raise ValueError('Existing candidate PR head changed; it will not be overwritten')
        dispatch_consumer_tests(repo, kind, branch, head, pulls[0]['number'])
        return pulls[0]['html_url']
    pull = github_json(f'repos/{repo}/pulls', method='POST',
                       payload={'title': title, 'body': body, 'head': branch, 'base': 'main', 'draft': True})
    if (not pull['draft'] or pull['head']['sha'] != head or pull['base']['ref'] != 'main' or pull['state'] != 'open'):
        raise ValueError('Created candidate PR does not match the prepared head, base or draft state')
    print('PR author: ' + pull['user']['login'], flush=True)
    dispatch_consumer_tests(repo, kind, branch, head, pull['number'])
    return pull['html_url']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['publish', 'issue', 'pr'])
    args = parser.parse_args()
    repo = os.environ['GITHUB_REPOSITORY']
    if repo != REPOSITORY:
        raise ValueError('Readiness publication is restricted to the Triton repository')
    platforms = required_platforms()
    if args.action == 'publish':
        publish_receipt(repo, json.loads(os.environ['SOURCE_JSON']), platforms, Path(os.environ['VALIDATED_DIR']),
                        f"https://github.com/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                        int(os.environ['GITHUB_RUN_ATTEMPT']), os.environ['GITHUB_SHA'])
        return
    candidates = json.loads(os.environ['CANDIDATES'])
    failures = []
    for candidate in candidates:
        try:
            if args.action == 'issue':
                print(ensure_issue(repo, candidate, platforms), flush=True)
            else:
                print(ensure_candidate_pr(repo, candidate, platforms, os.environ['CANDIDATE_BASE_SHA']), flush=True)
        except BundleNotReady as error:
            print(error, flush=True)
        except Exception as error:
            detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
            print(f"{candidate['kind']} handoff failed: {detail}", flush=True)
            failures.append(candidate['kind'])
    if failures:
        raise RuntimeError('Readiness handoff failed for ' + ', '.join(failures))


if __name__ == '__main__':
    main()
