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
    # INVARIANT: the caller holds the candidate handoff workflow lock.
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


def ensure_issue(repo, candidate, platforms, pr):
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
                'name': LABEL, 'color': '5319e7', 'description': 'Validated LLVM bundle with a draft candidate PR'
            })
    prefix = 'llvm' if kind == 'core' else 'amd-llvm'
    body = readiness_text(kind, sha, candidate['llvm_tree'], build, BRANCHES[kind],
                          assets[f'{prefix}-{sha[:8]}-{build}.source.json']['browser_download_url'],
                          assets[ready_name(kind, sha, build)]['browser_download_url'], marker, pr,
                          os.environ.get('RUN_URL', ''))
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


def dispatch_consumer_tests(repo, kind, branch, pr):
    # The AMD SDK only builds the AMD codegen library, so NVIDIA GPU tests cannot observe it.
    workflows = {
        'gpu-tests.yml': {'platforms': 'mi350'} if kind == 'amd' else {}, 'tlx-ops-perf-smoke.yml': {'pr': str(pr)}
    }
    for workflow, inputs in workflows.items():
        github_json(f'repos/{repo}/actions/workflows/{workflow}/dispatches', method='POST',
                    payload={'ref': branch, 'inputs': inputs})
        print(f'Dispatched {workflow} on {branch}', flush=True)


CANDIDATE = re.compile(r'<!-- llvm-candidate beta/(core|amd)/([0-9a-f]{40})/([0-9]+)/')


def close_older_drafts(repo, kind, branch, replacement):
    """Close the kind's other draft candidates and their readiness issues; a PR marked ready for review stays."""
    pages = github_json(f'repos/{repo}/pulls?state=open&per_page=100', paginate=True)
    pulls = [pull for page in pages for pull in page]
    issues = [
        issue for page in github_json(f'repos/{repo}/issues?labels={LABEL}&state=open&per_page=100', paginate=True)
        for issue in page
    ]
    note = {'body': f'Superseded by #{replacement}, which proposes a newer bundle.'}
    for pull in pulls:
        match = CANDIDATE.search(pull.get('body') or '')
        if (not pull['draft'] or match is None or match.group(1) != kind or pull['head']['ref'] == branch
                or not pull['head']['ref'].startswith('scratch/llvm-canary/')):
            continue
        marker = issue_marker(*match.groups())
        for item in [pull] + [issue for issue in issues if marker in (issue.get('body') or '')]:
            github_json(f"repos/{repo}/issues/{item['number']}/comments", method='POST', payload=note)
            github_json(f"repos/{repo}/issues/{item['number']}", method='PATCH', payload={'state': 'closed'})
        print(f"Closed superseded candidate #{pull['number']}", flush=True)


def ensure_candidate_pr(repo, candidate, platforms):
    """Create the candidate's draft PR; return {'url', 'old'}, or None when there is nothing to hand off."""
    validated = validated_candidate(repo, candidate, platforms)
    if validated is None:
        print('Candidate superseded before draft PR creation', flush=True)
        return None
    assets, receipt = validated
    kind, sha, build = candidate['kind'], candidate['llvm_sha'], int(candidate['build_number'])
    path = 'cmake/meta-amd-llvm-info.json' if kind == 'amd' else 'cmake/meta-llvm-info.json'
    base = github_json(f'repos/{repo}/commits/main')['sha']
    old = read_manifest(repo, path, base)
    if old['llvm_hash'] == sha and old['build_number'] >= build:
        print('Main already selects this bundle', flush=True)
        return None
    # Branches created before proposals were keyed this way also end in the base commit.
    branch = f'scratch/llvm-canary/{kind}-{sha[:8]}-{build}'
    pages = github_json(f'repos/{repo}/pulls?state=open&per_page=100', paginate=True)
    for pull in (pull for page in pages for pull in page):
        if pull['head']['ref'] == branch or pull['head']['ref'].startswith(branch + '-'):
            return {'url': pull['html_url'], 'old': old['llvm_hash']}
    pages = github_json(f'repos/{repo}/pulls?state=closed&head=facebookexperimental:{quote(branch, safe="")}',
                        paginate=True)
    if any(page for page in pages):
        print(f'The candidate PR for {branch} was closed; it will not be recreated', flush=True)
        return None
    new = dict(old, llvm_hash=sha, build_number=build,
               release_base_url=f'https://github.com/{repo}/releases/download/meta-llvm',
               sha256sum=receipt['sha256sum'])
    new.pop('based_on_oai_llvm_hash', None)
    title, message, body = candidate_text(kind, old, receipt['source'],
                                          assets[ready_name(kind, sha, build)]['browser_download_url'], base)
    existing = github_json(f'repos/{repo}/git/ref/heads/{branch}', missing_ok=True)
    if existing is None:
        parent = github_json(f'repos/{repo}/git/commits/{base}')
        blob = github_json(f'repos/{repo}/git/blobs', method='POST',
                           payload={'content': json.dumps(new, indent=2) + '\n', 'encoding': 'utf-8'})
        tree = github_json(
            f'repos/{repo}/git/trees', method='POST', payload={
                'base_tree': parent['tree']['sha'], 'tree':
                [{'path': path, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']}]
            })
        commit = github_json(f'repos/{repo}/git/commits', method='POST',
                             payload={'message': message, 'tree': tree['sha'], 'parents': [base]})
        github_json(f'repos/{repo}/git/refs', method='POST',
                    payload={'ref': 'refs/heads/' + branch, 'sha': commit['sha']})
    pull = github_json(f'repos/{repo}/pulls', method='POST',
                       payload={'title': title, 'body': body, 'head': branch, 'base': 'main', 'draft': True})
    print('Created ' + pull['html_url'], flush=True)
    dispatch_consumer_tests(repo, kind, branch, pull['number'])
    close_older_drafts(repo, kind, branch, pull['number'])
    return {'url': pull['html_url'], 'old': old['llvm_hash']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['publish', 'handoff'])
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
            try:
                pr = ensure_candidate_pr(repo, candidate, platforms)
            except BundleNotReady:
                raise
            except Exception as error:
                detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
                print(f"{candidate['kind']} candidate PR failed: {detail}", flush=True)
                failures.append(candidate['kind'])
                pr = {'url': None}
            if pr is not None:
                print(ensure_issue(repo, candidate, platforms, pr), flush=True)
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
