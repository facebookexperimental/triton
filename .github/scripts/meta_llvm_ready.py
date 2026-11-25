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
BOT_EMAIL = '41898282+github-actions[bot]@users.noreply.github.com'
CANDIDATE_LABEL = 'llvm-candidate'


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


def ensure_label(repo, name, description):
    if github_json(f'repos/{repo}/labels/{name}', missing_ok=True) is None:
        github_json(f'repos/{repo}/labels', method='POST',
                    payload={'name': name, 'color': '5319e7', 'description': description})


def ensure_issue(repo, candidate, platforms, pr=None):
    pr = pr or {'status': 'not requested'}
    if pr['status'] in ('superseded', 'adopted', 'handled'):
        return f"No readiness issue: the candidate is {pr['status']}"
    if pr.get('number') and github_json(f"repos/{repo}/pulls/{pr['number']}")['state'] != 'open':
        return f"No readiness issue: draft PR #{pr['number']} was closed"
    validated = validated_candidate(repo, candidate, platforms)
    if validated is None:
        return 'Candidate superseded before readiness notification'
    assets, _ = validated
    kind, sha, build = candidate['kind'], candidate['llvm_sha'], int(candidate['build_number'])
    marker = issue_marker(kind, sha, build)
    prefix = 'llvm' if kind == 'core' else 'amd-llvm'
    body = readiness_text(kind, sha, candidate['llvm_tree'], build, BRANCHES[kind],
                          assets[f'{prefix}-{sha[:8]}-{build}.source.json']['browser_download_url'],
                          assets[ready_name(kind, sha, build)]['browser_download_url'], marker, pr,
                          os.environ.get('RUN_URL', ''))
    pages = github_json(f'repos/{repo}/issues?state=all&per_page=100', paginate=True)
    issue = next((i for page in pages for i in page if 'pull_request' not in i and marker in (i.get('body') or '')),
                 None)
    if issue:
        if pr.get('url') and pr['url'] not in (issue.get('body') or '') and issue['state'] == 'open':
            github_json(f"repos/{repo}/issues/{issue['number']}", method='PATCH', payload={'body': body})
        return issue['html_url']
    ensure_label(repo, LABEL, 'Validated LLVM bundle with a draft candidate PR to adopt')
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


MANIFESTS = {'core': 'cmake/meta-llvm-info.json', 'amd': 'cmake/meta-amd-llvm-info.json'}
CANDIDATE = re.compile(r'<!-- llvm-candidate beta/(core|amd)/([0-9a-f]{40})/([0-9]+)/([0-9a-f]{40}) -->')
READY = re.compile(r'<!-- llvm-bundle-ready beta/(core|amd)/([0-9a-f]{40})/([0-9]+) -->')


def labeled(repo, label, state, pattern, pulls):
    pages = github_json(f'repos/{repo}/issues?labels={label}&state={state}&per_page=100', paginate=True)
    found = []
    for item in (item for page in pages for item in page if ('pull_request' in item) == pulls):
        match = pattern.search(item.get('body') or '')
        if match:
            found.append((item, match.groups()))
    return found


def expected_manifest(repo, kind, base, sha, build, receipt):
    old = read_manifest(repo, MANIFESTS[kind], base)
    new = dict(old, llvm_hash=sha, build_number=build,
               release_base_url=f'https://github.com/{repo}/releases/download/meta-llvm',
               sha256sum=receipt['sha256sum'])
    new.pop('based_on_oai_llvm_hash', None)
    return old, new


def bot_head(repo, kind, base, head, new=None):
    commit = github_json(f'repos/{repo}/git/commits/{head}')
    changed = github_json(f'repos/{repo}/compare/{base}...{head}')['files']
    return ([p['sha'] for p in commit['parents']] == [base] and [f['filename'] for f in changed] == [MANIFESTS[kind]]
            and (new is None or read_manifest(repo, MANIFESTS[kind], head) == new))


def check_base(repo, base):
    if re.fullmatch(r'[0-9a-f]{40}', base) is None:
        raise ValueError('Candidate PR requires a full Triton base SHA')
    if github_json(f'repos/{repo}/compare/{base}...main')['status'] not in ('ahead', 'identical'):
        raise ValueError('Candidate baseline is not on main history')
    for workflow in ('gpu-tests.yml', 'tlx-ops-perf-smoke.yml'):
        if github_json(f'repos/{repo}/contents/.github/workflows/{workflow}?ref={base}', missing_ok=True) is None:
            raise ValueError(f'Candidate baseline has no {workflow} to test the PR')


def newer(sha, build, other_sha, other_build):
    """Whether candidate (sha, build) replaces (other_sha, other_build) of the same kind."""
    return other_sha != sha or int(other_build) < build


PASSIVE_EVENTS = {'committed', 'cross-referenced', 'mentioned', 'referenced', 'subscribed', 'unsubscribed'}
EDITED = 'query($o: String!, $n: String!, $p: Int!) {repository(owner: $o, name: $n) {pullRequest(number: $p) {lastEditedAt}}}'


def untouched(repo, kind, base, number):
    """Whether a candidate PR is still the bot's draft: no human edit, event, review or commit since creation."""
    pull = github_json(f'repos/{repo}/pulls/{number}')
    if not pull['draft'] or pull['base']['ref'] != 'main':
        return False
    pages = github_json(f'repos/{repo}/pulls/{number}/commits?per_page=100', paginate=True)
    commits = [c for page in pages for c in page]
    if len(commits) != 1 or commits[0]['sha'] != pull['head']['sha']:
        return False
    commit = commits[0]['commit']
    if (commit['author']['email'] != BOT_EMAIL or commit['committer']['email'] != 'noreply@github.com'
            or not commit['verification']['verified']):
        return False
    pages = github_json(f'repos/{repo}/issues/{number}/timeline?per_page=100', paginate=True)
    for event in (event for page in pages for event in page):
        actor = event.get('actor') or event.get('user') or {}
        if event.get('event') not in PASSIVE_EVENTS and actor.get('type') != 'Bot':
            return False
    owner, name = repo.split('/')
    edited = github_json('graphql', method='POST',
                         payload={'query': EDITED, 'variables': {'o': owner, 'n': name, 'p': number}})
    if edited['data']['repository']['pullRequest']['lastEditedAt']:
        return False
    return bot_head(repo, kind, base, pull['head']['sha'])


def close_superseded(repo, kind, sha, build, replacement):
    """Close older untouched draft candidates of one kind, and readiness issues no open candidate PR still tracks."""
    note = {'body': f'Superseded by #{replacement}, which proposes a newer bundle.'}
    proposals = labeled(repo, CANDIDATE_LABEL, 'all', CANDIDATE, True)
    closed, kept = [], []
    for item, (k, old_sha, old_build, base) in proposals:
        if (k != kind or item['number'] == replacement or item['state'] != 'open'
                or not newer(sha, build, old_sha, old_build)):
            continue
        if not untouched(repo, kind, base, item['number']):
            kept.append(item['number'])
            continue
        github_json(f"repos/{repo}/issues/{item['number']}/comments", method='POST', payload=note)
        github_json(f"repos/{repo}/pulls/{item['number']}", method='PATCH', payload={'state': 'closed'})
        closed.append(item['number'])
    tracked = {groups[:3] for item, groups in proposals if item['state'] == 'open' and item['number'] not in closed}
    for issue, groups in labeled(repo, LABEL, 'open', READY, False):
        if groups[0] == kind and newer(sha, build, groups[1], groups[2]) and groups not in tracked:
            github_json(f"repos/{repo}/issues/{issue['number']}/comments", method='POST', payload=note)
            github_json(f"repos/{repo}/issues/{issue['number']}", method='PATCH',
                        payload={'state': 'closed', 'state_reason': 'not_planned'})
    return closed, sorted(kept)


def finish_proposal(repo, result, labels, branch, head):
    """Label, test and supersede around a proposal PR; a failure keeps the PR link in the result."""
    try:
        if CANDIDATE_LABEL not in labels:
            ensure_label(repo, CANDIDATE_LABEL, 'Draft PR proposing a validated Meta LLVM bundle')
            github_json(f"repos/{repo}/issues/{result['number']}/labels", method='POST',
                        payload={'labels': [CANDIDATE_LABEL]})
        dispatch_consumer_tests(repo, result['kind'], branch, head, result['number'])
        result['closed'], result['kept'] = close_superseded(repo, result['kind'], result['llvm_sha'],
                                                            result['build_number'], result['number'])
    except Exception as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        result['error'] = detail
    return result


def ensure_candidate_pr(repo, candidate, platforms, base=''):
    kind, sha, build = candidate['kind'], candidate['llvm_sha'], int(candidate['build_number'])
    result = {'kind': kind, 'llvm_sha': sha, 'build_number': build}
    validated = validated_candidate(repo, candidate, platforms)
    if validated is None:
        return dict(result, status='superseded')
    assets, receipt = validated
    pin = read_manifest(repo, MANIFESTS[kind], 'main')
    if pin['llvm_hash'] == sha and pin['build_number'] >= build:
        return dict(result, status='adopted')
    branch = f'scratch/llvm-candidate/{kind}-{sha[:8]}-{build}'
    # INVARIANT: one proposal per kind, LLVM commit and packaging revision, whatever Triton base it started from.
    proposals = labeled(repo, CANDIDATE_LABEL, 'all', CANDIDATE, True)
    refs = github_json(f'repos/{repo}/git/matching-refs/heads/scratch/llvm-candidate/{kind}-{sha[:8]}-')
    builds = [int(b) for _, (k, other, b, _) in proposals if (k, other) == (kind, sha)]
    builds += [int(ref['ref'].rsplit('-', 1)[1]) for ref in refs if ref['ref'].rsplit('-', 1)[1].isdigit()]
    if any(b > build for b in builds):
        return dict(result, status='superseded')
    numbers = {item['number'] for item, groups in proposals if groups[:3] == (kind, sha, str(build))}
    pages = github_json(f'repos/{repo}/pulls?state=all&head=facebookexperimental:{quote(branch, safe="")}&per_page=100',
                        paginate=True)
    numbers |= {pull['number'] for page in pages for pull in page}
    if numbers:
        pull = github_json(f'repos/{repo}/pulls/{max(numbers)}')
        match = CANDIDATE.search(pull.get('body') or '')
        result.update(number=pull['number'], url=pull['html_url'])
        if (pull['state'] != 'open' or not pull['draft'] or pull['base']['ref'] != 'main' or match is None
                or match.groups()[:3] != (kind, sha, str(build))):
            return dict(result, status='handled')
        old, new = expected_manifest(repo, kind, match.group(4), sha, build, receipt)
        if not bot_head(repo, kind, match.group(4), pull['head']['sha'], new):
            return dict(result, status='handled')
        result.update(status='open', old=old['llvm_hash'])
        return finish_proposal(repo, result, [l['name'] for l in pull['labels']], pull['head']['ref'],
                               pull['head']['sha'])
    ref = github_json(f'repos/{repo}/git/ref/heads/{branch}', missing_ok=True)
    if ref:
        head = ref['object']['sha']
        base = github_json(f'repos/{repo}/git/commits/{head}')['parents'][0]['sha']
    else:
        head = None
        base = base or github_json(f'repos/{repo}/commits/main')['sha']
    check_base(repo, base)
    old, new = expected_manifest(repo, kind, base, sha, build, receipt)
    if new == old:
        return dict(result, status='adopted')
    title, message, body = candidate_text(kind, old, receipt['source'],
                                          assets[ready_name(kind, sha, build)]['browser_download_url'], base)
    if head and not bot_head(repo, kind, base, head, new):
        raise ValueError('Existing candidate branch differs from the expected one-file update')
    if head is None:
        parent = github_json(f'repos/{repo}/git/commits/{base}')
        blob = github_json(f'repos/{repo}/git/blobs', method='POST',
                           payload={'content': json.dumps(new, indent=2) + '\n', 'encoding': 'utf-8'})
        tree = github_json(
            f'repos/{repo}/git/trees', method='POST', payload={
                'base_tree': parent['tree']['sha'], 'tree':
                [{'path': MANIFESTS[kind], 'mode': '100644', 'type': 'blob', 'sha': blob['sha']}]
            })
        head = github_json(f'repos/{repo}/git/commits', method='POST',
                           payload={'message': message, 'tree': tree['sha'], 'parents': [base]})['sha']
        github_json(f'repos/{repo}/git/refs', method='POST', payload={'ref': 'refs/heads/' + branch, 'sha': head})
    print(f'Candidate branch: {branch} at {head}. Main was not changed.', flush=True)
    pull = github_json(f'repos/{repo}/pulls', method='POST',
                       payload={'title': title, 'body': body, 'head': branch, 'base': 'main', 'draft': True})
    if (not pull['draft'] or pull['head']['sha'] != head or pull['base']['ref'] != 'main' or pull['state'] != 'open'):
        raise ValueError('Created candidate PR does not match the prepared head, base or draft state')
    result.update(status='created', number=pull['number'], url=pull['html_url'], old=old['llvm_hash'])
    return finish_proposal(repo, result, [], branch, head)


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
    prs = {pr['kind']: pr for pr in json.loads(os.environ.get('CANDIDATE_RESULTS') or '[]')}
    results, failures = [], []
    for candidate in candidates:
        try:
            if args.action == 'issue':
                print(ensure_issue(repo, candidate, platforms, prs.get(candidate['kind'])), flush=True)
            else:
                results.append(ensure_candidate_pr(repo, candidate, platforms, os.environ.get('CANDIDATE_BASE_SHA',
                                                                                              '')))
                print(json.dumps(results[-1]), flush=True)
                if results[-1].get('error'):
                    failures.append(candidate['kind'])
        except BundleNotReady as error:
            print(error, flush=True)
        except Exception as error:
            detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
            print(f"{candidate['kind']} handoff failed: {detail}", flush=True)
            failures.append(candidate['kind'])
            results.append({'kind': candidate['kind'], 'status': 'failed'})
    if args.action == 'pr' and os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write('results=' + json.dumps(results) + '\n')
    if failures:
        raise RuntimeError('Readiness handoff failed for ' + ', '.join(failures))


if __name__ == '__main__':
    main()
