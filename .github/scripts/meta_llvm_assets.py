import hashlib
import json
from pathlib import Path
import re
import subprocess

FORK = 'facebookexperimental/llvm-project'
BRANCHES = {'core': 'triton_beta', 'amd': 'fbtriton-beta-amd-codegen'}
BUILD_NUMBER = 1


def source_record(kind, sha, tree, build=BUILD_NUMBER):
    return {
        'schema_version': 1, 'kind': kind, 'llvm_repository': FORK, 'llvm_sha': sha, 'llvm_tree': tree, 'build_number':
        build
    }


def source_digest(kind, sha, tree, build=BUILD_NUMBER):
    content = json.dumps(source_record(kind, sha, tree, build), sort_keys=True, separators=(',', ':')) + '\n'
    return 'sha256:' + hashlib.sha256(content.encode()).hexdigest()


def required_platforms():
    import yaml
    configs = yaml.safe_load(
        Path('.github/workflows/llvm-build.yml').read_text())['jobs']['build']['strategy']['matrix']['config']
    return [f"{c['target-os']}-{c['arch']}" for c in configs if c['target-os'] in ('ubuntu', 'almalinux')]


def bundle_complete(kind, sha, tree, assets, platforms, build=BUILD_NUMBER):
    prefix = 'llvm' if kind == 'core' else 'amd-llvm'
    stem = f'{prefix}-{sha[:8]}'
    source_name = f'{stem}-{build}.source.json'
    source = assets.get(source_name)
    if source is None:
        return False
    if source.get('digest') != source_digest(kind, sha, tree, build):
        raise ValueError(f'{source_name}: published LLVM source provenance differs')
    for platform in platforms:
        name = f'{stem}-{platform}-{build}.tar.gz'
        archive, checksum = assets.get(name), assets.get(name + '.sha256')
        if archive is None or checksum is None:
            return False
        digest = archive.get('digest') or ''
        if re.fullmatch(r'sha256:[0-9a-f]{64}', digest) is None:
            raise ValueError(f'{name}: missing GitHub archive digest')
        content = f"{digest.removeprefix('sha256:')}  {name}\n"
        if checksum.get('digest') != 'sha256:' + hashlib.sha256(content.encode()).hexdigest():
            raise ValueError(f'{name}: published checksum differs from archive digest')
    return True


def read_json_asset(repo, asset):
    if asset.get('state') != 'uploaded' or asset.get('size', 0) <= 0:
        raise ValueError(f"{asset['name']}: asset is not completely uploaded")
    content = subprocess.check_output(
        ['gh', 'api', f"repos/{repo}/releases/assets/{asset['id']}", '-H', 'Accept: application/octet-stream'])
    if asset.get('digest') != 'sha256:' + hashlib.sha256(content).hexdigest():
        raise ValueError(f"{asset['name']}: asset digest differs")
    return json.loads(content)


def github_json(path, *, missing_ok=False, paginate=False, method=None, payload=None):
    command = ['gh', 'api', path]
    if paginate:
        command += ['--paginate', '--slurp']
    if method:
        command += ['--method', method]
    kwargs = {}
    if payload is not None:
        command += ['--input', '-']
        kwargs['input'] = json.dumps(payload)
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        if missing_ok and result.stderr.rstrip().endswith('(HTTP 404)'):
            return None
        result.check_returncode()
    if method == 'DELETE' or (method and not result.stdout.strip()):
        return None
    return json.loads(result.stdout)


def release_assets(repo, tag):
    release = github_json(f'repos/{repo}/releases/tags/{tag}', missing_ok=True)
    if release is None:
        return None, []
    pages = github_json(f"repos/{repo}/releases/{release['id']}/assets?per_page=100", paginate=True)
    return release, [asset for page in pages for asset in page]


def is_empty_starter(asset):
    return asset.get('state') == 'starter' and asset.get('size') == 0


def published_assets(assets):
    return {asset['name']: asset for asset in assets if not is_empty_starter(asset)}


def recover_starters(repo, assets, replacement_names):
    # INVARIANT: the caller holds the per-bundle publication lock.
    remaining = []
    for asset in assets:
        if asset['name'] in replacement_names and is_empty_starter(asset):
            path = f"repos/{repo}/releases/assets/{asset['id']}"
            current = github_json(path)
            if current['name'] == asset['name'] and is_empty_starter(current):
                github_json(path, method='DELETE')
                continue
            asset = current
        remaining.append(asset)
    return remaining
