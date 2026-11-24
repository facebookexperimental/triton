import json
import subprocess


def github_json(path, *, missing_ok=False, paginate=False, method=None):
    command = ["gh", "api", path]
    if paginate:
        command += ["--paginate", "--slurp"]
    if method:
        command += ["--method", method]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        if missing_ok and result.stderr.rstrip().endswith("(HTTP 404)"):
            return None
        result.check_returncode()
    if method == "DELETE":
        return None
    return json.loads(result.stdout)


def release_assets(repo, tag):
    release = github_json(f"repos/{repo}/releases/tags/{tag}", missing_ok=True)
    if release is None:
        return None, []
    pages = github_json(f"repos/{repo}/releases/{release['id']}/assets?per_page=100", paginate=True)
    return release, [asset for page in pages for asset in page]


def is_empty_starter(asset):
    return asset.get("state") == "starter" and asset.get("size") == 0


def published_assets(assets):
    return {asset["name"]: asset for asset in assets if not is_empty_starter(asset)}


def recover_starters(repo, assets, replacement_names):
    # INVARIANT: the caller holds the per-bundle publication lock.
    remaining = []
    for asset in assets:
        if asset["name"] in replacement_names and is_empty_starter(asset):
            path = f"repos/{repo}/releases/assets/{asset['id']}"
            current = github_json(path)
            if current["name"] == asset["name"] and is_empty_starter(current):
                github_json(path, method="DELETE")
                continue
            asset = current
        remaining.append(asset)
    return remaining
