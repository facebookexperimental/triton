"""Select GPU suites using the former workflows' ordered GitHub path filters."""
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

PATH_FILTERS = Path(__file__).resolve().parents[1] / "gpu-test-paths.json"
BUMP_PREFIX = "scratch/meta-llvm-bump/"
WORKFLOW = ".github/workflows/gpu-tests.yml"


def matches(path, pattern):
    # GitHub's * stays within a directory; ** crosses directories, and **/
    # also matches zero directories. These are the only wildcards in our rules.
    parts = []
    i = 0
    while i < len(pattern):
        if pattern[i:i + 3] == "**/":
            parts.append("(?:.*/)?")
            i += 3
        elif pattern[i:i + 2] == "**":
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.fullmatch("".join(parts), path) is not None


def select_platforms(paths, filters):
    selected = {}
    for platform, patterns in filters.items():
        selected[platform] = False
        for path in paths:
            included = False
            for pattern in patterns:
                if matches(path, pattern.removeprefix("!")):
                    included = not pattern.startswith("!")
            if included:
                selected[platform] = True
                break
    return selected


def changed_paths(event_name, event):
    if event_name == "pull_request":
        pr = event["pull_request"]
        revisions = [f"{pr['base']['sha']}...{pr['head']['sha']}"]
    elif event_name == "push" and event["before"].strip("0"):
        revisions = [event["before"], event["after"]]
    else:
        return None  # Schedules, manual runs, and new branches run all suites.
    # Disable rename detection so both old and new paths participate in filters.
    result = subprocess.run(["git", "diff", "--name-only", "--no-renames", "-z", *revisions, "--"], check=True,
                            capture_output=True)
    return result.stdout.decode().rstrip("\0").split("\0") if result.stdout else []


def bump_ref(event_name, event):
    """The Meta LLVM bump branch this event is about, or None."""
    if event_name == "push" and event.get("ref", "").startswith("refs/heads/" + BUMP_PREFIX):
        return event["ref"].removeprefix("refs/heads/")
    pr = event.get("pull_request") if event_name == "pull_request" else None
    if pr and pr["head"]["ref"].startswith(
            BUMP_PREFIX) and pr["head"]["repo"]["full_name"] == pr["base"]["repo"]["full_name"]:
        return pr["head"]["ref"]
    return None


def passed_push_run(runs):
    """URL of a successful push-triggered run of this workflow among runs, or None.

    The bot never pushes to a bump branch once its PR is open, so the PR head is the commit that run tested.
    """
    for run in runs:
        if run["event"] == "push" and run["path"] == WORKFLOW and run["conclusion"] == "success":
            return run["html_url"]
    return None


def fetch_runs(repo, sha):
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/runs?head_sha={sha}&event=push",
                                     headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["workflow_runs"]


def main():
    filters = json.loads(PATH_FILTERS.read_text())
    try:
        event_name = os.environ["GITHUB_EVENT_NAME"]
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        bump = bump_ref(event_name, event)
        if bump and event_name == "push":
            paths = None  # A bump commit verifies the whole line, whatever the diff to the old bump.
        elif bump and (url := passed_push_run(
                fetch_runs(os.environ["GITHUB_REPOSITORY"], event["pull_request"]["head"]["sha"]))):
            print(f"{bump} head already passed GPU tests on push: {url}")
            paths = []
        else:
            paths = changed_paths(event_name, event)
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"::warning::Cannot determine changed paths; running all GPU suites: {exc}")
        paths = None
    selected = dict.fromkeys(filters, True) if paths is None else select_platforms(paths, filters)
    selected["build"] = any(selected.values())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"GPU selection for {sha}: {selected}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"sha={sha}\n")
        for name, enabled in selected.items():
            output.write(f"{name}={str(enabled).lower()}\n")


if __name__ == "__main__":
    main()
