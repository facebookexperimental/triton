"""Select GPU suites using the former workflows' ordered GitHub path filters."""
import json
import os
from pathlib import Path
import re
import subprocess

PATH_FILTERS = Path(__file__).resolve().parents[1] / "gpu-test-paths.json"


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


def main():
    filters = json.loads(PATH_FILTERS.read_text())
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        paths = changed_paths(os.environ["GITHUB_EVENT_NAME"], event)
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
