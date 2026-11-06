import json
import os
from pathlib import Path
import re
import subprocess

MANIFESTS = {
    "core": "cmake/meta-llvm-build-info.json",
    "amd": "cmake/meta-amd-llvm-build-info.json",
}


def build_requests(paths, root=Path(".")):
    requests = []
    for kind, path in MANIFESTS.items():
        if path not in paths:
            continue
        info = json.loads((root / path).read_text())
        if info["repository"] != "facebookexperimental/llvm-project":
            raise ValueError(f"{path}: expected the Meta LLVM fork")
        if re.fullmatch(r"[0-9a-f]{40}", info["llvm_hash"]) is None:
            raise ValueError(f"{path}: expected a full immutable LLVM SHA")
        if type(info["build_number"]) is not int or info["build_number"] < 0:
            raise ValueError(f"{path}: expected a non-negative integer build number")
        requests.append({
            "kind": kind,
            "llvm_sha": info["llvm_hash"],
            "build_number": str(info["build_number"]),
        })
    return requests


def main():
    if os.environ["EVENT_NAME"] == "workflow_dispatch":
        kind = os.environ["KIND"]
        paths = list(MANIFESTS.values()) if kind == "all" else [MANIFESTS[kind]]
    else:
        paths = subprocess.check_output(["git", "diff", "--name-only", os.environ["BASE_SHA"], os.environ["HEAD_SHA"]],
                                        text=True).splitlines()
    requests = build_requests(paths)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("requests=" + json.dumps(requests) + "\n")


if __name__ == "__main__":
    main()
