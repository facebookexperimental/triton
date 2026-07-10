#!/usr/bin/env python3
"""Classify a nightly failure summary as an external-dependency error or not.

Nightly issue-filing runs auto-bisection to blame the PR that introduced a
regression. That only makes sense when the failure is caused by a code change in
*this* repo. Failures caused by the environment -- a busy/unhealthy GPU, a driver
mismatch, a broken LLVM/toolchain download, a network/proxy hiccup, a full disk --
are NOT anyone's PR, so we skip bisection and annotate the issue instead.

The classifier is intentionally conservative: it only flags a failure as external
when a known infra signature matches, and it explicitly does NOT flag
compiler/runtime resource errors (e.g. ``OutOfResources: shared memory``) or
assertion failures, which are real regressions that bisection should chase.

Public API:
    classify(summary) -> (is_external: bool, reason: str)
"""

import re

# Ordered (category, human-reason, compiled-pattern) signatures. First match wins.
# Patterns run case-insensitively against the failure summary (first error line).
_SIGNATURES = [
    # --- GPU / driver / runner-hardware infra ---
    ("gpu-infra", "GPU busy or unavailable (runner infra)",
     r"all CUDA-capable devices are busy or unavailable"),
    ("gpu-infra", "No CUDA-capable device on the runner (runner infra)",
     r"no CUDA-capable device (is|are) detected|No CUDA GPUs are available"),
    ("gpu-infra", "CUDA driver too old for the toolkit (runner infra)",
     r"CUDA driver version is insufficient"),
    ("gpu-infra", "GPU/HIP device error (runner infra)",
     r"HIP error: (invalid device|no ROCm-capable device)"),
    ("gpu-infra", "NVML/driver-library mismatch (runner infra)",
     r"Failed to initialize NVML|Driver/library version mismatch"),

    # --- toolchain / dependency build & download ---
    ("toolchain", "LLVM build/download failed (external toolchain)",
     r"llvm.*(download|build) failed|Failed to download LLVM|could not.*llvm-install"),
    ("toolchain", "ptxas/toolchain binary error (external toolchain)",
     r"ptxas.*(not found|command not found|No such file)"),

    # --- package / network / proxy ---
    ("network", "Proxy authentication/blocked (network infra)",
     r"407 Proxy Authentication Required|\b403 Forbidden\b.*proxy|fwdproxy"),
    ("network", "DNS/connection failure fetching a dependency (network infra)",
     r"Could not resolve host|Temporary failure in name resolution|"
     r"Connection timed out|Failed to establish a new connection"),
    ("network", "pip/uv/conda could not install a dependency (network infra)",
     r"(pip|uv|conda).*(Could not find a version|No matching distribution|"
     r"ResolutionImpossible|HTTPError|Read timed out)"),
    ("network", "git clone/fetch of a dependency failed (network infra)",
     r"fatal: unable to access|fatal: could not read from remote|"
     r"RPC failed; curl|early EOF"),

    # --- runner disk / OS ---
    ("runner", "Runner out of disk space (runner infra)",
     r"No space left on device"),
    ("runner", "Runner out of host memory / OOM-killed (runner infra)",
     r"Cannot allocate memory|Out of memory: Killed process|oom-kill"),
]

_COMPILED = [(cat, reason, re.compile(pat, re.IGNORECASE)) for cat, reason, pat in _SIGNATURES]

# Signatures that look infra-ish but are REAL regressions bisection must chase.
# If any of these match, we never classify as external, even if a broad infra
# pattern above would otherwise fire.
_NOT_EXTERNAL = re.compile(
    r"OutOfResources|shared memory|Required:\s*\d+|"
    r"AssertionError|assert_close|Tensor-likes are not close|"
    r"failed to legalize|\berror: |LLVM ERROR",
    re.IGNORECASE,
)


def classify(summary):
    """Return (is_external, reason) for a failure summary line.

    A non-external failure returns (False, "").
    """
    text = (summary or "").strip()
    if not text:
        return False, ""
    if _NOT_EXTERNAL.search(text):
        # A concrete correctness/compiler error dominates: chase it, don't excuse it.
        return False, ""
    for _cat, reason, pat in _COMPILED:
        if pat.search(text):
            return True, reason
    return False, ""


if __name__ == "__main__":
    import sys
    is_ext, why = classify(" ".join(sys.argv[1:]))
    print(f"external={is_ext} reason={why}")
