#!/bin/bash
# Optionally check out a TritonBench PR referenced from a meta-triton PR body.
#
# Usage:
#   checkout.sh <pr-body-file> [--ref-only]
#
# The PR body is scanned for a line of the form:
#   X-link: https://github.com/meta-pytorch/tritonbench/pull/<PR-NUMBER>
#
# When such a link is found:
#   * The corresponding PR number and git ref are written to GITHUB_OUTPUT
#     (pr_number / ref) so an actions/checkout step can consume `ref`.
#   * Unless --ref-only is given, TritonBench is cloned into the `tritonbench/`
#     subdirectory at that PR and TRITONBENCH_ROOT is exported via GITHUB_ENV
#     for subsequent steps.
#
# When no link is found the script is a no-op (exit 0), leaving callers to fall
# back to their default TRITONBENCH_ROOT.
set -euo pipefail

PR_BODY_FILE="${1:-}"
REF_ONLY=0
if [ "${2:-}" = "--ref-only" ]; then
  REF_ONLY=1
fi

if [ -z "${PR_BODY_FILE}" ] || [ ! -f "${PR_BODY_FILE}" ]; then
  echo "Usage: $0 <pr-body-file> [--ref-only]" >&2
  exit 1
fi

# Match: X-link: https://github.com/meta-pytorch/tritonbench/pull/<PR-NUMBER>
PR_NUMBER=$(grep -Eo \
  'X-link:[[:space:]]*https://github\.com/meta-pytorch/tritonbench/pull/[0-9]+' \
  "${PR_BODY_FILE}" | grep -Eo '[0-9]+$' | head -n 1 || true)

if [ -z "${PR_NUMBER}" ]; then
  echo "No TritonBench X-link found in PR body; keeping default TRITONBENCH_ROOT."
  exit 0
fi

TRITONBENCH_REF="refs/pull/${PR_NUMBER}/head"
echo "Found TritonBench PR #${PR_NUMBER} (ref: ${TRITONBENCH_REF})."

# Expose the PR number and ref for actions/checkout-based consumers.
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "pr_number=${PR_NUMBER}"
    echo "ref=${TRITONBENCH_REF}"
  } >> "${GITHUB_OUTPUT}"
fi

if [ "${REF_ONLY}" -eq 1 ]; then
  exit 0
fi

TRITONBENCH_DIR="${GITHUB_WORKSPACE:-$(pwd)}/tritonbench"
echo "Cloning TritonBench PR #${PR_NUMBER} into ${TRITONBENCH_DIR}"
rm -rf "${TRITONBENCH_DIR}"
git clone https://github.com/meta-pytorch/tritonbench.git "${TRITONBENCH_DIR}"
git -C "${TRITONBENCH_DIR}" fetch origin "${TRITONBENCH_REF}:pr-${PR_NUMBER}"
git -C "${TRITONBENCH_DIR}" checkout "pr-${PR_NUMBER}"
git -C "${TRITONBENCH_DIR}" submodule update --init --recursive

# Export for subsequent workflow steps.
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "TRITONBENCH_ROOT=${TRITONBENCH_DIR}" >> "${GITHUB_ENV}"
fi
echo "TRITONBENCH_ROOT set to ${TRITONBENCH_DIR}"
