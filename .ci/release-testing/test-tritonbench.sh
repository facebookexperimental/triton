#!/bin/bash
# Release testing: TritonBench GPU tests.
#
# Usage:
#   test-tritonbench.sh <hardware> [extra unittest args...]
#
# Mirrors the "Run TritonBench tests" step of h100.yml / b200.yml / mi350.yml:
# it delegates to .ci/tritonbench/test-gpu.sh with the fork's skip list applied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "${SCRIPT_DIR}/common.sh"

HARDWARE="${1:-}"
validate_hardware "${HARDWARE}"
shift

activate_env
disable_torchtlx
print_versions

# Known-failing / unsupported cases for the fork are listed here rather than
# skipped upstream.
export TRITONBENCH_TEST_SKIP_FILE="${TRITONBENCH_TEST_SKIP_FILE:-${TRITON_SRC_DIR}/.ci/tritonbench/fbtriton_skip_tests.yaml}"

if [ ! -d "${TRITONBENCH_ROOT}" ]; then
  echo "ERROR: TRITONBENCH_ROOT does not exist: ${TRITONBENCH_ROOT}" >&2
  exit 1
fi

cd "${TRITON_SRC_DIR}"
bash ./.ci/tritonbench/test-gpu.sh "$@"
