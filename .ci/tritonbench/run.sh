#!/bin/bash
# Release testing entry point for the benchmark suites, mirroring
# .ci/release-testing/run.sh for the correctness suites.
#
# Usage:
#   run.sh <suite> <conda-env>
#
# Thin wrapper over TritonBench's own run-benchmark.sh (which lives in
# TRITONBENCH_ROOT, not in this repo): it runs <suite> under <conda-env> and
# stages the results as BENCHMARK_OUTPUT/<conda-env>, the layout the upload
# steps expect. TritonBench's .benchmarks scratch dir is cleared afterwards so
# that an A/B run does not pick up the previous side's results.
#
# Environment consumed:
#   TRITONBENCH_ROOT  TritonBench checkout. Defaults to /workspace/tritonbench.
#   BENCHMARK_OUTPUT  Directory results are staged in. Defaults to
#                     benchmark-output under GITHUB_WORKSPACE (cwd if unset).
set -euo pipefail

SUITE="${1:-}"
BENCHMARK_CONDA_ENV="${2:-}"
if [ -z "${SUITE}" ] || [ -z "${BENCHMARK_CONDA_ENV}" ]; then
  echo "Usage: $0 <suite> <conda-env>" >&2
  exit 1
fi

TRITONBENCH_ROOT="${TRITONBENCH_ROOT:-/workspace/tritonbench}"
if [ ! -d "${TRITONBENCH_ROOT}" ]; then
  echo "ERROR: TRITONBENCH_ROOT does not exist: ${TRITONBENCH_ROOT}" >&2
  exit 1
fi

BENCHMARK_OUTPUT="${BENCHMARK_OUTPUT:-${GITHUB_WORKSPACE:-$(pwd)}/benchmark-output}"

bash "${TRITONBENCH_ROOT}/.ci/tritonbench/run-benchmark.sh" "${SUITE}" \
  --conda-env "${BENCHMARK_CONDA_ENV}"

mkdir -p "${BENCHMARK_OUTPUT}"
cp -r "${TRITONBENCH_ROOT}/.benchmarks/${SUITE}" "${BENCHMARK_OUTPUT}/${BENCHMARK_CONDA_ENV}"
sudo rm -rf "${TRITONBENCH_ROOT}/.benchmarks" || true
