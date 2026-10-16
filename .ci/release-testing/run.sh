#!/bin/bash
# Release testing entry point: dispatch the per-hardware test suites.
#
# Usage:
#   run.sh <hardware> [<conda-env>]
#
# <hardware> is one of: h100, b200, mi350. It only selects which suites to run;
# the suite scripts themselves take no arguments and detect what they need
# (e.g. NVIDIA vs ROCm) from the environment.
#
# <conda-env> is the env to test in, exported as CONDA_ENV for SETUP_SCRIPT to
# pick up. Omit it to test whatever env is already active.
#
# Every selected suite runs even if an earlier one fails, so a single release
# run reports the complete picture instead of stopping at the first red suite.
# The exit code is non-zero if any suite failed.
#
# Set RELEASE_TESTING_TESTS to narrow a run, e.g.
#   RELEASE_TESTING_TESTS="tlx gluon" bash .ci/release-testing/run.sh b200

# Deliberately no `set -e`: individual suite failures are collected, not fatal.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "${SCRIPT_DIR}/common.sh"

HARDWARE="${1:-}"
if [ -z "${HARDWARE}" ]; then
  echo "Usage: $0 <hardware> [<conda-env>]   # hardware: ${RELEASE_TESTING_HARDWARES[*]}" >&2
  exit 1
fi
validate_hardware "${HARDWARE}" || exit 1

if [ -n "${2:-}" ]; then
  export CONDA_ENV="$2"
fi

# Suites to run per hardware. This mirrors the coverage of the nightly
# workflows (h100.yml, b200.yml, mi350.yml, torchtlx.yml): H100 has neither a
# Gluon nor a torchTLX job there, so the release gate does not add coverage
# that has never been vetted on that hardware.
#
# torchtlx is intentionally last: it reprovisions torch to a fresh nightly
# (see test-torchtlx.sh) and would otherwise change the environment out from
# under the other suites.
case "${HARDWARE}" in
  h100)
    SUITES=(tritonbench tlx)
    ;;
  b200)
    SUITES=(tritonbench tlx gluon torchtlx)
    ;;
  mi350)
    SUITES=(tritonbench tlx gluon torchtlx)
    ;;
esac

if [ -n "${RELEASE_TESTING_TESTS:-}" ]; then
  read -r -a SUITES <<< "${RELEASE_TESTING_TESTS}"
fi

echo "==> Release testing on ${HARDWARE}"
echo "    Conda env:      ${CONDA_ENV:-<current>}"
echo "    Triton source:  ${TRITON_SRC_DIR}"
echo "    TritonBench:    ${TRITONBENCH_ROOT}"
echo "    Results:        ${RELEASE_TESTING_OUTPUT}"
echo "    Suites:         ${SUITES[*]}"

FAILED_SUITES=()
PASSED_SUITES=()

for suite in "${SUITES[@]}"; do
  suite_script="${SCRIPT_DIR}/test-${suite}.sh"
  if [ ! -f "${suite_script}" ]; then
    echo "ERROR: no such suite '${suite}' (${suite_script} not found)" >&2
    FAILED_SUITES+=("${suite}")
    continue
  fi

  echo ""
  echo "=================================================================="
  echo "==> ${suite} (${HARDWARE})"
  echo "=================================================================="
  # Each suite runs in a subshell so that env mutations (conda activation,
  # exported knobs) do not leak between suites.
  if bash "${suite_script}"; then
    PASSED_SUITES+=("${suite}")
  else
    echo "==> ${suite} FAILED (exit $?)" >&2
    FAILED_SUITES+=("${suite}")
  fi
done

echo ""
echo "=================================================================="
echo "==> Release testing summary (${HARDWARE})"
echo "=================================================================="
for suite in "${PASSED_SUITES[@]:-}"; do
  [ -n "${suite}" ] && echo "  PASS  ${suite}"
done
for suite in "${FAILED_SUITES[@]:-}"; do
  [ -n "${suite}" ] && echo "  FAIL  ${suite}"
done

if [ "${#FAILED_SUITES[@]}" -gt 0 ]; then
  echo "==> ${#FAILED_SUITES[@]} suite(s) failed: ${FAILED_SUITES[*]}" >&2
  exit 1
fi

echo "==> All suites passed."
