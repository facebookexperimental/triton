#!/bin/bash
# Shared helpers for the release-testing scripts in this directory.
#
# Sourced (not executed) by run.sh and by each test-*.sh. Every test script is
# also runnable standalone, e.g.:
#
#   bash .ci/release-testing/test-tlx.sh b200
#
# Environment consumed:
#   SETUP_SCRIPT            Runner env activation script (e.g. /workspace/setup_instance.sh).
#                           Optional: a local run is assumed to be pre-activated.
#   TRITON_SRC_DIR          Triton source checkout under test. Defaults to the repo
#                           this script lives in; CI points it at the exact tree
#                           that was built and installed.
#   TRITONBENCH_ROOT        TritonBench checkout. Defaults to /workspace/tritonbench.
#   RELEASE_TESTING_OUTPUT  Directory for junit XML results.

# Absolute path of .ci/release-testing.
RELEASE_TESTING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${TRITON_SRC_DIR:-}" ]; then
  TRITON_SRC_DIR="$(cd "${RELEASE_TESTING_DIR}/../.." && pwd)"
fi
export TRITON_SRC_DIR

export TRITONBENCH_ROOT="${TRITONBENCH_ROOT:-/workspace/tritonbench}"

RELEASE_TESTING_OUTPUT="${RELEASE_TESTING_OUTPUT:-${GITHUB_WORKSPACE:-/tmp}/release-testing-output}"
export RELEASE_TESTING_OUTPUT
mkdir -p "${RELEASE_TESTING_OUTPUT}"

# Valid values for the <hardware> argument.
RELEASE_TESTING_HARDWARES=(h100 b200 mi350)

# validate_hardware <hardware>
# Return non-zero with a usage message if <hardware> is not recognized.
validate_hardware() {
  local hardware="${1:-}"
  local known
  for known in "${RELEASE_TESTING_HARDWARES[@]}"; do
    if [ "${hardware}" = "${known}" ]; then
      return 0
    fi
  done
  echo "ERROR: unknown hardware '${hardware}'. Expected one of: ${RELEASE_TESTING_HARDWARES[*]}" >&2
  return 1
}

# activate_env
# Source the runner environment activation script when one is configured.
activate_env() {
  if [ -n "${SETUP_SCRIPT:-}" ] && [ -f "${SETUP_SCRIPT}" ]; then
    # shellcheck disable=SC1090
    . "${SETUP_SCRIPT}"
  else
    echo "SETUP_SCRIPT not set or not found; using the current environment."
  fi
}

# pip_install <args...>
# Install with uv when available (the CI runners provision it), else plain pip.
pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install "$@"
  else
    python -m pip install "$@"
  fi
}

# print_versions
# Log the torch/triton actually under test, to make release runs triageable.
print_versions() {
  python -c "import torch; print('torch version: ', torch.__version__); print('torch location: ', torch.__file__)" || true
  python -c "import triton; print('triton version: ', triton.__version__); print('triton location: ', triton.__file__)" || true
}

# disable_torchtlx
# torchTLX (TLX as an Inductor backend) is covered only by test-torchtlx.sh,
# which provisions its own nightly torch. Everywhere else force it off: only
# "allow"/"force" enable it, so an empty value overrides any environment default.
disable_torchtlx() {
  export TORCHINDUCTOR_TLX_MODE=""
}
