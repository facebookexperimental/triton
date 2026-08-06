#!/bin/bash
# Release testing: torchTLX (TLX as an Inductor backend) template + fusion tests.
#
# Usage:
#   test-torchtlx.sh <hardware>
#
# Mirrors torchtlx.yml. torchTLX plugs into the newest Inductor internals
# (config.triton.tlx_mode, template/fusion hooks), so unlike every other suite
# here it needs a *fresh nightly* PyTorch rather than the pre-provisioned one,
# with the fork Triton reinstalled on top.
#
# WARNING: that reprovisioning mutates the shared environment. run.sh therefore
# schedules this suite last. Set RELEASE_TESTING_TORCHTLX_REFRESH_TORCH=0 to
# test against whatever torch is already installed instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "${SCRIPT_DIR}/common.sh"

HARDWARE="${1:-}"
validate_hardware "${HARDWARE}"

activate_env

# Unlike the other suites, do NOT force TORCHINDUCTOR_TLX_MODE off here: this is
# the suite that exercises the torchTLX backend.

REFRESH_TORCH="${RELEASE_TESTING_TORCHTLX_REFRESH_TORCH:-1}"

if [ "${REFRESH_TORCH}" = "1" ]; then
  if [ "${HARDWARE}" = "mi350" ]; then
    # Omitting --install-torch-wheel makes setup-env.sh fall back to the latest
    # (floating) rocm nightly torch rather than the pinned wheel the other AMD
    # jobs use; --custom-triton then installs the fork Triton on top.
    if [ ! -d "${TRITONBENCH_ROOT}" ]; then
      echo "ERROR: TRITONBENCH_ROOT does not exist: ${TRITONBENCH_ROOT}" >&2
      exit 1
    fi
    (cd "${TRITONBENCH_ROOT}" && bash ./.ci/tritonbench/setup-env.sh --hip --custom-triton "${TRITON_SRC_DIR}")
    activate_env
  else
    # Drop any pre-provisioned torch first, then install the latest --pre
    # nightly (--upgrade defeats a cached-version match). It pulls its own
    # pytorch-triton, which the fork Triton install below overrides.
    if command -v uv >/dev/null 2>&1; then
      uv pip uninstall torch pytorch-triton || true
    else
      python -m pip uninstall -y torch pytorch-triton || true
    fi
    pip_install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu128

    TRITON_INSTALL_UTILS="${TRITONBENCH_ROOT}/.ci/triton/triton_install_utils.sh"
    if [ ! -f "${TRITON_INSTALL_UTILS}" ]; then
      echo "ERROR: cannot reinstall the fork Triton, missing ${TRITON_INSTALL_UTILS}" >&2
      exit 1
    fi
    # shellcheck disable=SC1090
    . "${TRITON_INSTALL_UTILS}"
    install_triton "${TRITON_SRC_DIR}"
  fi
else
  echo "RELEASE_TESTING_TORCHTLX_REFRESH_TORCH=0: keeping the existing torch install."
fi

pip_install pytest expecttest
print_versions

cd "${TRITON_SRC_DIR}"

# On AMD, gfx950 addmm warp-pipe cases run; Blackwell cases auto-skip via arch
# gates (and vice versa on NVIDIA).
python -m pytest \
  python/test/unit/language/test_torchtlx_templates.py \
  python/test/unit/language/test_torchtlx_fusions.py \
  -v --junitxml="${RELEASE_TESTING_OUTPUT}/torchtlx.xml"
