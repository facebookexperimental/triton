#!/bin/bash
# Release testing: TLX unit tests + TLX tutorial correctness tests.
#
# Usage:
#   test-tlx.sh
#
# Mirrors the "-tlx-test" jobs of h100.yml / b200.yml / mi350.yml. Both pytest
# invocations always run; the script exits non-zero if either failed, so a
# broken unit test does not hide a broken tutorial (or vice versa).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "${SCRIPT_DIR}/common.sh"

activate_env
disable_torchtlx

set -e
pip_install pytest

# torchao supplies the MX-format reference implementations used by the
# mxfp TLX tests. AMD installs the rocm nightly wheel; NVIDIA builds from
# source because there is no matching prebuilt nightly.
if is_amd; then
  pip_install --pre torchao --index-url https://download.pytorch.org/whl/nightly/rocm7.2
else
  pip_install --no-build-isolation --no-deps "git+https://github.com/pytorch/ao.git"
fi
python -c 'from torchao.prototype.mx_formats.mx_tensor import MXTensor, ScaleCalculationMode'
set +e

print_versions
cd "${TRITON_SRC_DIR}"

STATUS=0

# TLX-only: the test_tlx_*.py glob covers manual TLX (tlx.async_tasks) and
# structurally excludes the autoWS suites (test_autows_*.py,
# test_warp_specialization.py, test_tutorial09_warp_specialization.py), which
# are never collected here and are tested separately.
echo "==> TLX unit tests"
python -m pytest python/test/unit/language/test_tlx_*.py \
  -v --junitxml="${RELEASE_TESTING_OUTPUT}/tlx-unit.xml" || STATUS=1

# On AMD, gfx950/CDNA4 + ikbo cases run; Hopper/Blackwell and gfx1250 cases
# auto-skip via @pytest.mark.skipif arch gates (and vice versa on NVIDIA).
echo "==> TLX tutorial correctness tests"
python -m pytest third_party/tlx/tutorials/testing/test_correctness.py \
  -v --junitxml="${RELEASE_TESTING_OUTPUT}/tlx-correctness.xml" || STATUS=1

exit "${STATUS}"
