#!/bin/bash
# Release testing: Gluon unit tests.
#
# Usage:
#   test-gluon.sh <hardware>
#
# Mirrors the "-gluon-test" jobs of b200.yml / mi350.yml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "${SCRIPT_DIR}/common.sh"

HARDWARE="${1:-}"
validate_hardware "${HARDWARE}"

activate_env
disable_torchtlx

pip_install pytest expecttest
print_versions

cd "${TRITON_SRC_DIR}"

# Full Gluon unit-test coverage: every python/test/gluon/test_*.py. This spans
# the compile-only frontend tests plus the GPU-execution suites (core, lowerings,
# consan, fpsan, layout_format_view). Known-failing cases and their root causes
# are tracked in README.md ("Gluon support" -> TODO gluon-ci).
python -m pytest python/test/gluon/ \
  -v --junitxml="${RELEASE_TESTING_OUTPUT}/gluon.xml"
