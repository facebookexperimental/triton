#!/bin/bash
set -euo pipefail

if [ -z "${TRITONBENCH_ROOT:-}" ]; then
  # TRITONBENCH_ROOT not set: guess it lives next to the repo dir, i.e. in
  # the parent dir of the "meta-triton" checkout (<...>/tritonbench).
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  GUESSED_ROOT="$(dirname "${REPO_ROOT}")/tritonbench"
  if [ -d "${GUESSED_ROOT}" ]; then
    echo "TRITONBENCH_ROOT not set, using guessed location: ${GUESSED_ROOT}"
    export TRITONBENCH_ROOT="${GUESSED_ROOT}"
  else
    echo "ERROR: TRITONBENCH_ROOT is not set and could not be found at ${GUESSED_ROOT}"
    exit 1
  fi
fi

# print pytorch and triton versions for debugging
python -c "import torch; print('torch version: ', torch.__version__); print('torch location: ', torch.__file__)"
python -c "import triton; print('triton version: ', triton.__version__); print('triton location: ', triton.__file__)"

# workaround: disable inductor subprocess compilation to avoid
# "Could not find an active GPU backend" in subprocess workers
export TORCHINDUCTOR_COMPILE_THREADS=1
# best effort disable autotune to speedup test
export TORCHINDUCTOR_MAX_AUTOTUNE=0

cd ${TRITONBENCH_ROOT}
python -m unittest test.test_gpu.main -v $@
cd -
