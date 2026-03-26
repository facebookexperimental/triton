#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

TRITON_PASS_PLUGIN_PATH="$REPO_ROOT/python/triton/plugins/libTLXMemOpsPlugin.so" \
    python "$SCRIPT_DIR/amd-gemm-pipelined.py" "$@"
