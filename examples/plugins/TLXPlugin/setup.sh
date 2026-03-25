#!/bin/bash
# TLXPlugin setup — symlink the Python DSL into triton.language.extra.tlx
#
# Run once from the triton repo root:
#   bash examples/plugins/TLXPlugin/setup.sh
#
# This creates a symlink so that:
#   import triton.language.extra.tlx as tlx
# resolves to examples/plugins/TLXPlugin/python/tlx/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET="$SCRIPT_DIR/python/tlx"
LINK="$TRITON_ROOT/python/triton/language/extra/tlx"

if [ -L "$LINK" ]; then
    echo "Symlink already exists: $LINK -> $(readlink "$LINK")"
    exit 0
fi

if [ -e "$LINK" ]; then
    echo "Error: $LINK already exists and is not a symlink. Remove it first."
    exit 1
fi

ln -s "$(realpath --relative-to="$(dirname "$LINK")" "$TARGET")" "$LINK"
echo "Created symlink: $LINK -> $(readlink "$LINK")"
