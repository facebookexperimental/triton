#!/usr/bin/env bash
# Hook this FBTriton checkout into an existing PyTorch virtual environment.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd -P)"
PY=""

usage() {
  echo "usage: $0 --python /path/to/pytorch/.venv/bin/python"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --python)
      PY="${2:?--python requires an interpreter}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$PY" ]; then
  echo "--python is required" >&2
  usage >&2
  exit 2
fi
if [ ! -x "$PY" ]; then
  echo "not an executable Python interpreter: $PY" >&2
  exit 2
fi
PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"

log() { echo "[torchtlx-setup] $*"; }

log "validating user-built PyTorch"
"$PY" - <<'PY'
import ast
import pathlib
import sys

import torch
from torch._inductor import config

if sys.prefix == sys.base_prefix:
    raise SystemExit("refusing to modify a non-virtualenv Python installation")
if not hasattr(config.triton, "tlx_mode"):
    raise SystemExit(
        "incompatible PyTorch: torch._inductor.config.triton.tlx_mode is missing"
    )

custom_op_path = (
    pathlib.Path(torch.__file__).resolve().parent
    / "_inductor"
    / "kernel"
    / "custom_op.py"
)
module = ast.parse(custom_op_path.read_text(), filename=str(custom_op_path))
registration = next(
    (
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "register_custom_op_autotuning"
    ),
    None,
)
if registration is None:
    raise SystemExit(
        "incompatible PyTorch: register_custom_op_autotuning is missing"
    )
parameters = {
    argument.arg
    for argument in (*registration.args.args, *registration.args.kwonlyargs)
}
if "include_fallback" not in parameters:
    raise SystemExit(
        "incompatible PyTorch: "
        "register_custom_op_autotuning(include_fallback=...) is required"
    )

print(f"torch={torch.__version__}")
print(f"torch_path={pathlib.Path(torch.__file__).resolve()}")
print(f"torch_git={torch.version.git_version}")
PY

if [ ! -f "$REPO/python/triton/_C/libtriton.so" ]; then
  echo "FBTriton is not built: $REPO/python/triton/_C/libtriton.so is missing" >&2
  echo "build FBTriton with $PY before running this hook" >&2
  exit 1
fi

# Resolve the environment's platform-independent library directory instead of
# assuming the first global site-packages entry is the virtual environment.
PTH_PATH="$($PY - <<'PY'
import pathlib
import sysconfig

print(pathlib.Path(sysconfig.get_path("purelib")) / "fbtriton_checkout.pth")
PY
)"

# Validate the complete integration before changing the environment. Explicitly
# prepending the checkout here prevents an older .pth hook from shadowing it.
if ! "$PY" - "$REPO" <<'PY'
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo / "python"))

import torch
import triton
from torch._inductor import config
from triton._C.libtriton import ir  # noqa: F401

triton_path = pathlib.Path(triton.__file__).resolve()
if repo not in triton_path.parents:
    raise SystemExit(f"FBTriton checkout is shadowed by {triton_path}")
if not hasattr(config.triton, "tlx_mode"):
    raise SystemExit(
        "incompatible PyTorch: torch._inductor.config.triton.tlx_mode is missing"
    )

import triton.language.extra.tlx.inductor.registry  # noqa: F401
PY
then
  echo "TorchTLX integration validation failed for $PY" >&2
  echo "ensure FBTriton is built with that same Python interpreter" >&2
  exit 1
fi

# Prepend this checkout in new Python processes. The atomic replacement leaves
# an existing hook intact if writing the new one fails.
"$PY" - "$REPO/python" "$PTH_PATH" <<'PY'
import os
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).resolve()
pth_path = pathlib.Path(sys.argv[2])
temporary = pth_path.with_name(f".{pth_path.name}.{os.getpid()}.tmp")
temporary.write_text(f"import sys; sys.path.insert(0, {str(source)!r})\n")
temporary.replace(pth_path)
PY
log "hooked FBTriton through $PTH_PATH"

"$PY" - <<PY
import pathlib

import torch
import triton
from torch._inductor import config

repo = pathlib.Path("$REPO").resolve()
triton_path = pathlib.Path(triton.__file__).resolve()
if repo not in triton_path.parents:
    raise SystemExit(f"FBTriton checkout is shadowed by {triton_path}")
if not hasattr(config.triton, "tlx_mode"):
    raise SystemExit("incompatible PyTorch: torch._inductor.config.triton.tlx_mode is missing")

import triton.language.extra.tlx.inductor.registry  # noqa: F401

print(f"triton_path={triton_path}")
print("TorchTLX registry=compatible")
PY

log "ready"
ACTIVE_PY="$(command -v python3 || true)"
if [ -z "$ACTIVE_PY" ] || [ "$(dirname "$ACTIVE_PY")" != "$(dirname "$PY")" ]; then
  log "the current shell still resolves python3 to ${ACTIVE_PY:-<not found>}"
  if [ -f "$(dirname "$PY")/activate" ]; then
    log "activate the configured environment before using python3:"
    log "source $(dirname "$PY")/activate"
  fi
fi
log "run directly:"
log "$PY python/test/torchtlx_benchmark/run_torchtlx_fusions.py --list"
