"""Install the pinned OSS benchmark and its pending consolidation patch."""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
REVISION = "6394394849c41c23699f757b4cc2712d35ad4d86"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path,
                        help="Use a local tritonbench checkout containing the consolidation changes")
    args = parser.parse_args()
    source = str(
        args.source.resolve()) if args.source else f"git+https://github.com/meta-pytorch/tritonbench@{REVISION}"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", source], check=True)
    spec = importlib.util.find_spec("tritonbench")
    if spec is None or spec.origin is None:
        raise RuntimeError("tritonbench install did not provide its package")
    package_root = pathlib.Path(spec.origin).parent.parent
    if args.source is None:
        with (ROOT / "tritonbench.patch").open() as patch:
            subprocess.run(["patch", "--batch", "--forward", "-p1"], cwd=package_root, stdin=patch, check=True)

    metadata = {
        "revision": REVISION if args.source is None else "local", "patch_sha256": hashlib.sha256(
            (ROOT / "tritonbench.patch").read_bytes()).hexdigest(), "source":
        str(args.source.resolve()) if args.source else "OSS"
    }
    (package_root / "tritonbench" / "tlx_l2_install.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()
