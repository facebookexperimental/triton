"""Pin the FA4 source tree — bindings are valid only against this exact code.

The fingerprint covers every .py under flash_attn/cute/ (the whole runtime
surface, not a hand-picked file list — drift in any unpinned helper could
change behavior under the shims).  verify_pin() fingerprints the directory
the running process actually imports, so a venv/clone mismatch fails loudly.
"""

import hashlib
import json
from pathlib import Path

PINNED_COMMIT = "2409214a03797b168f648ea30df1adbc09ce658a"
PIN_FILE = Path(__file__).resolve().parent / "fa4_pin.json"


def fingerprint_tree(cute_dir) -> dict:
    cute_dir = Path(cute_dir)
    files = sorted(p.relative_to(cute_dir).as_posix()
                   for p in cute_dir.rglob("*.py")
                   if "__pycache__" not in p.parts and "egg-info" not in p.parts)
    tree = hashlib.sha256()
    per_file = {}
    for rel in files:
        h = hashlib.sha256((cute_dir / rel).read_bytes()).hexdigest()
        per_file[rel] = h
        tree.update(rel.encode())
        tree.update(h.encode())
    return {"commit": PINNED_COMMIT, "tree_sha256": tree.hexdigest(),
            "num_files": len(files), "files": per_file}


def write_pin(cute_dir):
    PIN_FILE.write_text(json.dumps(fingerprint_tree(cute_dir), indent=1))
    return PIN_FILE


def verify_pin(cute_dir=None) -> dict:
    """Verify the imported (or given) flash_attn/cute tree matches the pin.

    Raises RuntimeError on any mismatch — a binding must never run against
    drifted sources.
    """
    if cute_dir is None:
        import flash_attn.cute as fc
        cute_dir = Path(fc.__file__).parent
    expected = json.loads(PIN_FILE.read_text())
    actual = fingerprint_tree(cute_dir)
    if actual["tree_sha256"] != expected["tree_sha256"]:
        changed = [f for f in expected["files"]
                   if actual["files"].get(f) != expected["files"][f]]
        missing = [f for f in expected["files"] if f not in actual["files"]]
        added = [f for f in actual["files"] if f not in expected["files"]]
        raise RuntimeError(
            f"FA4 source drift at {cute_dir}: refusing to bind. "
            f"changed={changed[:5]} missing={missing[:3]} added={added[:3]} "
            f"(pinned commit {expected['commit']})")
    return expected
