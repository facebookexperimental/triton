"""CPU tests for the magnon ACF store layout lookup."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from triton.magnon import store
except ImportError:  # buck python_unittest, which ships store.py without the triton package
    from python.triton.magnon import store

SHA = "ab" * 32
ARCH = "sm_100a"


def _write(path: Path, data: bytes = b"acf", ir_hash: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if ir_hash is not None:
        path.with_name(path.name + ".json").write_text(json.dumps({"ir_hash": ir_hash}))
    return path


class LayoutTest(unittest.TestCase):

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        env = patch.dict(os.environ, {"COMPILE_IQ_STORE": str(self.root)})
        env.start()
        self.addCleanup(env.stop)

    def test_flat(self) -> None:
        p = _write(self.root / ARCH / f"{SHA}.13.3.acf")
        self.assertEqual(store.acf_path(SHA, ARCH, "13.3"), str(p))
        self.assertEqual(store.read_acf(SHA, ARCH, "13.3"), b"acf")

    def test_nested_short_and_full_names_win_over_flat(self) -> None:
        _write(self.root / ARCH / f"{SHA}.13.3.acf", b"flat")
        full = _write(self.root / ARCH / SHA[:16] / f"{SHA}.13.3.acf", b"full")
        self.assertEqual(store.acf_path(SHA, ARCH, "13.3"), str(full))
        short = _write(self.root / ARCH / SHA[:16] / f"{SHA[:16]}.13.3.acf", b"short")
        self.assertEqual(store.acf_path(SHA, ARCH, "13.3"), str(short))
        self.assertEqual(store.acf_cubin_path(SHA, ARCH, "13.3"), str(short) + ".cubin")

    def test_nested_short_name_needs_matching_sidecar(self) -> None:
        flat = _write(self.root / ARCH / f"{SHA}.13.3.acf", b"flat")
        _write(self.root / ARCH / SHA[:16] / f"{SHA[:16]}.13.3.acf", b"other", ir_hash=SHA[:16] + "cd" * 24)
        self.assertEqual(store.acf_path(SHA, ARCH, "13.3"), str(flat))

    def test_miss_returns_flat_path(self) -> None:
        self.assertEqual(store.acf_path(SHA, ARCH, "13.3"), str(self.root / ARCH / f"{SHA}.13.3.acf"))
        self.assertIsNone(store.read_acf(SHA, ARCH, "13.3"))


if __name__ == "__main__":
    unittest.main()
