import json
from pathlib import Path
import tempfile
import unittest

from select_meta_llvm_builds import MANIFESTS, build_requests


class BuildRequestsTest(unittest.TestCase):

    def test_only_changed_build_requests_are_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind, path in MANIFESTS.items():
                manifest = root / path
                manifest.parent.mkdir(exist_ok=True)
                manifest.write_text(
                    json.dumps({
                        "repository": "facebookexperimental/llvm-project",
                        "llvm_hash": ("a" if kind == "core" else "b") * 40,
                        "build_number": 7,
                    }))
            core = {"kind": "core", "llvm_sha": "a" * 40, "build_number": "7"}
            amd = {"kind": "amd", "llvm_sha": "b" * 40, "build_number": "7"}
            before = {p: (root / p).read_bytes() for p in MANIFESTS.values()}
            self.assertEqual(build_requests([MANIFESTS["core"]], root), [core])
            self.assertEqual(build_requests([MANIFESTS["amd"]], root), [amd])
            self.assertEqual(build_requests(list(MANIFESTS.values()), root), [core, amd])
            self.assertEqual(build_requests(["cmake/meta-llvm-info.json", "cmake/meta-amd-llvm-info.json"], root), [])
            self.assertEqual({p: (root / p).read_bytes() for p in MANIFESTS.values()}, before)


if __name__ == "__main__":
    unittest.main()
