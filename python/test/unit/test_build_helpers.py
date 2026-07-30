import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python import build_helpers


class _Response(io.BytesIO):
    headers: dict[str, str] = {}


class DownloadAndExtractTest(unittest.TestCase):

    def setUp(self) -> None:
        archive = io.BytesIO()
        payload = b"verified payload"
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            info = tarfile.TarInfo("payload.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        self.archive = archive.getvalue()

    def _open_url(self, _url: str) -> _Response:
        return _Response(self.archive)

    def test_matching_checksum_extracts_archive(self) -> None:
        expected_sha256 = hashlib.sha256(self.archive).hexdigest()
        with tempfile.TemporaryDirectory() as output_dir, patch.object(build_helpers, "open_url",
                                                                       side_effect=self._open_url):
            build_helpers._download_and_extract(
                "https://example.com/llvm.tar.gz",
                output_dir,
                "LLVM",
                expected_sha256,
            )

            self.assertEqual(
                b"verified payload",
                (Path(output_dir) / "payload.txt").read_bytes(),
            )

    def test_mismatched_checksum_rejects_archive(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir, patch.object(build_helpers, "open_url",
                                                                       side_effect=self._open_url), patch.dict(
                                                                           os.environ,
                                                                           {"TRITON_UNSAFE_DISABLE_SHA_CHECK": ""}):
            with self.assertRaisesRegex(RuntimeError, "failed checksum validation"):
                build_helpers._download_and_extract(
                    "https://example.com/llvm.tar.gz",
                    output_dir,
                    "LLVM",
                    "0" * 64,
                )

            self.assertFalse((Path(output_dir) / "payload.txt").exists())

    def test_unsafe_override_extracts_mismatched_archive(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as output_dir, patch.object(
                build_helpers, "open_url", side_effect=self._open_url), patch.dict(
                    os.environ, {"TRITON_UNSAFE_DISABLE_SHA_CHECK": "1"}), patch("sys.stderr", stderr):
            build_helpers._download_and_extract(
                "https://example.com/llvm.tar.gz",
                output_dir,
                "LLVM",
                "0" * 64,
            )

            self.assertTrue((Path(output_dir) / "payload.txt").exists())
            self.assertIn("WARNING:", stderr.getvalue())

    def test_llvm_package_uses_platform_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as base_dir:
            cmake_dir = Path(base_dir) / "cmake"
            cmake_dir.mkdir()
            (cmake_dir / "llvm-info.json").write_text(
                json.dumps({
                    "llvm_hash": "abcdef0123456789",
                    "build_number": 7,
                    "sha256sum": {"test-platform": "expected-checksum"},
                }))
            helper_args = build_helpers.BuildHelperArgs(
                cache_path=base_dir,
                offline_build=False,
                llvm_system_suffix="test-platform",
                llvm_syspath=None,
                json_syspath=None,
                ptxas_path=None,
                ptxas_blackwell_path=None,
                cuobjdump_path=None,
                nvdisasm_path=None,
                cudacrt_path=None,
                cudart_path=None,
                cupti_include_path=None,
                cupti_lib_path=None,
                cupti_lib_blackwell_path=None,
            )
            with patch.object(build_helpers, "get_base_dir", return_value=base_dir):
                package = build_helpers.get_llvm_package_info(helper_args)

            self.assertEqual("llvm-abcdef01-test-platform-7", package.name)
            self.assertEqual("expected-checksum", package.sha256sum)
