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
        with tempfile.TemporaryDirectory() as base_dir, patch.object(build_helpers, "open_url",
                                                                     side_effect=self._open_url), patch.object(
                                                                         build_helpers.shutil, "which",
                                                                         return_value=None):
            output_dir = Path(base_dir) / "output"
            archives_dir = Path(base_dir) / "archives"
            build_helpers._download_and_extract(
                "https://example.com/llvm.tar.gz",
                output_dir,
                "LLVM",
                archives_dir,
                expected_sha256,
            )

            self.assertEqual(
                b"verified payload",
                (Path(output_dir) / "payload.txt").read_bytes(),
            )
            self.assertFalse((archives_dir / "llvm.tar.gz").exists())

    def test_mismatched_checksum_preserves_destination_and_deletes_archive(self) -> None:
        with tempfile.TemporaryDirectory() as base_dir, patch.object(
                build_helpers, "open_url", side_effect=self._open_url), patch.object(
                    build_helpers.shutil, "which",
                    return_value=None), patch.dict(os.environ, {"TRITON_UNSAFE_DISABLE_SHA_CHECK": ""}):
            output_dir = Path(base_dir) / "output"
            output_dir.mkdir()
            existing_file = output_dir / "existing.txt"
            existing_file.write_bytes(b"existing payload")
            archives_dir = Path(base_dir) / "archives"

            with self.assertRaisesRegex(RuntimeError, "failed checksum validation"):
                build_helpers._download_and_extract(
                    "https://example.com/llvm.tar.gz",
                    output_dir,
                    "LLVM",
                    archives_dir,
                    "0" * 64,
                )

            self.assertEqual(b"existing payload", existing_file.read_bytes())
            self.assertFalse((archives_dir / "llvm.tar.gz").exists())

    def test_unsafe_override_extracts_mismatched_archive(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as base_dir, patch.object(
                build_helpers, "open_url",
                side_effect=self._open_url), patch.object(build_helpers.shutil, "which", return_value=None), patch.dict(
                    os.environ, {"TRITON_UNSAFE_DISABLE_SHA_CHECK": "1"}), patch("sys.stderr", stderr):
            output_dir = Path(base_dir) / "output"
            archives_dir = Path(base_dir) / "archives"
            build_helpers._download_and_extract(
                "https://example.com/llvm.tar.gz",
                output_dir,
                "LLVM",
                archives_dir,
                "0" * 64,
            )

            self.assertTrue((Path(output_dir) / "payload.txt").exists())
            self.assertFalse((archives_dir / "llvm.tar.gz").exists())
            self.assertIn("WARNING:", stderr.getvalue())

    def test_curl_resumes_azure_download_with_range_capable_api(self) -> None:
        with patch.object(build_helpers.subprocess, "run") as run:
            build_helpers._download_file_with_curl(
                "/usr/bin/curl",
                "https://example.blob.core.windows.net/container/llvm.tar.gz",
                "/cache/archives/llvm.tar.gz",
                "downloading LLVM",
            )

        command = run.call_args.args[0]
        self.assertIn("--continue-at", command)
        self.assertEqual("-", command[command.index("--continue-at") + 1])
        self.assertIn("x-ms-version: 2011-08-18", command)
        self.assertEqual("/cache/archives/llvm.tar.gz", command[command.index("--output") + 1])

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


class MetaLlvmInfoTest(unittest.TestCase):
    OAI_HASH = "5f07f818b51b786b0a87b4e514882600ecba112f"
    META = {
        "based_on_oai_llvm_hash": OAI_HASH,
        "llvm_hash": "1889ed344a2ea019320c1a49fc5b7103d0d9468d",
        "llvm_repository": "https://github.com/facebookexperimental/llvm-project",
        "build_number": 2,
        "release_base_url": "https://github.com/facebookexperimental/llvm-project/releases/download/llvm-1889ed34-2",
        "sha256sum": {"ubuntu-x64": "meta-ubuntu-x64"},
    }

    def _write(self, base_dir: str, meta: dict) -> None:
        cmake_dir = Path(base_dir) / "cmake"
        cmake_dir.mkdir()
        (cmake_dir / "llvm-info.json").write_text(
            json.dumps({"llvm_hash": self.OAI_HASH, "build_number": 1, "sha256sum": {"ubuntu-x64": "oai-ubuntu-x64"}}))
        (cmake_dir / "meta-llvm-info.json").write_text(json.dumps(meta))

    def test_meta_overrides_hash_url_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as base_dir:
            self._write(base_dir, self.META)
            package = build_helpers.llvm_package_for_suffix("ubuntu-x64", base_dir=base_dir)
        self.assertEqual("llvm-1889ed34-ubuntu-x64-2", package.name)
        self.assertEqual(
            "https://github.com/facebookexperimental/llvm-project/releases/download/llvm-1889ed34-2/"
            "llvm-1889ed34-ubuntu-x64-2.tar.gz", package.url)
        self.assertEqual("meta-ubuntu-x64", package.sha256sum)

    def test_meta_base_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as base_dir:
            self._write(base_dir, {**self.META, "based_on_oai_llvm_hash": "0" * 40})
            with self.assertRaisesRegex(RuntimeError, "based on OAI LLVM 000000000000"):
                build_helpers.llvm_package_for_suffix("ubuntu-x64", base_dir=base_dir)

    def test_unpublished_platform_ok_with_llvm_syspath(self) -> None:
        # A local LLVM build must keep working on platforms with no Meta bundle.
        with tempfile.TemporaryDirectory() as base_dir:
            self._write(base_dir, self.META)
            helper_args = build_helpers.BuildHelperArgs(
                cache_path=base_dir,
                offline_build=False,
                llvm_system_suffix="macos-arm64",
                llvm_syspath="/opt/llvm",
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
        self.assertEqual("", package.url)
