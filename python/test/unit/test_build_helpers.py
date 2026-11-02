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


def _helper_args(cache_path: str, **overrides) -> build_helpers.BuildHelperArgs:
    fields = {name: None for name in build_helpers.BuildHelperArgs.__dataclass_fields__}
    fields.update(cache_path=cache_path, offline_build=False)
    fields.update(overrides)
    return build_helpers.BuildHelperArgs(**fields)


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
            helper_args = _helper_args(base_dir, llvm_system_suffix="test-platform")
            with patch.object(build_helpers, "get_base_dir", return_value=base_dir):
                package = build_helpers.get_llvm_package_info(helper_args)

            self.assertEqual("llvm-abcdef01-test-platform-7", package.name)
            self.assertEqual("expected-checksum", package.sha256sum)


class MetaLlvmInfoTest(unittest.TestCase):
    OAI_INFO_HASH = "ce3529423abda3fc4ad0b542daa50af21e539a29"
    OAI_BUILD_HASH = "1df311a6f7d8af05d068ce835b60dfa5b199d2f8"
    META = {
        "based_on_oai_llvm_hash": OAI_INFO_HASH,
        "llvm_hash": "a6533ab0f5a1df16ff3dbf7358fb79097c60855e",
        "llvm_repository": "https://github.com/facebookexperimental/llvm-project",
        "build_number": 2,
        "release_base_url": "https://github.com/facebookexperimental/llvm-project/releases/download/llvm-a6533ab0-2",
        "sha256sum": {"ubuntu-x64": "meta-ubuntu-x64"},
    }

    def _write(self, base_dir: str, meta: dict, build_info: bool = True) -> None:
        cmake_dir = Path(base_dir) / "cmake"
        cmake_dir.mkdir()
        (cmake_dir / "llvm-info.json").write_text(
            json.dumps({"llvm_hash": self.OAI_INFO_HASH, "build_number": 1, "sha256sum": {"ubuntu-x64": "oai"}}))
        if build_info:
            (cmake_dir / "llvm-build-info.json").write_text(
                json.dumps({"llvm_hash": self.OAI_BUILD_HASH, "repository": "triton-lang/llvm-project"}))
        (cmake_dir / "meta-llvm-info.json").write_text(json.dumps(meta))

    def test_meta_overrides_hash_url_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as base_dir:
            self._write(base_dir, self.META)
            package = build_helpers.llvm_package_for_suffix("ubuntu-x64", base_dir=base_dir)
        self.assertEqual("llvm-a6533ab0-ubuntu-x64-2", package.name)
        self.assertEqual(
            "https://github.com/facebookexperimental/llvm-project/releases/download/llvm-a6533ab0-2/"
            "llvm-a6533ab0-ubuntu-x64-2.tar.gz", package.url)
        self.assertEqual("meta-ubuntu-x64", package.sha256sum)

    def test_meta_base_ignores_build_info(self) -> None:
        # llvm-build-info.json moves before the Triton sources adopt that LLVM.
        with tempfile.TemporaryDirectory() as base_dir:
            self._write(base_dir, {**self.META, "based_on_oai_llvm_hash": self.OAI_BUILD_HASH})
            with self.assertRaisesRegex(RuntimeError,
                                        "based on OAI LLVM 1df311a6f7d8 but cmake/llvm-info.json pins ce3529423abd"):
                build_helpers.llvm_package_for_suffix("ubuntu-x64", base_dir=base_dir)

    def test_unpublished_platform_ok_with_llvm_syspath(self) -> None:
        # A local LLVM build must keep working on platforms with no Meta bundle.
        with tempfile.TemporaryDirectory() as base_dir:
            self._write(base_dir, self.META)
            helper_args = _helper_args(base_dir, llvm_system_suffix="macos-arm64", llvm_syspath="/opt/llvm")
            with patch.object(build_helpers, "get_base_dir", return_value=base_dir):
                package = build_helpers.get_llvm_package_info(helper_args)
        self.assertEqual("", package.url)

    def test_amd_codegen_builds_against_core_meta_llvm(self) -> None:
        with tempfile.TemporaryDirectory() as base_dir:
            self._write(base_dir, self.META)
            built = Path(base_dir) / "built" / "libtriton_amd_codegen.so"
            built.parent.mkdir()
            built.write_bytes(b"so")
            helper_args = _helper_args(base_dir, llvm_system_suffix="macos-arm64", llvm_syspath="/opt/llvm")
            with patch.object(build_helpers, "get_base_dir", return_value=base_dir), \
                    patch.object(build_helpers, "amd_codegen_library_name", return_value=built.name), \
                    patch.object(build_helpers, "build_amd_codegen", return_value=str(built)) as build, \
                    patch.object(build_helpers, "get_amd_codegen_package_info") as oai_package:
                build_helpers.download_and_copy_amd_codegen(helper_args)
            info, _ = build.call_args.args
            self.assertEqual(self.META["llvm_hash"], info["llvm_hash"])
            self.assertEqual("/opt/llvm", build.call_args.kwargs["llvm_path"])
            oai_package.assert_not_called()
            copied = Path(base_dir) / "third_party" / "amd" / "backend" / "lib" / built.name
            self.assertEqual(b"so", copied.read_bytes())
