"""Meta LLVM fork builds for facebookexperimental/triton.

cmake/llvm-info.json is OAI's pin and stays byte-identical to upstream, so
upstream picks apply cleanly. cmake/meta-llvm-info.json, when present, names
the Meta LLVM fork build, which must sit on OAI's pin. build_helpers.py calls
in here from two hooks; keep Meta's dependency logic in this file.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import build_helpers as bh
else:
    try:
        import build_helpers as bh
    except ImportError:  # run as a package (e.g. setup.py imports python.meta_llvm)
        from python import build_helpers as bh


def enabled(base_dir: str) -> bool:
    return os.path.exists(os.path.join(base_dir, "cmake", "meta-llvm-info.json"))


def load_llvm_info(base_dir: Optional[str] = None) -> dict:
    # INVARIANT: compare against llvm-info.json, the LLVM the Triton sources
    # are written against. llvm-build-info.json only requests the next OAI
    # build and moves days before the sources do.
    cmake_dir = os.path.join(base_dir or bh.get_base_dir(), "cmake")
    with open(os.path.join(cmake_dir, "llvm-info.json"), "r") as llvm_info_file:
        oai_info = json.load(llvm_info_file)
    with open(os.path.join(cmake_dir, "meta-llvm-info.json"), "r") as meta_info_file:
        meta_info = json.load(meta_info_file)
    if meta_info["based_on_oai_llvm_hash"] != oai_info["llvm_hash"]:
        raise RuntimeError(f"cmake/meta-llvm-info.json is based on OAI LLVM {meta_info['based_on_oai_llvm_hash'][:12]} "
                           f"but cmake/llvm-info.json pins {oai_info['llvm_hash'][:12]}; "
                           "rebuild the Meta LLVM on the new OAI pin")
    return meta_info


def llvm_package_for_suffix(system_suffix: str, base_dir: Optional[str] = None) -> "bh.Package":
    llvm_info = load_llvm_info(base_dir)
    sha256sums = llvm_info["sha256sum"]
    if system_suffix not in sha256sums:
        raise RuntimeError(f"LLVM pre-compiled image is not available for {system_suffix} "
                           f"(published: {', '.join(sorted(sha256sums))}); set LLVM_SYSPATH to a local LLVM build")
    name = f"llvm-{llvm_info['llvm_hash'][:8]}-{system_suffix}-{llvm_info['build_number']}"
    return bh.Package(
        "llvm",
        name,
        f"{llvm_info['release_base_url'].rstrip('/')}/{name}.tar.gz",
        "LLVM_INCLUDE_DIRS",
        "LLVM_LIBRARY_DIR",
        "LLVM_SYSPATH",
        # Stable symlink that doesn't include the revision.
        sym_name=f"llvm-{system_suffix}",
        sha256sum=sha256sums[system_suffix],
    )


def llvm_package(helper_args: "bh.BuildHelperArgs") -> "bh.Package":
    system_suffix = bh.get_llvm_system_suffix(helper_args)
    if system_suffix is None:
        return bh.Package("llvm", "LLVM-C.lib", "", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")
    try:
        return llvm_package_for_suffix(system_suffix)
    except RuntimeError as error:
        if helper_args.llvm_syspath is None:
            raise
        print(f"{error}. Proceeding with LLVM_SYSPATH={helper_args.llvm_syspath}.")
        return bh.Package("llvm", "LLVM-C.lib", "", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")


def build_amd_codegen(llvm_info: dict, helper_args: "bh.BuildHelperArgs", llvm_path: str) -> str:
    # Builds against the core Meta LLVM with the host compiler, as libtriton
    # does: the bundle's clang may not find a matching libstdc++.
    revision = f'{llvm_info["llvm_hash"]}-{llvm_info["build_number"]}'
    system_suffix = bh.get_llvm_system_suffix(helper_args)
    source_path = os.path.join(bh.get_base_dir(), "third_party", "amd", "backend", "codegen")
    checkout_hash = hashlib.sha256(os.path.realpath(source_path).encode()).hexdigest()[:12]
    build_path = os.path.join(helper_args.cache_path, "amd",
                              f"codegen-bootstrap-{revision}-{system_suffix}-{checkout_hash}")
    install_path = os.path.join(build_path, "install")
    llvm_cmake_path = os.path.join(llvm_path, "lib", "cmake", "llvm")
    command = [
        "cmake",
        "-G",
        "Ninja",
        "-S",
        source_path,
        "-B",
        build_path,
        f"-DLLVM_DIR={llvm_cmake_path}",
        f"-DCMAKE_INSTALL_PREFIX={install_path}",
        f"-DTRITON_AMD_LLVM_REVISION={revision}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    ninja_path = shutil.which("ninja")
    if ninja_path is not None:
        command.append(f"-DCMAKE_MAKE_PROGRAM={ninja_path}")
    if sys.platform == "darwin":
        sdk = os.environ.get("SDKROOT", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
        linker = "/Library/Developer/CommandLineTools/usr/bin/ld"
        if os.path.isdir(sdk):
            command.extend([
                f"-DCMAKE_OSX_SYSROOT={sdk}",
                f"-DCMAKE_CXX_FLAGS=-isystem{sdk}/usr/include/c++/v1",
            ])
        if os.path.isfile(linker):
            linker_flag = f"-fuse-ld={linker}"
            command.extend([
                f"-DCMAKE_EXE_LINKER_FLAGS={linker_flag}",
                f"-DCMAKE_SHARED_LINKER_FLAGS={linker_flag}",
            ])
    subprocess.check_call(command)
    subprocess.check_call(["cmake", "--build", build_path, "--config", "Release", "--target", "triton_amd_codegen"])
    subprocess.check_call(["cmake", "--install", build_path, "--config", "Release"])
    return os.path.join(install_path, "lib", bh.amd_codegen_library_name())


def copy_amd_codegen(helper_args: "bh.BuildHelperArgs", library: str) -> None:
    # The Meta LLVM fork carries AMDGPU fixes that OAI's codegen artifact lacks.
    # TODO: build against a Meta AMD codegen LLVM once one is pinned.
    core_llvm = bh._get_thirdparty_package_cmake_vars(llvm_package(helper_args), helper_args)
    source_path = build_amd_codegen(load_llvm_info(), helper_args, llvm_path=core_llvm["LLVM_SYSPATH"])
    if not os.path.isfile(source_path):
        raise RuntimeError(f"Cannot find AMD code generation at {source_path}; set TRITON_AMD_CODEGEN_PATH")
    destination_path = os.path.join(bh.get_base_dir(), "third_party", "amd", "backend", "lib", library)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    print(f"copy {source_path} to {destination_path} ...")
    shutil.copy2(source_path, destination_path)
