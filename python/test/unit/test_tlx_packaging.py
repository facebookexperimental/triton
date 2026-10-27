"""Regression test: a fresh (no build symlinks) tree must package `triton.tlx`.

`python/triton/tlx/__init__.py` does `from . import ops`, where `ops` is
linked in from `third_party/tlx/ops` by setup.py. On a fresh release cut
find_packages() cannot see that link yet, so setup.py must declare the
`triton.tlx.ops.*` packages explicitly and create the link before files are
copied; otherwise `import triton.tlx` fails in the installed wheel/sdist.

setup.py is exec'd with a stubbed setuptools so the test needs no cmake,
GPU, or compiled extension. Skipped when the source layout is unavailable
(e.g. a buck link tree without setup.py).
"""

import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

TRITON_ROOT = Path(__file__).resolve().parents[3]
SETUP_PY = TRITON_ROOT / "setup.py"


def _find_packages(where="."):
    """Mimic setuptools.find_packages on the fixture tree."""
    found = []
    for dirpath, dirnames, filenames in os.walk(where):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel = os.path.relpath(dirpath, where)
        if rel == ".":
            continue
        if "__init__.py" not in filenames:
            dirnames[:] = []
            continue
        found.append(rel.replace(os.sep, "."))
    return sorted(found)


def _install_stubs(captured):
    def _cls(name):
        return type(name, (), {
            "__init__": lambda self, *a, **k: None,
            "user_options": [],
            "boolean_options": [],
        })

    st = types.ModuleType("setuptools")
    st.Extension = _cls("Extension")
    st.find_packages = _find_packages
    st.setup = lambda **kwargs: captured.update(kwargs)
    cmd = types.ModuleType("setuptools.command")
    st.command = cmd
    saved = {k: sys.modules.get(k) for k in list(sys.modules) if k.startswith(("setuptools", "distutils"))}
    sys.modules["setuptools"] = st
    sys.modules["setuptools.command"] = cmd
    for sub in ("build_ext", "build_py", "develop", "egg_info", "install", "sdist",
                "bdist_wheel", "editable_wheel"):
        mod = types.ModuleType(f"setuptools.command.{sub}")
        setattr(mod, sub, _cls(sub))
        sys.modules[f"setuptools.command.{sub}"] = mod
        setattr(cmd, sub, mod)
    try:
        import distutils.command.clean  # noqa: F401
    except ImportError:
        dist = types.ModuleType("distutils")
        dcmd = types.ModuleType("distutils.command")
        clean_mod = types.ModuleType("distutils.command.clean")
        clean_mod.clean = _cls("clean")
        sys.modules["distutils"] = dist
        sys.modules["distutils.command"] = dcmd
        sys.modules["distutils.command.clean"] = clean_mod
    pb = types.ModuleType("pybind11")
    pb.get_include = lambda: ""
    pb.get_cmake_dir = lambda: ""
    sys.modules["pybind11"] = pb
    return saved


def _restore_modules(saved):
    for key in [k for k in list(sys.modules) if k.startswith(("setuptools", "distutils"))]:
        del sys.modules[key]
    sys.modules.update({k: v for k, v in saved.items() if v is not None})


def _build_fixture(root: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="tlx-packaging-"))
    shutil.copy(SETUP_PY, tmp / "setup.py")
    (tmp / "python").mkdir()
    shutil.copy(TRITON_ROOT / "python" / "build_helpers.py", tmp / "python" / "build_helpers.py")
    for backend in ("nvidia", "amd", "cpu"):
        bdir = tmp / "third_party" / backend / "backend"
        bdir.mkdir(parents=True)
        (bdir / "compiler.py").touch()
        (bdir / "driver.py").touch()
    # Real TLX sources when the checkout provides them; a minimal mirror of
    # the same package shape otherwise (buck link trees).
    real_ops = TRITON_ROOT / "third_party" / "tlx" / "ops"
    ops_dst = tmp / "third_party" / "tlx" / "ops"
    if real_ops.is_dir():
        shutil.copytree(real_ops, ops_dst, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        lang_src = TRITON_ROOT / "third_party" / "tlx" / "language"
        shutil.copytree(lang_src, tmp / "third_party" / "tlx" / "language", symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    else:
        for pkg in ("", "kernels", "kernels/mm", "kernels/flash_attn"):
            d = ops_dst / pkg
            d.mkdir(parents=True, exist_ok=True)
            (d / "__init__.py").touch()
        lang_tlx = tmp / "third_party" / "tlx" / "language" / "tlx"
        lang_tlx.mkdir(parents=True)
        (lang_tlx / "__init__.py").touch()
    tlx_pkg = tmp / "python" / "triton" / "tlx"
    tlx_pkg.mkdir(parents=True)
    shutil.copy(TRITON_ROOT / "python" / "triton" / "tlx" / "__init__.py", tlx_pkg / "__init__.py")
    for parent in ("python/triton/__init__.py", "python/triton/language/__init__.py",
                   "python/triton/language/extra/__init__.py"):
        p = tmp / parent
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)
    return tmp


@unittest.skipUnless(SETUP_PY.is_file(), "setup.py not available in this layout")
class TlxPackagingTest(unittest.TestCase):

    def test_fresh_tree_packages_and_imports_tlx(self) -> None:
        captured: dict = {}
        saved = _install_stubs(captured)
        fixture = _build_fixture(TRITON_ROOT)
        old_cwd = os.getcwd()
        try:
            os.chdir(fixture)
            sys.path.insert(0, str(fixture))
            ns = {"__name__": "__main__", "__file__": str(fixture / "setup.py")}
            exec(compile((fixture / "setup.py").read_text(), "setup.py", "exec"), ns)

            packages = captured.get("packages", [])
            for want in ("triton.tlx", "triton.tlx.ops", "triton.tlx.ops.kernels",
                         "triton.tlx.ops.kernels.mm", "triton.tlx.ops.kernels.flash_attn"):
                self.assertIn(want, packages)
            # Declared in Beta's position, but only when the source exists;
            # 3.8 has no third_party/tlx/language/tlx/ops.
            self.assertNotIn("triton.language.extra.tlx.ops", packages)

            ns["add_link_to_tlx"]()
            ops_link = fixture / "python" / "triton" / "tlx" / "ops"
            self.assertTrue((ops_link / "__init__.py").exists())

            for mod in [m for m in list(sys.modules) if m == "triton" or m.startswith("triton.")]:
                del sys.modules[mod]
            stub = types.ModuleType("triton")
            stub.__path__ = [str(fixture / "python" / "triton")]
            sys.modules["triton"] = stub
            tlx = importlib.import_module("triton.tlx")
            self.assertTrue(hasattr(tlx, "ops"))
            self.assertIsNotNone(importlib.util.find_spec("triton.tlx.ops.kernels.mm"))
        finally:
            os.chdir(old_cwd)
            if str(fixture) in sys.path:
                sys.path.remove(str(fixture))
            _restore_modules(saved)
            shutil.rmtree(fixture, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
