from __future__ import annotations

import pytest
import tempfile

from pathlib import Path

import triton

from triton.runtime.build import compile_module_from_src

TEST_MODULE_C = """
#include <Python.h>
#include <string.h>

static PyObject* go(PyObject* self, PyObject* args) {
    const char *command;
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    const char* res;
    if (strcmp(command, "hello") == 0) {
        res = "hiya";
    } else {
        res = "huh";
    }
    return PyUnicode_FromString(res);
}

static PyMethodDef ModuleMethods[] = {
  {"go", go, METH_VARARGS, "test_module.go for testing"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "test_module",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit_test_module(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""


def test_compile_module(fresh_triton_cache):
    mod = compile_module_from_src(TEST_MODULE_C, "test_module")

    with pytest.raises(Exception):
        mod.go()

    assert mod.go("huh") == "huh"
    assert mod.go("hello") == "hiya"

    # Make sure the module is cached
    mod2 = compile_module_from_src(TEST_MODULE_C, "test_module")
    assert mod2.__file__ == mod.__file__


def test_compile_module_bad_cache(fresh_knobs):
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        called_get_file = False

        class InvalidFileCacheManager(triton.runtime.cache.FileCacheManager):

            def get_file(self, filename: str) -> str | None:
                nonlocal called_get_file
                called_get_file = True
                (tmp / filename).write_text("not an so")
                return str(tmp / filename)

        # First corrupt the cache
        fresh_knobs.cache.manager_class = InvalidFileCacheManager

        mod = compile_module_from_src(TEST_MODULE_C, "test_module")
        assert called_get_file

        with pytest.raises(Exception):
            mod.go()

        assert mod.go("huh") == "huh"
        assert mod.go("hello") == "hiya"


def _cache_key(build, root, library_dirs, include_dirs):
    config = dict(language=["c"], library_dirs=library_dirs, include_dirs=include_dirs, libraries=None, ccflags=None)
    return build._get_cache_manager(b"src", config=config).key


def test_cache_key_ignores_triton_install_location(monkeypatch, fresh_triton_cache):
    from triton.runtime import build

    keys = []
    for root in ("/mnt/xarfuse/uid-0/seed-a/triton", "/mnt/xarfuse/uid-1/seed-b/triton"):
        monkeypatch.setattr(build, "_TRITON_ROOT", root)
        keys.append(_cache_key(build, root, [f"{root}/backends/nvidia/lib"], [f"{root}/backends/nvidia/include"]))
    assert keys[0] == keys[1]


def test_cache_key_tracks_dirs_outside_triton(monkeypatch, fresh_triton_cache):
    from triton.runtime import build

    monkeypatch.setattr(build, "_TRITON_ROOT", "/opt/triton")
    assert _cache_key(build, "/opt/triton", ["/usr/lib/a"], None) != _cache_key(build, "/opt/triton", ["/usr/lib/b"], None)
