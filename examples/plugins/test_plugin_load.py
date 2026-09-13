"""Regression test for the TRITON_PLUGIN_PATHS extension mechanism.

Loads the example plugin from examples/plugins/TritonPlugin.cpp (README example 1,
a pass that renames every function to "foo") and checks it actually rewrites a
module. This covers the whole chain in one shot: libtriton is built with
TRITON_EXT_ENABLED so loadPlugins() does not refuse, its symbols are exported so
the plugin can bind to them, and -Bsymbolic keeps its LLVM separate from the
llvm-fb/19 that torch drags in.

Three things this test depends on that are easy to break:

  * RTLD_GLOBAL. CPython dlopens extension modules RTLD_LOCAL, which leaves
    libtriton's symbols out of the global scope, and then no plugin can bind to
    them however much is in .dynsym. This is the in-process equivalent of the
    LD_PRELOAD in test/Plugins/test-plugin.mlir.
  * Default visibility on the plugin. Built with -fvisibility=hidden it keeps
    private copies of vague-linkage symbols (TypeID statics, vtables, RTTI)
    instead of binding to libtriton's, and segfaults when the pass is added.
  * -Bsymbolic on libtriton. The target's caffe2:torch dep statically links
    llvm-fb/19 into the test binary, so this process holds two LLVMs, both
    registering the same llvm::cl:: option names. Without -Bsymbolic the dynamic
    linker merges the registries when libtriton is dlopened RTLD_GLOBAL, and LLVM
    aborts. Two tests cover it: one checks the second LLVM is actually here (else
    the coverage is vacuous), the other pins the flag on the artifact. Reaching
    either at all is itself the behavioral half -- a regression aborts the
    process during import, before any test runs.
"""

import os
import re
import struct
import sys
import tempfile
import unittest

import torch  # noqa: F401

MLIR = """
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:80"} {
  tt.func @bar() {
    tt.return
  }
}
"""

# Must happen before triton is imported anywhere in this process.
if "triton" in sys.modules:
    raise RuntimeError("triton already imported before RTLD_GLOBAL setup")
_OLD_DLOPEN_FLAGS = sys.getdlopenflags()
sys.setdlopenflags(_OLD_DLOPEN_FLAGS | os.RTLD_NOW | os.RTLD_GLOBAL)
try:
    from triton._C import libtriton
    from triton._C.libtriton import ir, passes, tlx
finally:
    sys.setdlopenflags(_OLD_DLOPEN_FLAGS)

from triton import knobs

# The cl::opt that llvm/lib/IR/Constants.cpp registers. Both LLVMs in this process
# compile that file, which is what makes the duplicate registration fatal.
_DUPLICATE_CL_OPT = b"use-constant-int-for-fixed-length-splat"

# ELF64 dynamic-section constants, so the -Bsymbolic check needs no binutils.
_DT_NULL = 0
_DT_SYMBOLIC = 16
_DT_FLAGS = 30
_DF_SYMBOLIC = 0x02
_PT_DYNAMIC = 2


def _mapped_files_containing(needle):
    """Files mapped into this process whose contents contain `needle`.

    Not just shared objects: torch's LLVM is statically linked into the test
    binary rather than dlopened, so a .so-only scan finds nothing and would make
    the -Bsymbolic coverage look absent when it is present.
    """
    with open("/proc/self/maps") as f:
        paths = sorted({m for m in re.findall(r"(/\S+)$", f.read(), re.MULTILINE)})

    hits = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                if needle in f.read():
                    hits.append(path)
        except OSError:
            continue
    return hits


def _dynamic_tags(path):
    """The (d_tag, d_val) pairs of an ELF64 little-endian shared object."""
    with open(path, "rb") as f:
        elf = f.read()
    e_phoff, = struct.unpack_from("<Q", elf, 0x20)
    e_phentsize, e_phnum = struct.unpack_from("<HH", elf, 0x36)
    for i in range(e_phnum):
        phdr = e_phoff + i * e_phentsize
        p_type, = struct.unpack_from("<I", elf, phdr)
        if p_type != _PT_DYNAMIC:
            continue
        p_offset, = struct.unpack_from("<Q", elf, phdr + 0x08)
        p_filesz, = struct.unpack_from("<Q", elf, phdr + 0x20)
        for off in range(p_offset, p_offset + p_filesz, 16):
            d_tag, d_val = struct.unpack_from("<qQ", elf, off)
            if d_tag == _DT_NULL:
                return
            yield d_tag, d_val


class TritonPluginLoadTest(unittest.TestCase):
    def test_plugin_path_is_wired_up(self):
        paths = knobs.compilation.plugin_paths
        self.assertTrue(paths, "TRITON_PLUGIN_PATHS unset; the target's env is not reaching the test")
        for p in [x for x in paths.split(os.pathsep) if x]:
            self.assertTrue(os.path.exists(p), f"plugin {p} does not exist")

    def test_process_really_holds_a_second_llvm(self):
        # Guards the test below from going quietly vacuous. If caffe2:torch stops
        # dragging llvm-fb/19 into this binary, nothing here exercises -Bsymbolic
        # any more and a green run stops meaning anything.
        carriers = _mapped_files_containing(_DUPLICATE_CL_OPT)
        self.assertGreaterEqual(
            len(carriers), 2,
            "expected libtriton's LLVM and torch's llvm-fb/19 to both be mapped into this process, "
            f"found only {carriers}; check the caffe2:torch dep on the target",
        )

    def test_libtriton_binds_its_own_llvm(self):
        # Exporting libtriton's symbols for plugins also exposes its llvm::cl::
        # registry. -Bsymbolic is what stops the dynamic linker merging it with
        # the second LLVM above, which would re-register the same option names and
        # abort the process at import. Reaching this assertion at all is the
        # behavioral half; the assertion itself pins the flag on the artifact.
        symbolic = [(tag, val) for tag, val in _dynamic_tags(libtriton.__file__)
                    if tag == _DT_SYMBOLIC or (tag == _DT_FLAGS and val & _DF_SYMBOLIC)]
        self.assertTrue(symbolic, "libtriton is not linked -Bsymbolic; its LLVM will merge with torch's")

    def test_plugin_registers_its_pass(self):
        # loadPlugins() refuses outright when libtriton lacks TRITON_EXT_ENABLED, so
        # an empty submodule here means the build config regressed, not the plugin.
        self.assertTrue(hasattr(passes, "plugin"), "passes.plugin submodule missing")
        self.assertTrue(
            hasattr(passes.plugin, "add_plugin"),
            "plugin loaded no pass; check for a 'not built with TRITON_EXT_ENABLED' warning on stderr",
        )

    def test_plugin_pass_rewrites_module(self):
        self._assert_pass_rewrites_module(passes.plugin.add_plugin)

    def test_tlx_plugin_registers_its_pass(self):
        # tlx_passes.plugin is built by init_triton_tlx (third_party/tlx/dialect/
        # triton_tlx.cc) off the same memoized loadPlugins() list, so anything in
        # passes.plugin must show up here too. TLX is a Triton *backend* submodule
        # (main.cc INIT_BACKEND), which is why this hangs off libtriton.tlx rather
        # than libtriton.passes.
        self.assertTrue(hasattr(tlx, "tlx_passes"), "tlx.tlx_passes submodule missing")
        self.assertTrue(hasattr(tlx.tlx_passes, "plugin"), "tlx_passes.plugin submodule missing")
        self.assertTrue(
            hasattr(tlx.tlx_passes.plugin, "add_plugin"),
            "TLX plugin namespace registered no pass; core passes.plugin is populated from the "
            "same list, so a mismatch means init_triton_tlx_plugin_passes regressed",
        )

    def test_tlx_plugin_pass_rewrites_module(self):
        self._assert_pass_rewrites_module(tlx.tlx_passes.plugin.add_plugin)

    def _assert_pass_rewrites_module(self, add_pass):
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
            f.write(MLIR)
            path = f.name
        self.addCleanup(os.unlink, path)

        ctx = ir.context()
        ir.load_dialects(ctx)
        tlx.load_dialects(ctx)
        mod = ir.parse_mlir_module(path, ctx)
        self.assertIn("@bar", str(mod))

        pm = ir.pass_manager(ctx)
        add_pass(pm)
        pm.run(mod, "test_plugin_load")

        after = str(mod)
        self.assertIn("@foo", after)
        self.assertNotIn("@bar", after)
