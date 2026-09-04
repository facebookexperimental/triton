"""Regression test for the TRITON_PLUGIN_PATHS extension mechanism.

Loads the example plugin from examples/plugins/TritonPlugin.cpp (README example 1,
a pass that renames every function to "foo") and checks it actually rewrites a
module. This covers the whole chain in one shot: libtriton is built with
TRITON_EXT_ENABLED so loadPlugins() does not refuse, its symbols are exported so
the plugin can bind to them, and -Bsymbolic keeps its LLVM separate from the
llvm-fb/19 that torch drags in.

Two things this test depends on that are easy to break:

  * RTLD_GLOBAL. CPython dlopens extension modules RTLD_LOCAL, which leaves
    libtriton's symbols out of the global scope, and then no plugin can bind to
    them however much is in .dynsym. This is the in-process equivalent of the
    LD_PRELOAD in test/Plugins/test-plugin.mlir.
  * Default visibility on the plugin. Built with -fvisibility=hidden it keeps
    private copies of vague-linkage symbols (TypeID statics, vtables, RTTI)
    instead of binding to libtriton's, and segfaults when the pass is added.
"""

import os
import sys
import tempfile
import unittest

MLIR = """
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:80"} {
  tt.func @bar() {
    tt.return
  }
}
"""

# Must happen before triton is imported anywhere in this process.
assert "triton" not in sys.modules, "triton already imported before RTLD_GLOBAL setup"
_OLD_DLOPEN_FLAGS = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
try:
    from triton._C.libtriton import ir, passes, tlx
finally:
    sys.setdlopenflags(_OLD_DLOPEN_FLAGS)


class TritonPluginLoadTest(unittest.TestCase):
    def test_plugin_path_is_wired_up(self):
        paths = os.environ.get("TRITON_PLUGIN_PATHS")
        self.assertTrue(paths, "TRITON_PLUGIN_PATHS unset; the target's env is not reaching the test")
        for p in [x for x in paths.split(os.pathsep) if x]:
            self.assertTrue(os.path.exists(p), f"plugin {p} does not exist")

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
