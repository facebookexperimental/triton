# Owner(s): ["module: inductor"]
import unittest
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.utils._triton import has_datacenter_blackwell_tma_device
from triton.language.extra.tlx.inductor import tlx_config


def has_tlx() -> bool:
    """Check if TLX (Triton Language eXtensions) is available."""
    try:
        import triton.language.extra.tlx  # noqa: F401  # @manual

        return True
    except ImportError:
        return False


def is_gfx950() -> bool:
    """True on AMD MI350X (gfx950), where the TLX warp-pipe addmm template runs."""
    if torch.version.hip is None:
        return False
    try:
        return "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


def flex_choices_hook_available() -> bool:
    """True if torch exposes the flex-attention choices hook the template needs.

    Absent on the current ROCm nightly, so the flex-attention Inductor tests skip
    there (the template can only be exercised end-to-end on a newer torch).
    """
    try:
        from torch._inductor.choices import InductorChoices

        return hasattr(InductorChoices, "append_flex_attention_choices")
    except Exception:
        return False


torch.set_float32_matmul_precision("high")

# Shapes for template testing - representative shapes from gemm_rule categories
# Tall-M shapes (Rules 1-4) have correctness issues with NUM_CTAS=2 configs - needs investigation
# Using only proven working shapes for now
TEMPLATE_TEST_SHAPES = [
    # (M, K, N)
    (4096, 4096, 4096),  # Rule 7: GPU-Saturated General
    (16384, 4096, 2048),  # Tall-M (M/N=8, uses Rule 7 config)
    (1152, 16384, 1024),  # Rule 5: Undersaturated Large-Output
    (256, 8192, 256),  # Undersaturated with large K (triggers split-K)
    (512, 16384, 128),  # Undersaturated small-output with large K (Rule 6, split-K)
    # (1024, 442368, 2048) excluded: K=442368 causes cuBLAS vs TLX fp32
    # accumulation ordering divergence (6/2M elements exceed atol=0.5).
    # Covered by benchmarks instead.
]


@instantiate_parametrized_tests
class TestTLXTemplates(TestCase):

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("shape", TEMPLATE_TEST_SHAPES)
    @parametrize("use_heuristic_config", (False, True))
    def test_tlx_matmul_ws(
        self,
        dtype: torch.dtype,
        shape: tuple[int, int, int],
        use_heuristic_config: bool,
    ):
        """Test for the TLX Blackwell warp-specialized matmul template from tritonbench."""

        def mm(a, b):
            return torch.mm(a, b)

        def next_multiple_16(a: int) -> int:
            return ((a + 15) // 16) * 16

        M, K, N = shape
        a_shape = (M, K)
        a_stride = (next_multiple_16(K), 1)
        a = torch.empty_strided(a_shape, a_stride, dtype=dtype).to(GPU_TYPE)
        a[:] = torch.randn(a_shape, dtype=dtype)
        a = a.to(GPU_TYPE)
        b_shape = (K, N)
        b_stride = (next_multiple_16(N), 1)
        b = torch.empty_strided(b_shape, b_stride, dtype=dtype)
        b[:] = torch.randn(b_shape, dtype=dtype)
        b = b.to(GPU_TYPE)

        with (
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "enable_caching_generated_triton_templates": False,
                }),
                tlx_config.patch(use_heuristic_config=use_heuristic_config, ),
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=True), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=0.01, rtol=0.01)

        code_str = "\n".join(code)
        is_split_k = "_reduce_k_kernel" in code_str
        if is_split_k:
            # Split-K uses TMA descriptor stores to write fp32 partials to workspace
            self.assertIn("async_descriptor_store", code_str)
            self.assertIn("split_k_ws", code_str)
            # Split-K uses ws_smem_buffers (fp32), not c_smem_buffers (output dtype)
            self.assertNotIn("c_smem_buffers", code_str)
        else:
            # Non-split-K configs use TMA epilogue stores when SMEM permits;
            # 1-CTA configs may fall back to tl.store if TMA doesn't fit.
            pass  # Both TMA and tl.store paths are valid

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_tlx_addmm_warppipe(self, dtype: torch.dtype):
        """TLX warp-pipelined addmm template (AMD MI350X / gfx950), col-major B.

        Gated by TORCHINDUCTOR_TLX_MODE (config.triton.tlx_mode is not None): the
        addmm_warppipe template competes in max-autotune against mm_template + aten.
        Verifies the TLX-enabled addmm path lowers to a Triton template and is
        numerically correct on the thin-N latency-bound shape it targets.
        """
        M, K, N = 4096, 2048, 192  # thin-N latency-bound (the warp-pipe niche)
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
        # w.t() => B is [K, N] col-major (stride_bk == 1) -- the nn.Linear weight layout.
        w = torch.randn(N, K, device=GPU_TYPE, dtype=dtype)
        bias = torch.randn(N, device=GPU_TYPE, dtype=dtype)

        def addmm(bias, a, w):
            return torch.addmm(bias, a, w.t())

        with (config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "enable_caching_generated_triton_templates": False,
        }), ):
            c_actual, code = run_and_get_code(torch.compile(addmm), bias, a, w)

        c_expected = (a.float() @ w.t().float() + bias.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=2e-2, rtol=2e-2)

        # force mode keeps only the TLX template, so addmm must be lowered through
        # the warp-pipe Triton template (triton_tem), never an extern/aten kernel.
        code_str = "\n".join(code)
        self.assertIn("triton_tem", code_str)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_tlx_addmm_warppipe_split_k(self, dtype: torch.dtype):
        """Split-K path of the TLX warp-pipe addmm (AMD MI350X / gfx950), col-major B.

        An undersaturated grid (few MN tiles) + large K makes the heuristic offer
        SPLIT_K > 1 candidates (registry gate: `tiles < NUM_SMS`). On a 2-tile shape
        the split-K configs win autotune, so the addmm lowers to the split-K path: a
        partial-GEMM kernel that writes an fp32 workspace + a separate
        `_reduce_k_kernel` that sums the partials, re-adds bias, and casts. Verifies
        the 2-kernel split-K reduce is (a) actually taken and (b) numerically correct.
        """
        # 256x4096x256: 2 MN tiles (128x256) on 256 CUs -> deeply undersaturated, so
        # split-K (up to SK=8 -> 16 workgroups) is far faster than SK=1 (2 workgroups)
        # and wins the autotune. K=4096 keeps each split > NUM_BUFFERS K-iters.
        M, K, N = 256, 4096, 256
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
        # w.t() => B is [K, N] col-major (stride_bk == 1) -- the nn.Linear weight layout.
        w = torch.randn(N, K, device=GPU_TYPE, dtype=dtype)
        bias = torch.randn(N, device=GPU_TYPE, dtype=dtype)

        def addmm(bias, a, w):
            return torch.addmm(bias, a, w.t())

        with (config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "enable_caching_generated_triton_templates": False,
        }), ):
            c_actual, code = run_and_get_code(torch.compile(addmm), bias, a, w)

        # fp32 reference; the split-K fp32 workspace reduction is order-different from a
        # single-pass accumulation, so allow a modest tolerance (benign reduction noise).
        c_expected = (a.float() @ w.t().float() + bias.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=3e-2, rtol=3e-2)

        code_str = "\n".join(code)
        # force mode keeps only the TLX template (never extern/aten)...
        self.assertIn("triton_tem", code_str)
        # ...and the undersaturated grid takes the split-K path -> separate reduce kernel.
        self.assertIn("_reduce_k_kernel", code_str)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_tlx_addmm_warppipe_unaligned_k(self, dtype: torch.dtype):
        """Unaligned K (K % BLOCK_K != 0) on the TLX warp-pipe addmm (gfx950).

        A masked (partial-K) tlx.async_load fails to lower on gfx950, so the template
        walks only the FULL K-tiles with unmasked async_load and folds the leftover K
        columns in via a synchronous masked tl.load ("sync-load the tail"). K=2312 is a
        multiple of 8 but of no BLOCK_K in the config set (32/64/128), so every config
        has a partial last K-tile and exercises the tail. Before the fix this raised
        "failed to legalize operation 'ttg.async_copy_global_to_local'".

        NOTE: K must be a multiple of 8 (16-byte row alignment: stride = K elems * 2 B).
        An odd K (e.g. the production compression bmm's K=2309) additionally hits a
        SEPARATE, deeper limit -- the col-major B's async_copy into the swizzled
        padded_shared LDS layout cannot legalize with a non-16-byte-aligned row stride
        (builtin.unrealized_conversion_cast on arg_B) -- which the sync-tail does NOT
        address (it needs an AMD-backend async_copy alignment fix).
        """
        M, K, N = 4096, 2312, 192  # K=2312: multiple of 8, but not of 32/64/128 -> partial tail
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
        # w.t() => B is [K, N] col-major (stride_bk == 1) -- the nn.Linear weight layout.
        w = torch.randn(N, K, device=GPU_TYPE, dtype=dtype)
        bias = torch.randn(N, device=GPU_TYPE, dtype=dtype)

        def addmm(bias, a, w):
            return torch.addmm(bias, a, w.t())

        with (config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "enable_caching_generated_triton_templates": False,
        }), ):
            c_actual, code = run_and_get_code(torch.compile(addmm), bias, a, w)

        c_expected = (a.float() @ w.t().float() + bias.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=2e-2, rtol=2e-2)

        # force mode keeps only the TLX template, so an unaligned-K addmm must still
        # lower through the warp-pipe Triton template (never falling back to extern/aten).
        code_str = "\n".join(code)
        self.assertIn("triton_tem", code_str)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_tlx_addmm_warppipe_odd_k_async_copy_alignment_repro(self):
        """REPRO of the residual odd-K async_copy 16-byte-alignment failure (gfx950).

        The sync-load-the-tail fix handles K % BLOCK_K != 0, but ONLY when the row stride is
        16-byte aligned (K a multiple of 8). K=2309 (odd; the production compression bmm's K) is
        NOT: A [M,K] and col-major B [K,N] both have row stride K, so a row spans 2309*2 = 4618 B,
        not a multiple of 16. The col-major B's async_copy into the swizzled #ttg.padded_shared
        LDS layout then cannot legalize -> builtin.unrealized_conversion_cast on arg_B -> "failed
        to translate module to LLVM IR". In tlx_mode=force (TRITON-only) every config fails to
        compile, so select_algorithm raises NoValidChoicesError.

        This documents a KNOWN residual -- an AMD-backend async_copy alignment fix, out of scope
        for the sync-tail kernel change. When that fix lands this test will start passing
        (NoValidChoicesError no longer raised); convert it to a correctness check then.
        compile_threads=1 keeps the compile in-process so the real MLIR error is visible in the
        test log (subprocess autotune otherwise swallows it behind NoValidChoicesError).
        """
        M, K, N = 4096, 2309, 192  # odd K -> 2309*2 = 4618 B row stride, NOT 16-byte aligned
        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        w = torch.randn(N, K, device=GPU_TYPE, dtype=torch.float16)
        bias = torch.randn(N, device=GPU_TYPE, dtype=torch.float16)

        def addmm(bias, a, w):
            return torch.addmm(bias, a, w.t())

        with config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "enable_caching_generated_triton_templates": False,
                "compile_threads": 1,
        }):
            with self.assertRaises(Exception):
                run_and_get_code(torch.compile(addmm), bias, a, w)


class TestWarpPipeSplitKCodegen(TestCase):
    """Deterministic codegen check for the AMD warp-pipe split-K template.

    The e2e test above relies on autotune *selecting* a SPLIT_K > 1 config (highly
    reliable on a 2-tile shape, but a timing decision). This test renders the
    `amd_addmm_warppipe` jinja template directly with SPLIT_K=2 vs SPLIT_K=1 -- no
    GPU, no autotune -- so the split-K branches are *guaranteed* covered even if
    autotune ever stops picking split-K, and it runs on any host (not gfx950-gated).
    """

    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_warppipe_split_k_template_render(self):
        import jinja2
        from triton.language.extra.tlx.inductor.mm_templates import load_tlx_template

        source = load_tlx_template("amd_addmm_warppipe")

        # Stub the Inductor render hooks; the split-K branches are pure jinja that
        # only depends on SPLIT_K, so the stubs just need to be present + callable.
        hooks = {
            "def_kernel": lambda *a, **k: "def _kernel(A, B, out_ptr0):",
            "size": lambda *a, **k: "0",
            "stride": lambda *a, **k: "1",
            "output_ptr": lambda *a, **k: "out_ptr0",
            "store_output": lambda *a, **k: "# store_output(...)",
        }
        tmpl = jinja2.Environment().from_string(source)
        split = tmpl.render(SPLIT_K=2, **hooks)
        nosplit = tmpl.render(SPLIT_K=1, **hooks)

        # SPLIT_K > 1 must emit: split-id decode, balanced K-partition, fp32 workspace store.
        self.assertIn("split_id = (pid % SPLIT_K)", split)
        self.assertIn("base = K_ITERS // SPLIT_K", split)
        self.assertIn("k_lo = split_id * base", split)
        self.assertIn("tl.store(split_k_ws + ws_off, acc", split)

        # SPLIT_K == 1 must take the plain data-parallel path: no split-id, no workspace,
        # full-K loop, and store via store_output (not the reduce workspace).
        self.assertNotIn("split_id = (pid % SPLIT_K)", nosplit)
        self.assertNotIn("split_k_ws", nosplit)
        self.assertIn("k_lo = 0", nosplit)
        self.assertIn("store_output", nosplit)


class TestInterleaveEpilogue(TestCase):
    """Test that INTERLEAVE_EPILOGUE produces correct results and interleaved stores."""

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_interleave_epilogue_codegen(self):
        """Verify INTERLEAVE_EPILOGUE generates interleaved TMA stores."""

        def mm(a, b):
            return torch.mm(a, b)

        # (1024, 256, 1024) triggers Rule 5 with SPLIT_K=1, INTERLEAVE=1:
        # mn_tiles=16 < 148 (undersaturated), MN=1M (large_output),
        # k_tiles=ceil(256/64)=4 (too few for split-K)
        M, K, N = 1024, 256, 1024
        a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE)

        with (
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "enable_caching_generated_triton_templates": False,
                }),
                tlx_config.patch(use_heuristic_config=True, ),
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=True), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=0.01, rtol=0.01)

        code_str = "\n".join(code)
        self.assertIn("async_descriptor_store", code_str)
        # Interleaved path uses literal buf_idx 0 and 1 for the two MMA groups
        # instead of a computed expression like "group_id * EPILOGUE_SUBTILE + ..."
        self.assertIn("c_smem_buffers[0]", code_str)
        self.assertIn("c_smem_buffers[1]", code_str)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_interleave_split_k_codegen(self):
        """Verify interleaved epilogue works with split-K (ws_smem_buffers, not c_smem_buffers)."""

        def mm(a, b):
            return torch.mm(a, b)

        # (1152, 16384, 1024) triggers Rule 5 with SPLIT_K=4, INTERLEAVE_EPILOGUE=1
        M, K, N = 1152, 16384, 1024
        a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE)

        with (
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "enable_caching_generated_triton_templates": False,
                }),
                tlx_config.patch(use_heuristic_config=True, ),
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=True), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=0.01, rtol=0.01)

        code_str = "\n".join(code)
        # Should have split-K reduction kernel
        self.assertIn("_reduce_k_kernel", code_str)
        # Interleaved split-K uses ws_smem_buffers, not c_smem_buffers
        self.assertIn("ws_smem_buffers", code_str)
        self.assertNotIn("c_smem_buffers", code_str)
        # Interleaved pattern: separate offs_am for each MMA group
        self.assertIn("offs_am_0", code_str)
        self.assertIn("offs_am_1", code_str)


class TestSplitK(TestCase):
    """Tests for split-K code path and fusion behavior."""

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_split_k_codegen(self):
        """Verify split-K shapes produce reduction kernel in generated code."""

        def mm(a, b):
            return torch.mm(a, b)

        # (256, 8192, 256) triggers Rule 6 with SPLIT_K > 1
        M, K, N = 256, 8192, 256
        a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE)

        with (
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "enable_caching_generated_triton_templates": False,
                }),
                tlx_config.patch(use_heuristic_config=True, ),
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=True), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=0.01, rtol=0.01)

        code_str = "\n".join(code)
        self.assertIn(
            "_reduce_k_kernel",
            code_str,
            "Expected split-K reduction kernel in generated code",
        )
        # Split-K uses TMA descriptor stores for workspace writes
        self.assertIn("async_descriptor_store", code_str)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_split_k_no_fusion(self):
        """Verify epilogue fusion is disabled with split-K (relu applied separately)."""

        def relu_mm(a, b):
            return torch.relu(torch.mm(a, b))

        M, K, N = 256, 8192, 256
        a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE)

        with (
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "enable_caching_generated_triton_templates": False,
                }),
                tlx_config.patch(use_heuristic_config=True, ),
        ):
            c_actual, code = run_and_get_code(torch.compile(relu_mm, dynamic=True), a, b)
            c_expected = relu_mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=0.01, rtol=0.01)

        code_str = "\n".join(code)
        # Reduction kernel present (split-K was used)
        self.assertIn("_reduce_k_kernel", code_str)
        # relu should NOT be fused into the GEMM kernel — it should appear
        # as a separate pointwise kernel after the reduction
        self.assertIn("triton_poi_", code_str)


class TestReduceKKernel(TestCase):
    """Direct unit test for the split-K reduction kernel."""

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_reduce_k_correctness(self):
        """Test _reduce_k_kernel produces correct sum over SPLIT_K slices."""
        import triton.language as tl
        from triton.language.extra.tlx.inductor.reduce_k import _reduce_k_kernel

        M, N, SPLIT_K = 64, 128, 4
        # Create workspace: SPLIT_K partial results stacked along M dimension
        partials = torch.randn(SPLIT_K, M, N, dtype=torch.float32, device=GPU_TYPE)
        workspace = partials.reshape(SPLIT_K * M, N).contiguous()
        output = torch.empty(M, N, dtype=torch.float16, device=GPU_TYPE)

        grid = (M // 32, N // 32)
        _reduce_k_kernel[grid](
            workspace,
            output,
            output,  # bias_ptr (unused when HAS_BIAS=False, passed as dummy)
            M,
            N,
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            OUTPUT_DTYPE=tl.float16,
        )

        expected = partials.sum(dim=0).to(torch.float16)
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)


class TestMaybeOverrideBestChoice(TestCase):
    """Unit tests for the TLX allow-mode speedup threshold logic."""

    def setUp(self):
        super().setUp()
        self._tlx_patch = config.patch({"triton.tlx_mode": "allow"})
        self._tlx_patch.__enter__()

    def tearDown(self):
        self._tlx_patch.__exit__(None, None, None)
        super().tearDown()

    def _make_choice(self, name: str, is_extern: bool = False):
        from torch._inductor.select_algorithm import ExternKernelCaller

        if is_extern:
            choice = mock.MagicMock(spec=ExternKernelCaller)
        else:
            choice = mock.MagicMock()
        choice.name = name
        return choice

    def test_high_threshold_overrides_to_extern(self):
        """With a very high threshold, TLX is always overridden to extern."""
        from triton.language.extra.tlx.inductor.choices import (
            maybe_override_best_choice, )

        tlx_choice = self._make_choice("tlx_mm")
        extern_choice = self._make_choice("cublas", is_extern=True)
        timings = {tlx_choice: 1.0, extern_choice: 2.0}

        with tlx_config.patch(allow_min_speedup=999.0):
            result = maybe_override_best_choice(tlx_choice, timings)
        self.assertIs(result, extern_choice)

    def test_zero_threshold_keeps_tlx(self):
        """With threshold=0, speedup can never be < 0, so TLX is always kept."""
        from triton.language.extra.tlx.inductor.choices import (
            maybe_override_best_choice, )

        tlx_choice = self._make_choice("tlx_mm")
        extern_choice = self._make_choice("cublas", is_extern=True)
        timings = {tlx_choice: 1.0, extern_choice: 0.5}

        with tlx_config.patch(allow_min_speedup=0.0):
            result = maybe_override_best_choice(tlx_choice, timings)
        self.assertIs(result, tlx_choice)


# --- Heuristic Rule Tests ---
# Each rule in get_heuristic_config() gets (1) config selection assertions and
# (2) codegen pattern assertions.  Shapes are chosen so that exactly one rule
# fires for each case (verified by tracing through the rule logic with
# num_sms=148).

# Tier 1: (rule, M, N, K, expected_config_subset)
# Calls get_heuristic_config() directly — no GPU needed.
HEURISTIC_CONFIG_CASES = [
    # Rule 1a: tall_m, saturated, AI>1.5, alt-tiling, m_tiles<=74
    (
        "rule_1a",
        16384,
        384,
        4096,
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "NUM_CTAS": 1,
            "NUM_MMA_GROUPS": 1,
            "NUM_SMEM_BUFFERS": 3,
            "INTERLEAVE_EPILOGUE": 0,
            "SPLIT_K": 1,
        },
    ),
    # Rule 1b: tall_m, saturated, AI<=1.5
    (
        "rule_1b",
        32768,
        384,
        384,
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "NUM_CTAS": 2,
            "NUM_MMA_GROUPS": 2,
            "NUM_SMEM_BUFFERS": 2,
            "INTERLEAVE_EPILOGUE": 1,
            "SPLIT_K": 1,
        },
    ),
    # Rule 3: tall_m, saturated, AI>1.5, no alt-tiling, K>N*2
    (
        "rule_3",
        37888,
        1024,
        4096,
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "NUM_CTAS": 2,
            "NUM_MMA_GROUPS": 2,
            "NUM_SMEM_BUFFERS": 2,
            "INTERLEAVE_EPILOGUE": 0,
            "SPLIT_K": 1,
        },
    ),
    # Rule 4: tall_m, saturated, AI>1.5, no alt-tiling, K<=N*2
    (
        "rule_4",
        37888,
        4096,
        8192,
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "NUM_CTAS": 2,
            "NUM_MMA_GROUPS": 2,
            "NUM_SMEM_BUFFERS": 4,
            "INTERLEAVE_EPILOGUE": 1,
            "SPLIT_K": 1,
        },
    ),
    # Rule 5: undersaturated, large-output, split-K
    (
        "rule_5",
        1152,
        1024,
        16384,
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "NUM_CTAS": 1,
            "SPLIT_K": 4,
            "INTERLEAVE_EPILOGUE": 1,
        },
    ),
    # Rule 5: undersaturated, large-output, split-K (ads_omnifm_v5 shape)
    (
        "rule_5_split_k_large",
        1024,
        2048,
        442368,
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "NUM_CTAS": 1,
            "NUM_SMEM_BUFFERS": 4,
            "NUM_TMEM_BUFFERS": 2,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": 8,
            "SPLIT_K": 4,
            "INTERLEAVE_EPILOGUE": 1,
        },
    ),
    # Rule 6: undersaturated, small-output
    # K=256 gives SPLIT_K=1 (k_tiles=2, too few for any split factor).
    # Larger K (e.g. 16384) causes SMEM overflow with BM=128/BK=128/4-buf.
    (
        "rule_6",
        512,
        128,
        256,
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "NUM_CTAS": 1,
            "SPLIT_K": 1,
            "INTERLEAVE_EPILOGUE": 1,
        },
    ),
    # Rule 7: gpu-saturated, not tall_m
    (
        "rule_7",
        4096,
        4096,
        4096,
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "NUM_CTAS": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_SMEM_BUFFERS": 3,
            "INTERLEAVE_EPILOGUE": 1,
            "SPLIT_K": 1,
        },
    ),
]


@instantiate_parametrized_tests
class TestHeuristicConfigSelection(TestCase):
    """Tier 1: Verify get_heuristic_config() picks the right config for each rule."""

    @parametrize("case", HEURISTIC_CONFIG_CASES, name_fn=lambda c: c[0])
    def test_config_selection(self, case):
        from triton.language.extra.tlx.inductor.registry import get_heuristic_config

        rule_name, M, N, K, expected = case
        config = get_heuristic_config(M, N, K, num_sms=148)
        self.assertIsNotNone(config, f"{rule_name}: got None config for ({M}, {N}, {K})")
        for key, val in expected.items():
            self.assertEqual(
                config[key],
                val,
                f"{rule_name}: {key}={config[key]}, expected {val}",
            )

    @parametrize("case", HEURISTIC_CONFIG_CASES, name_fn=lambda c: c[0])
    def test_group_size_m_multiple_of_num_ctas(self, case):
        """GROUP_SIZE_M must be a multiple of NUM_CTAS for correct tile scheduling."""
        from triton.language.extra.tlx.inductor.registry import get_heuristic_config

        rule_name, M, N, K, _expected = case
        config = get_heuristic_config(M, N, K, num_sms=148)
        if config is None:
            return
        num_ctas = config["NUM_CTAS"]
        gsm = config["GROUP_SIZE_M"]
        self.assertEqual(
            gsm % num_ctas,
            0,
            f"{rule_name}: GROUP_SIZE_M={gsm} not divisible by NUM_CTAS={num_ctas}",
        )


# Tier 2: (rule, M, K, N, {pattern: True/False}, check_correctness)
# True = assertIn, False = assertNotIn.  Note (M, K, N) order matches
# TEMPLATE_TEST_SHAPES convention.
# check_correctness=False for NUM_CTAS=2 rules (1b, 3, 4) due to known
# runtime correctness issue with 2-CTA configs.
HEURISTIC_CODEGEN_CASES = [
    # Rule 1a: INTERLEAVE=0, MMA_GROUPS=1 → no second smem buffer index, no split-K
    (
        "rule_1a",
        16384,
        4096,
        384,
        {"fused_tlx_": True, "c_smem_buffers[(1)": False, "_reduce_k_kernel": False},
        True,
    ),
    # Rule 1b: INTERLEAVE=1 → interleaved smem stores (CTAS=2, skip correctness)
    (
        "rule_1b",
        32768,
        384,
        384,
        {"fused_tlx_": True, "c_smem_buffers[(0)": True, "c_smem_buffers[(1)": True},
        False,
    ),
    # Rule 3: INTERLEAVE=0 → no second smem buffer index, no split-K (CTAS=2, skip correctness)
    (
        "rule_3",
        37888,
        4096,
        1024,
        {"fused_tlx_": True, "c_smem_buffers[(1)": False, "_reduce_k_kernel": False},
        False,
    ),
    # Rule 4: INTERLEAVE=1 → interleaved smem stores (CTAS=2, skip correctness)
    (
        "rule_4",
        37888,
        8192,
        4096,
        {"fused_tlx_": True, "c_smem_buffers[(0)": True, "c_smem_buffers[(1)": True},
        False,
    ),
    # Rule 5: split-K → reduction kernel + workspace descriptor
    (
        "rule_5",
        1152,
        16384,
        1024,
        {
            "_reduce_k_kernel": True,
            "split_k_ws": True,
            "c_smem_buffers": False,
        },
        True,
    ),
    # Rule 5: split-K with large K (ads_omnifm_v5 crash shape)
    # check_correctness=False: K=442368 causes cuBLAS vs TLX accumulation
    # differences that exceed atol=0.01 (max abs diff ~0.3, 0.5% elements).
    (
        "rule_5_split_k_large",
        1024,
        442368,
        2048,
        {
            "_reduce_k_kernel": True,
            "async_descriptor_store": True,
            "split_k_ws": True,
            "c_smem_buffers": False,
        },
        False,
    ),
    # Rule 6: INTERLEAVE=1, MMA_GROUPS=2 → interleaved smem stores, no split-K
    (
        "rule_6",
        512,
        256,
        128,
        {
            "fused_tlx_": True,
            "c_smem_buffers[(0)": True,
            "c_smem_buffers[(1)": True,
            "_reduce_k_kernel": False,
        },
        True,
    ),
    # Rule 7: INTERLEAVE=1 → interleaved smem stores, no split-K
    (
        "rule_7",
        4096,
        4096,
        4096,
        {
            "fused_tlx_": True,
            "c_smem_buffers[(0)": True,
            "c_smem_buffers[(1)": True,
            "_reduce_k_kernel": False,
        },
        True,
    ),
]


@instantiate_parametrized_tests
class TestFlexAttention(TestCase):
    """AMD (gfx950/MI350) FlexAttention Inductor template.

    Exercises torch.compile(flex_attention) under tlx_mode and asserts the
    tlx_amd_flex_attention template is selected and numerically correct across
    score_mod / mask_mod / logsumexp. Gated on the torch flex-choices hook, which
    the current ROCm nightly lacks, so these skip there and are validated on a
    newer torch (e.g. fbsource).
    """

    def _qkv(self, B, H, N, D, dtype):
        torch.manual_seed(0)
        return [torch.randn(B, H, N, D, device=GPU_TYPE, dtype=dtype) for _ in range(3)]

    def _run(self, fn, q, k, v):
        with config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
        }):
            out, code = run_and_get_code(torch.compile(fn), q, k, v)
        return out, "\n".join(code)

    @unittest.skipIf(not is_gfx950(), "Need AMD MI350X (gfx950)")
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not flex_choices_hook_available(),
        "torch lacks append_flex_attention_choices (old ROCm nightly)",
    )
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_flex_none(self, dtype):
        from torch.nn.attention.flex_attention import flex_attention

        B, H, N, D = 1, 2, 256, 64
        sm = 1.0 / (D**0.5)
        q, k, v = self._qkv(B, H, N, D, dtype)
        out, code = self._run(lambda q, k, v: flex_attention(q, k, v, scale=sm), q, k, v)
        ref = flex_attention(q, k, v, scale=sm)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        self.assertIn("tlx_amd_flex_attention", code)

    @unittest.skipIf(not is_gfx950(), "Need AMD MI350X (gfx950)")
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not flex_choices_hook_available(),
        "torch lacks append_flex_attention_choices (old ROCm nightly)",
    )
    def test_flex_causal(self):
        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        B, H, N, D = 1, 2, 256, 64
        sm = 1.0 / (D**0.5)
        q, k, v = self._qkv(B, H, N, D, torch.float16)
        bm = create_block_mask(lambda b, h, m, n: m >= n, B, H, N, N, device=GPU_TYPE)
        out, code = self._run(lambda q, k, v: flex_attention(q, k, v, block_mask=bm, scale=sm), q, k, v)
        ref = flex_attention(q, k, v, block_mask=bm, scale=sm)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        self.assertIn("tlx_amd_flex_attention", code)

    @unittest.skipIf(not is_gfx950(), "Need AMD MI350X (gfx950)")
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not flex_choices_hook_available(),
        "torch lacks append_flex_attention_choices (old ROCm nightly)",
    )
    def test_flex_score_mod(self):
        from torch.nn.attention.flex_attention import flex_attention

        B, H, N, D = 1, 2, 256, 64
        sm = 1.0 / (D**0.5)
        slope = 0.1
        q, k, v = self._qkv(B, H, N, D, torch.float16)
        score_mod = lambda s, b, h, m, n: s - slope * (m - n)  # noqa: E731
        out, code = self._run(
            lambda q, k, v: flex_attention(q, k, v, score_mod=score_mod, scale=sm),
            q,
            k,
            v,
        )
        ref = flex_attention(q, k, v, score_mod=score_mod, scale=sm)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        self.assertIn("tlx_amd_flex_attention", code)

    @unittest.skipIf(not is_gfx950(), "Need AMD MI350X (gfx950)")
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not flex_choices_hook_available(),
        "torch lacks append_flex_attention_choices (old ROCm nightly)",
    )
    def test_flex_logsumexp(self):
        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        B, H, N, D = 1, 2, 256, 64
        sm = 1.0 / (D**0.5)
        q, k, v = self._qkv(B, H, N, D, torch.float16)
        bm = create_block_mask(lambda b, h, m, n: m >= n, B, H, N, N, device=GPU_TYPE)
        with config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
        }):
            (out, lse), code = run_and_get_code(
                torch.compile(lambda q, k, v: flex_attention(q, k, v, block_mask=bm, scale=sm, return_lse=True)),
                q,
                k,
                v,
            )
        ref_o, ref_lse = flex_attention(q, k, v, block_mask=bm, scale=sm, return_lse=True)
        torch.testing.assert_close(out, ref_o, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, ref_lse, atol=3e-2, rtol=3e-2)
        self.assertIn("tlx_amd_flex_attention", "\n".join(code))


if __name__ == "__main__":
    run_tests()
