# Owner(s): ["module: inductor"]
import inspect
import unittest
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.utils._triton import has_datacenter_blackwell_tma_device
from triton.language.extra.tlx.inductor import tlx_config
from triton.language.extra.tlx.hw.target import current_target


def has_tlx() -> bool:
    """Check if TLX (Triton Language eXtensions) is available."""
    try:
        import triton.language.extra.tlx  # noqa: F401  # @manual

        return True
    except ImportError:
        return False


# Arch gates. All of these go through the one shared target model, so adding
# MI300X/MI450X coverage is a change to the arch classes rather than to every
# gate in the suite. With no GPU visible current_target() resolves to no arch
# at all, so each predicate is already False on a build host.


def is_gfx950() -> bool:
    """True on AMD MI350X (gfx950), where the TLX warp-pipe addmm template runs."""
    return current_target().is_gfx950


def is_blackwell() -> bool:
    """True on a Blackwell device, where the TLX WS GEMM template runs."""
    return current_target().is_blackwell


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


def flex_backward_choices_hook_available() -> bool:
    """True when torch exposes the complete backward choices-hook contract."""
    try:
        from torch._inductor.choices import InductorChoices

        parameters = inspect.signature(InductorChoices.append_flex_attention_choices).parameters
        return "mutated_inputs" in parameters
    except (AttributeError, ImportError, TypeError, ValueError):
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
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("layout", ("a_col", "b_col", "both_col"))
    def test_tlx_matmul_ws_column_major(self, layout: str):
        """A column-major operand goes through the transposed TMA descriptor path.

        A column-major (M, K) operand has strides (1, M), so its .T is a row-major
        (K, M) view of the same memory. The template describes that view, loads the
        flipped tile shape, and recovers the MMA operand with tlx.local_trans -- no
        contiguous copy. tlx_mode=force makes the TLX template the only choice, so a
        layout it cannot handle fails here rather than quietly losing autotune.
        """

        def mm(a, b):
            return torch.mm(a, b)

        # Saturated (Rule 7) so the heuristic picks SPLIT_K=1: a split-K config would
        # trip the separate reduce-k launch bug, which has nothing to do with layout.
        M, K, N = 4096, 2048, 4096
        dtype = torch.bfloat16
        a = torch.randn((M, K), dtype=dtype, device=GPU_TYPE)
        b = torch.randn((K, N), dtype=dtype, device=GPU_TYPE)
        if layout in ("a_col", "both_col"):
            a = a.t().contiguous().t()
        if layout in ("b_col", "both_col"):
            b = b.t().contiguous().t()

        with config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "enable_caching_generated_triton_templates": False,
        }):
            c_actual, code = run_and_get_code(torch.compile(mm), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=0.01, rtol=0.01)

        # Both layout branches are constexpr and so both appear in the emitted source;
        # the compiled-in flag is what says which one Triton folded to.
        code_str = "\n".join(code)
        self.assertIn("triton_tem_fused_tlx_mm", code_str)
        if layout in ("a_col", "both_col"):
            self.assertIn("A_ROW_MAJOR : tl.constexpr = False", code_str)
        if layout in ("b_col", "both_col"):
            self.assertIn("B_ROW_MAJOR : tl.constexpr = False", code_str)

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
        "Need AMD MI350X (gfx950) for the TLX inter-wave addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("k", (256, 328))
    @parametrize("column_major_b", (False, True))
    def test_tlx_addmm_interwave(
        self,
        dtype: torch.dtype,
        k: int,
        column_major_b: bool,
    ):
        """The gfx950 inter-wave template handles B layouts and M/K tails."""
        from triton.language.extra.tlx.inductor import mm_templates as _tlx_mm
        from triton.language.extra.tlx.inductor import registry as _tlx_registry

        M, K, N = 256, k, 256
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
        torch._dynamo.mark_dynamic(a, 0)
        b = (
            torch.randn(N, K, device=GPU_TYPE, dtype=dtype).t()
            if column_major_b
            else torch.randn(K, N, device=GPU_TYPE, dtype=dtype)
        )
        bias = torch.randn(N, device=GPU_TYPE, dtype=dtype)

        def addmm(bias, a, b):
            return torch.addmm(bias, a, b)

        def _only_interwave(templates, op_name="mm"):
            from torch._inductor.kernel.mm import mm_template

            uids = {getattr(t, "uid", None) for t in templates}
            if op_name == "addmm" and mm_template.uid in uids:
                template = _tlx_mm.gfx950_addmm_interwave_template
                if template.uid not in uids:
                    templates.append(template)
            return templates

        with (
            mock.patch.object(_tlx_mm, "append_tlx", _only_interwave),
            mock.patch.object(
                _tlx_registry.Gfx950AddMMInterWaveTemplateConfigHeuristic,
                "INTERWAVE_CONFIGS",
                [(256, 256, 64, 4, 8, 0, 8)],
            ),
            config.patch(
                {
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }
            ),
        ):
            compiled_addmm = torch.compile(addmm)
            c_actual, code = run_and_get_code(compiled_addmm, bias, a, b)

            # Reuse the same symbolic-M graph with a non-tile-aligned runtime M.
            a_tail = torch.randn(M + 8, K, device=GPU_TYPE, dtype=dtype)
            c_tail_actual = compiled_addmm(bias, a_tail, b)

        c_expected = (a.float() @ b.float() + bias.float()).to(dtype)
        c_tail_expected = (a_tail.float() @ b.float() + bias.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            c_tail_actual, c_tail_expected, atol=2e-2, rtol=2e-2
        )
        self.assertIn("smem_a_top", "\n".join(code))

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX inter-wave addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_tlx_addmm_interwave_symbolic_m_xcd(self):
        """The 8-XCD remap preserves the index type for a symbolic M expression."""
        from triton.language.extra.tlx.inductor import mm_templates as _tlx_mm
        from triton.language.extra.tlx.inductor import registry as _tlx_registry

        batch, rows, K, N = 32, 2000, 256, 512
        a = torch.randn(batch, rows, K, device=GPU_TYPE, dtype=torch.float16)
        torch._dynamo.mark_dynamic(a, 0)
        b = torch.randn(N, K, device=GPU_TYPE, dtype=torch.float16).t()
        bias = torch.randn(N, device=GPU_TYPE, dtype=torch.float16)

        def addmm(bias, a, b):
            return torch.addmm(bias, a.flatten(0, 1), b)

        def _add_interwave(templates, op_name="mm"):
            from torch._inductor.kernel.mm import mm_template

            uids = {getattr(t, "uid", None) for t in templates}
            if op_name == "addmm" and mm_template.uid in uids:
                template = _tlx_mm.gfx950_addmm_interwave_template
                if template.uid not in uids:
                    templates.append(template)
            return templates

        with (
            mock.patch.object(_tlx_mm, "append_tlx", _add_interwave),
            mock.patch.object(
                _tlx_registry.Gfx950AddMMInterWaveTemplateConfigHeuristic,
                "INTERWAVE_CONFIGS",
                [(256, 256, 64, 4, 8, 0, 8)],
            ),
            config.patch(
                {
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "autotune_fallback_to_aten": False,
                    "test_configs.autotune_choice_name_regex": (
                        "tlx_gfx950_addmm_interwave"
                    ),
                    "enable_caching_generated_triton_templates": False,
                }
            ),
        ):
            c_actual, code = run_and_get_code(torch.compile(addmm), bias, a, b)

        c_expected = addmm(bias, a, b)
        torch.testing.assert_close(c_actual, c_expected, atol=2e-2, rtol=2e-2)
        self.assertIn("smem_a_top", "\n".join(code))

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX inter-wave addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("shape", ((1024, 864, 1024), (64000, 256, 256)))
    def test_tlx_addmm_interwave_real_autotune(self, shape: tuple[int, int, int]):
        """The registered inter-wave choice competes on production B layouts."""
        from torch._inductor.select_algorithm import TritonTemplateCaller
        from triton.language.extra.tlx.inductor import registry as _tlx_registry  # noqa: F401

        M, K, N = shape
        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(N, K, device=GPU_TYPE, dtype=torch.float16).t()
        bias = torch.randn(N, device=GPU_TYPE, dtype=torch.float16)

        benchmarked = []
        benchmark = TritonTemplateCaller.benchmark

        def record_benchmark(choice, *args, out):
            benchmarked.append(choice.name)
            return benchmark(choice, *args, out=out)

        with (
            mock.patch.object(TritonTemplateCaller, "benchmark", record_benchmark),
            config.patch(
                {
                    "triton.tlx_mode": "allow",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "ATEN,TRITON",
                    "enable_caching_generated_triton_templates": False,
                }
            ),
        ):
            c_actual, code = run_and_get_code(
                torch.compile(torch.addmm), bias, a, b
            )

        c_expected = torch.addmm(bias, a, b)
        torch.testing.assert_close(c_actual, c_expected, atol=2e-2, rtol=2e-2)
        self.assertTrue(code)
        self.assertTrue(
            any("tlx_gfx950_addmm_interwave" in name for name in benchmarked),
            benchmarked,
        )
        self.assertTrue(
            any(name.startswith("triton_mm") for name in benchmarked), benchmarked
        )

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
    def test_tlx_addmm_warppipe_split_k_cpp_wrapper(self):
        M, K, N = 256, 4096, 256
        dtype = torch.float16
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
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
                "cpp_wrapper": True,
        }), ):
            c_actual, code = run_and_get_code(torch.compile(addmm), bias, a, w)

        c_expected = (a.float() @ w.t().float() + bias.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=3e-2, rtol=3e-2)

        code_str = "\n".join(code)
        self.assertIn("split_k_ws", code_str)
        # Every split-K reducer is now code-generated via define_kernel, even with no
        # fused epilogue (identity epilogue), so the define_user_defined_triton_kernel
        # reducer is gone from the forward.
        self.assertIn("_reduce_k", code_str)
        self.assertNotIn("call__reduce_k_kernel_", code_str)
        self.assertNotIn("from triton.language.extra.tlx.inductor.reduce_k import", code_str)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_tlx_addmm_warppipe_split_k_python_epilogue_fused(self):
        """Split-K + pointwise epilogue under the Python/JIT wrapper.

        The generated reducer replays the fused epilogue in both wrappers, so the
        epilogue runs after the partials are summed and no separate pointwise kernel
        is emitted.
        """
        M, K, N = 256, 4096, 256
        dtype = torch.float16
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
        w = torch.randn(N, K, device=GPU_TYPE, dtype=dtype)
        bias = torch.randn(N, device=GPU_TYPE, dtype=dtype)

        def addmm_gelu(bias, a, w):
            return torch.nn.functional.gelu(torch.addmm(bias, a, w.t()))

        with config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "enable_caching_generated_triton_templates": False,
                "cpp_wrapper": False,
        }):
            c_actual, code = run_and_get_code(torch.compile(addmm_gelu), bias, a, w)

        c_expected = torch.nn.functional.gelu(a.float() @ w.t().float() + bias.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=3e-2, rtol=3e-2)

        code_str = "\n".join(code)
        self.assertIn("_reduce_k", code_str)
        self.assertNotIn("triton_poi_", code_str)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("epilogue", ("gelu", "mul"))
    def test_tlx_addmm_warppipe_split_k_cpp_wrapper_fused_epilogue(self, epilogue: str):
        M, K, N = 256, 4096, 256
        dtype = torch.float16
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
        w = torch.randn(N, K, device=GPU_TYPE, dtype=dtype)
        bias = torch.randn(N, device=GPU_TYPE, dtype=dtype)

        def addmm_epilogue(bias, a, w):
            out = torch.addmm(bias, a, w.t())
            if epilogue == "gelu":
                return torch.nn.functional.gelu(out)
            return out * 0.5

        with (config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "enable_caching_generated_triton_templates": False,
                "cpp_wrapper": True,
        }), ):
            c_actual, code = run_and_get_code(torch.compile(addmm_epilogue), bias, a, w)

        reference = a.float() @ w.t().float() + bias.float()
        if epilogue == "gelu":
            reference = torch.nn.functional.gelu(reference)
        else:
            reference = reference * 0.5
        torch.testing.assert_close(c_actual, reference.to(dtype), atol=4e-2, rtol=4e-2)

        code_str = "\n".join(code)
        self.assertIn("split_k_ws", code_str)
        self.assertIn("_reduce_k", code_str)
        self.assertNotIn("triton_poi_", code_str)

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
    def test_tlx_addmm_warppipe_regpath_odd_k(self):
        """Odd K on the TLX addmm template -> register-path branch (USE_ASYNC=0), T280910119.

        K=2309 (odd) has a 2309*2 = 4618 B row stride that is never 16-byte aligned, so the
        direct-to-LDS async_load path cannot legalize on CDNA4 (col-major B's async_copy into the
        swizzled #ttg.padded_shared LDS layout). The heuristic sets USE_ASYNC=0 and the template
        takes the register-path fallback (tl.load -> tl.dot, auto-pipelined), which lowers for ANY
        alignment -- same mechanism as the bmm template. In tlx_mode=force (TRITON-only) this must
        lower to the Triton template (never extern/aten) and be numerically correct. (Previously
        this shape declined -> NoValidChoicesError; the addmm register-path fallback fixed it.)
        """
        M, K, N = 4096, 2309, 192  # odd K -> 4618 B row stride, not 16-byte aligned -> register path
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
        }):
            c_actual, code = run_and_get_code(torch.compile(addmm), bias, a, w)

        c_expected = (a.float() @ w.t().float() + bias.float()).to(torch.float16)
        torch.testing.assert_close(c_actual, c_expected, atol=2e-2, rtol=2e-2)
        # force mode keeps only the TLX template -> odd-K addmm must lower via the register branch.
        self.assertIn("triton_tem", "\n".join(code))

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe bmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_tlx_bmm_warppipe(self, dtype: torch.dtype):
        """TLX warp-pipelined bmm template (AMD MI350X / gfx950).

        Batched C[b] = A[b] @ B[b], standard torch.bmm layout (B [batch,K,N] row-major, no
        transpose). Same warp-pipe core as the addmm + a batch axis + per-batch int64 base
        advance. K=272 is a multiple of 16 (the heuristic's alignment gate) but of no BLOCK_K in
        the config set, so every config exercises the sync-tail. Verifies the TLX bmm lowers to a
        Triton template (never extern/aten in force mode) and is numerically correct.
        """
        batch, M, K, N = 8, 256, 272, 256  # K=272 = 16*17: passes the ÷16 gate, exercises the tail
        a = torch.randn(batch, M, K, device=GPU_TYPE, dtype=dtype)
        b = torch.randn(batch, K, N, device=GPU_TYPE, dtype=dtype)

        def bmm(a, b):
            return torch.bmm(a, b)

        with (config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "enable_caching_generated_triton_templates": False,
        }), ):
            c_actual, code = run_and_get_code(torch.compile(bmm), a, b)

        c_expected = torch.bmm(a.float(), b.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=2e-2, rtol=2e-2)

        code_str = "\n".join(code)
        self.assertIn("triton_tem", code_str)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe bmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_tlx_bmm_warppipe_regpath_odd_k(self, dtype: torch.dtype):
        """Odd K on the TLX bmm template -> register-path branch (USE_ASYNC=0), T280910119.

        K=2309 (odd; the production compression bmm's K) has a 2309*2 = 4618 B row stride that is
        never 16-byte aligned, so the direct-to-LDS async_load path cannot legalize on CDNA4. The
        heuristic sets USE_ASYNC=0 and the template takes the register-path fallback (tl.load ->
        registers -> tl.dot, auto-pipelined), which lowers for ANY alignment. In tlx_mode=force
        (TRITON-only) this must still lower to the Triton template (never extern/aten) and be
        numerically correct -- exactly the shape the aligned-K async path could not compile.
        """
        batch, M, K, N = 8, 256, 2309, 256  # odd K -> register-path branch (USE_ASYNC=0)
        a = torch.randn(batch, M, K, device=GPU_TYPE, dtype=dtype)
        b = torch.randn(batch, K, N, device=GPU_TYPE, dtype=dtype)

        def bmm(a, b):
            return torch.bmm(a, b)

        with (config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "enable_caching_generated_triton_templates": False,
        }), ):
            c_actual, code = run_and_get_code(torch.compile(bmm), a, b)

        c_expected = torch.bmm(a.float(), b.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=2e-2, rtol=2e-2)

        code_str = "\n".join(code)
        self.assertIn("triton_tem", code_str)

    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_tlx_bmm_warppipe_dual_path_source(self):
        """The non-persistent bmm template carries BOTH branches: the async_load direct-to-LDS path
        (aligned K) and the register-path fallback (unaligned/odd K, T280910119). Source check --
        no GPU needed."""
        from triton.language.extra.tlx.inductor.mm_templates import load_tlx_template

        src = load_tlx_template("gfx950_bmm_warppipe")
        self.assertIn("USE_ASYNC", src)  # dual-path selector constexpr
        self.assertIn("tlx.async_load", src)  # aligned-K async path
        self.assertIn("a_reg = tl.load", src)  # register-path fallback load

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX persistent warp-pipe addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_tlx_addmm_persistent_warppipe(self, dtype: torch.dtype):
        """Persistent TLX warp-pipe addmm template (AMD MI350X / gfx950), col-major B.

        De-risks the persistent tile-loop + tlx.warp_pipeline_stage nesting (the exact
        pattern that failed to lower on beta triton 3.6). ``append_tlx`` is patched to
        offer ONLY the persistent template, so force mode is guaranteed to select and
        compile it (not the per-tile warp-pipe) -- proving it lowers and is numerically
        correct. The shape has many more MN tiles than SMs, so each program's persistent
        loop iterates over several output tiles.
        """
        from triton.language.extra.tlx.inductor import mm_templates as _tlx_mm

        M, K, N = 4096, 2048, 512  # many MN tiles -> persistent loop iterates per program
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
        # w.t() => B is [K, N] col-major (stride_bk == 1) -- the nn.Linear weight layout.
        w = torch.randn(N, K, device=GPU_TYPE, dtype=dtype)
        bias = torch.randn(N, device=GPU_TYPE, dtype=dtype)

        def addmm(bias, a, w):
            return torch.addmm(bias, a, w.t())

        def _only_persistent(templates, op_name="mm"):
            # Offer only the persistent template (drop the per-tile warp-pipe) so force
            # mode is guaranteed to select and compile the persistent kernel.
            from torch._inductor.kernel.mm import mm_template

            uids = {getattr(t, "uid", None) for t in templates}
            if op_name == "addmm" and mm_template.uid in uids:
                if _tlx_mm.gfx950_addmm_persistent_warppipe_template.uid not in uids:
                    templates.append(_tlx_mm.gfx950_addmm_persistent_warppipe_template)
            return templates

        with (
                mock.patch.object(_tlx_mm, "append_tlx", _only_persistent),
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }),
        ):
            c_actual, code = run_and_get_code(torch.compile(addmm), bias, a, w)

        c_expected = (a.float() @ w.t().float() + bias.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=2e-2, rtol=2e-2)

        # Only the persistent template is offered, so force mode must lower addmm
        # through it (triton_tem), never an extern/aten kernel.
        code_str = "\n".join(code)
        self.assertIn("triton_tem", code_str)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX persistent warp-pipe addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("K", (1024, 1032))
    def test_tlx_addmm_persistent_warppipe_split_k(self, K: int):
        """Persistent split-K handles complete and partial final K tiles."""
        from triton.language.extra.tlx.inductor import mm_templates as _tlx_mm
        from triton.language.extra.tlx.inductor import registry as _tlx_registry

        # With the one 64x64 tile config this is 136 output tiles. SPLIT_K=2 creates
        # 272 work items on 256 CUs, so the first 16 programs execute the persistent
        # loop twice and exercise its cross-work-item LDS reuse. K=1024 covers full
        # BLOCK_K tiles; K=1032 is fp16 16-byte aligned but leaves an 8-element tail.
        M, N = 1088, 512
        dtype = torch.float16
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
        w = torch.randn(N, K, device=GPU_TYPE, dtype=dtype)
        bias = torch.randn(N, device=GPU_TYPE, dtype=dtype)

        def addmm(bias, a, w):
            return torch.addmm(bias, a, w.t())

        def _only_persistent(templates, op_name="mm"):
            from torch._inductor.kernel.mm import mm_template

            uids = {getattr(t, "uid", None) for t in templates}
            if op_name == "addmm" and mm_template.uid in uids:
                if _tlx_mm.gfx950_addmm_persistent_warppipe_template.uid not in uids:
                    templates.append(_tlx_mm.gfx950_addmm_persistent_warppipe_template)
            return templates

        heuristic = _tlx_registry.Gfx950AddMMPersistentWarpPipeConfigHeuristic
        get_configs = heuristic._get_template_configs_impl

        def _split_k_two_only(instance, kernel_inputs, op_name):
            for template_kwargs in get_configs(instance, kernel_inputs, op_name):
                if template_kwargs.get("SPLIT_K") == 2:
                    yield template_kwargs

        with (
                mock.patch.object(_tlx_mm, "append_tlx", _only_persistent),
                mock.patch.object(
                    heuristic,
                    "WARPPIPE_CONFIGS",
                    [(64, 64, 64, 8, 8, 3)],
                ),
                mock.patch.object(
                    heuristic,
                    "_get_template_configs_impl",
                    _split_k_two_only,
                ),
                mock.patch.dict(
                    _tlx_registry.os.environ,
                    {"TORCHINDUCTOR_TLX_SPLIT_K": "1"},
                ),
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }),
        ):
            c_actual, code = run_and_get_code(torch.compile(addmm), bias, a, w)

        c_expected = (a.float() @ w.t().float() + bias.float()).to(dtype)
        torch.testing.assert_close(c_actual, c_expected, atol=3e-2, rtol=3e-2)

        code_str = "\n".join(code)
        self.assertIn("split_k_ws", code_str)
        self.assertIn("_reduce_k", code_str)


class TestInterWaveTemplateCodegen(TestCase):
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_interwave_template_renders_four_quadrants(self):
        import jinja2
        from triton.language.extra.tlx.inductor.mm_templates import load_tlx_template

        source = load_tlx_template("gfx950_addmm_interwave")
        stores = []

        def store_output(indices, val, mask, **kwargs):
            stores.append((indices, val, mask, kwargs["val_shape"]))
            return f"tl.store(A, {val}, mask={mask})"

        hooks = {
            "def_kernel": lambda *args, **kwargs: "def _kernel(A, B):",
            "size": lambda *args, **kwargs: "256",
            "stride": lambda *args, **kwargs: "1",
            "store_output": store_output,
        }
        rendered = jinja2.Environment().from_string(source).render(**hooks)

        compile(rendered, "<gfx950_addmm_interwave>", "exec")
        self.assertEqual(4, len(stores))
        self.assertCountEqual(
            ["acc_tl", "acc_bl", "acc_tr", "acc_br"],
            [store[1] for store in stores],
        )


class TestWarpPipeSplitKCodegen(TestCase):
    """Deterministic codegen checks for the AMD warp-pipe split-K templates.

    The per-tile e2e test relies on autotune *selecting* a SPLIT_K > 1 config. These
    tests render both warp-pipe Jinja templates directly with SPLIT_K=2 vs SPLIT_K=1,
    so their split-K interfaces are covered without a GPU or autotune timing decision.
    """

    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_warppipe_split_k_template_render(self):
        import jinja2
        from triton.language.extra.tlx.inductor.mm_templates import load_tlx_template

        source = load_tlx_template("gfx950_addmm_warppipe")

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
        split = tmpl.render(SPLIT_K=2, USE_ASYNC=True, **hooks)
        nosplit = tmpl.render(SPLIT_K=1, USE_ASYNC=True, **hooks)

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

    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_persistent_warppipe_split_k_template_render(self):
        import jinja2
        from triton.language.extra.tlx.inductor.mm_templates import load_tlx_template

        source = load_tlx_template("gfx950_addmm_persistent_warppipe")
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

        self.assertIn("num_work_items = num_tiles * SPLIT_K", split)
        self.assertIn("split_id = (work_id % SPLIT_K)", split)
        self.assertIn("base = K_ITERS // SPLIT_K", split)
        self.assertIn("k_lo = split_id * base", split)
        self.assertIn("tl.store(split_k_ws + ws_off, acc", split)

        self.assertNotIn("split_id = (work_id % SPLIT_K)", nosplit)
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
        """Verify the split-K epilogue is fused into the generated reducer."""

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
        # Reduction kernel present (split-K was used). The generated reducer is
        # named "{main_kernel}_reduce_k"; the legacy "_reduce_k_kernel" symbol is
        # only emitted on the no-epilogue path.
        self.assertIn("_reduce_k", code_str)
        # relu is fused into the reducer, so no separate pointwise kernel
        self.assertNotIn("triton_poi_", code_str)

    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_split_k_template_keeps_output_ptr(self):
        """The SPLIT_K > 1 branch must reference output_ptr(). No GPU needed.

        Split-K writes fp32 partials to split_k_ws and never calls store_output(), so
        without an explicit output_ptr() reference Inductor prunes out_ptr0 from the
        kernel signature. The autotune benchmark harness still passes `out`
        positionally, so the trailing argument lands on the launcher's `stream`
        parameter and every SPLIT_K > 1 candidate dies with "too many positional
        arguments ... 'stream' must be passed as a keyword argument" before it can be
        measured. Rendering the jinja directly pins the reference regardless of which
        config autotune happens to select.
        """
        import jinja2
        from triton.language.extra.tlx.inductor.mm_templates import load_tlx_template

        source = load_tlx_template("blackwell_gemm_ws")

        # Count output_ptr() invocations rather than grepping the rendered text: it is
        # calling the hook, not the spelling of the emitted line, that registers the
        # buffer in the kernel signature.
        output_ptr_calls = []

        def output_ptr(*a, **k):
            output_ptr_calls.append(1)
            return "out_ptr0"

        hooks = {
            "def_kernel": lambda *a, **k: "def _kernel(A, B, out_ptr0):",
            "size": lambda *a, **k: "0",
            "stride": lambda *a, **k: "1",
            "output_ptr": output_ptr,
            "store_output": lambda *a, **k: "# store_output(...)",
            "compute_epilogue": lambda *a, **k: "# compute_epilogue(...)",
        }
        tmpl = jinja2.Environment().from_string(source)
        common = dict(TMA_EPILOGUE_STORE=0, INTERLEAVE_EPILOGUE=0, **hooks)

        split = tmpl.render(SPLIT_K=4, **common)
        self.assertIn("split_k_ws", split)
        self.assertTrue(output_ptr_calls, "SPLIT_K > 1 must reference output_ptr()")

        # Negative control: the plain data-parallel path stores through store_output,
        # which registers the output buffer on its own, so output_ptr() is not needed.
        output_ptr_calls.clear()
        nosplit = tmpl.render(SPLIT_K=1, **common)
        self.assertFalse(output_ptr_calls)
        self.assertNotIn("split_k_ws", nosplit)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_split_k_candidate_survives_autotune_in_allow_mode(self):
        """A SPLIT_K > 1 candidate must survive being stood up by the benchmark harness.

        The other split-K tests here run in force mode, where the heuristic config is
        the only choice and autotuning short-circuits -- so the candidate is never
        handed to the benchmark harness and its arg-count mismatch stays invisible.
        allow mode makes it compete against the autotune pool, which is what every
        undersaturated large-K shape does in practice, and is the path that used to
        raise "'stream' must be passed as a keyword argument".
        """
        from triton.language.extra.tlx.inductor.registry import get_heuristic_config

        def mm(a, b):
            return torch.mm(a, b)

        # 8 MN tiles (128x64) on 148 SMs -> deeply undersaturated, and K=8192 leaves
        # each of 4 splits well above the 4-K-tile floor, so Rule 6 picks SPLIT_K > 1.
        M, K, N = 256, 8192, 256
        heuristic = get_heuristic_config(M, N, K)
        # Guard against the heuristic drifting and silently turning this into a
        # SPLIT_K=1 test that no longer covers the split-K benchmark path.
        self.assertGreater(heuristic.get("SPLIT_K", 1) if heuristic else 1, 1)

        a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE)

        with (
                config.patch({
                    "triton.tlx_mode": "allow",
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "force_disable_caches": True,
                    "enable_caching_generated_triton_templates": False,
                }),
                tlx_config.patch(use_heuristic_config=True, ),
        ):
            c_actual = torch.compile(mm)(a, b)

        torch.testing.assert_close(c_actual, mm(a, b), atol=0.01, rtol=0.01)


class TestReduceKKernel(TestCase):
    """Direct unit tests for the split-K reduction kernel."""

    def test_aoti_reduce_k_emission(self):
        from torch._inductor.virtualized import V
        from triton.language.extra.tlx.inductor.reduce_k import (
            emit_aoti_reduce_k_call, )

        wrapper = mock.MagicMock()
        wrapper.src_to_kernel = {}

        workspace = mock.MagicMock()
        workspace.inner_name = "split_k_ws"
        output = mock.MagicMock()
        output.get_device.return_value = torch.device(GPU_TYPE)

        def _argdef(name):
            arg = mock.MagicMock()
            arg.full_name.return_value = name
            return arg

        template_kernel = mock.MagicMock()
        template_kernel.args.python_argdefs.return_value = (
            [_argdef("split_k_ws"), _argdef("out_ptr0")],
            ["split_k_ws_0", "buf_out"],
            None,
            [torch.float32, torch.float16],
        )
        template_kernel.gen_common_triton_imports.return_value = "import triton"
        template_kernel.jit_lines.return_value = "@triton.jit"
        template_kernel.triton_meta = {"signature": {}}

        graph = mock.MagicMock()
        graph.get_current_device_or_throw.return_value = torch.device(GPU_TYPE)

        with V.set_graph_handler(graph):
            emit_aoti_reduce_k_call(
                wrapper,
                workspace_arg=workspace,
                output_node=output,
                bias_node=None,
                M=256,
                N=256,
                split_k=4,
                output_triton_dtype="tl.float16",
                template_kernel=template_kernel,
                main_kernel_name="triton_tem_fused_addmm_0",
                final_output_ptr="out_ptr0",
            )

        reduce_name = "triton_tem_fused_addmm_0_reduce_k"
        # The reducer is code-generated through define_kernel; the old
        # define_user_defined_triton_kernel path no longer exists.
        wrapper.define_user_defined_triton_kernel.assert_not_called()
        wrapper.define_kernel.assert_called_once()
        self.assertEqual(reduce_name, wrapper.define_kernel.call_args.args[0])

        body = wrapper.define_kernel.call_args.args[1]
        self.assertIn(f"def {reduce_name}(split_k_ws, out_ptr0):", body)
        # No fused epilogue -> identity epilogue, so the reducer still does sum + store.
        self.assertIn("acc += partial", body)
        self.assertIn("fused_result = acc", body)
        self.assertIn("tl.store(out_ptr0 + base_offs, fused_result", body)

        wrapper.generate_kernel_call.assert_called_once()
        call = wrapper.generate_kernel_call.call_args
        self.assertEqual(reduce_name, call.args[0])
        # Reuses the main template's runtime args, then the reducer grid.
        self.assertEqual(["split_k_ws_0", "buf_out"], call.args[1][:2])
        self.assertEqual(3, len(call.args[1]) - 2)
        self.assertTrue(call.kwargs["triton"])

    def test_fused_reduce_k_source(self):
        from triton.language.extra.tlx.inductor.reduce_k import _reduce_k_body_source

        source = _reduce_k_body_source(
            "split_k_ws",
            "out_ptr",
            "M",
            "N",
            4,
            "    fused_result = tl.libdevice.tanh(acc)",
        )
        self.assertLess(source.index("acc += partial"), source.index("fused_result"))
        self.assertLess(source.index("fused_result"), source.index("tl.store"))
        self.assertIn("tl.store(out_ptr + base_offs, fused_result", source)

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

# (rule, M, N, K, expected_config_subset)
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
    """Verify get_heuristic_config() picks the right config for each rule."""

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


class TestFlexAttentionChoiceRegistration(TestCase):
    """Host-only contracts for the forward/backward TLX choices hook."""

    class _Node:

        def __init__(self, size, dtype=torch.bfloat16, device="cuda:0"):
            self._size = list(size)
            self._dtype = dtype
            self._device = torch.device(device)

        def get_size(self):
            return self._size

        def get_dtype(self):
            return self._dtype

        def get_device(self):
            return self._device

    class _Template:

        def maybe_append_choice(self, choices, **kwargs):
            choices.append(kwargs)

    class _RejectingTemplate:

        def maybe_append_choice(self, choices, **kwargs):
            pass

    @staticmethod
    def _forward_inputs(dtype=torch.bfloat16, head_dim=128):
        query = TestFlexAttentionChoiceRegistration._Node([2, 8, 512, head_dim], dtype)
        return [query, object(), object(), object(), object()] + [object() for _ in range(4)]

    @staticmethod
    def _forward_config():
        return type(
            "ForwardConfig",
            (),
            {
                "block_m": 128,
                "block_n": 128,
                "num_warps": 4,
                "num_stages": 1,
            },
        )()

    @staticmethod
    def _backward_inputs(dtype=torch.bfloat16, head_dim=128):
        q = TestFlexAttentionChoiceRegistration._Node([2, 8, 512, head_dim], dtype)
        k = TestFlexAttentionChoiceRegistration._Node([2, 1, 512, head_dim], dtype)
        v = TestFlexAttentionChoiceRegistration._Node([2, 1, 512, head_dim], dtype)
        return [q, k, v] + [object() for _ in range(13)]

    @staticmethod
    def _backward_config():
        return type(
            "BackwardConfig",
            (),
            {
                "block_m1": 32,
                "block_n1": 128,
                "block_m2": 128,
                "block_n2": 32,
                "num_warps": 4,
                "num_stages": 2,
            },
        )()

    @staticmethod
    def _stock_backward_config():
        """Representative PyTorch default before its gfx950 config change."""
        return type(
            "StockBackwardConfig",
            (),
            {
                "block_m1": 64,
                "block_n1": 128,
                "block_m2": 128,
                "block_n2": 64,
                "num_warps": 8,
                "num_stages": 1,
            },
        )()

    def test_flex_backward_registration_does_not_depend_on_stock_config(self):
        """The reviewed TLX tile must be offered without PyTorch config support."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        choices = ["stock"]
        with (
                config.patch({"triton.tlx_mode": "force"}),
                mock.patch.object(
                    flex_attention_templates,
                    "_use_amd_flex_template",
                    return_value=True,
                ),
                mock.patch.object(
                    flex_attention_templates,
                    "gfx950_flex_attention_backward_template",
                    self._Template(),
                ),
        ):
            flex_attention_templates.append_tlx_flex(
                choices,
                [self._stock_backward_config()],
                self._backward_inputs(),
                [object(), object(), object(), []],
                object(),
                {},
                128,
                128,
                mutated_inputs=[],
            )

        self.assertEqual(1, len(choices))
        self.assertEqual(32, choices[0]["BLOCK_M1"])
        self.assertEqual(128, choices[0]["BLOCK_N1"])
        self.assertEqual(128, choices[0]["BLOCK_M2"])
        self.assertEqual(32, choices[0]["BLOCK_N2"])
        self.assertEqual(4, choices[0]["num_warps"])

    def test_flex_forward_force_replaces_stock_only_after_choice_appends(self):
        """Rejected or absent forward candidates must retain stock choices."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        for is_amd in (False, True):
            for case, template, configs, keeps_stock in (
                (
                    "rejected",
                    self._RejectingTemplate(),
                    [self._forward_config()],
                    True,
                ),
                ("empty", self._Template(), [], True),
                (
                    "accepted",
                    self._Template(),
                    [self._forward_config()],
                    False,
                ),
            ):
                with self.subTest(is_amd=is_amd, case=case):
                    choices = ["stock"]
                    with (
                            config.patch({"triton.tlx_mode": "force"}),
                            mock.patch.object(
                                flex_attention_templates,
                                "_use_amd_flex_template",
                                return_value=is_amd,
                            ),
                            mock.patch.object(
                                flex_attention_templates,
                                "gfx950_flex_attention_template",
                                template,
                            ),
                            mock.patch.object(
                                flex_attention_templates,
                                "blackwell_flex_attention_template",
                                template,
                            ),
                            mock.patch.object(
                                flex_attention_templates,
                                "current_target",
                                return_value=type("Target", (), {"num_sms": 256})(),
                            ),
                    ):
                        flex_attention_templates.append_tlx_flex(
                            choices,
                            configs,
                            self._forward_inputs(),
                            [object(), object()],
                            object(),
                            {},
                            128,
                            128,
                        )

                    if keeps_stock:
                        self.assertEqual(["stock"], choices)
                    else:
                        self.assertEqual(1, len(choices))
                        self.assertIsInstance(choices[0], dict)

    def test_flex_backward_registration_honors_effective_bwd_overrides(self):
        """Compatible user overrides must take precedence over the stock config."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        choices = ["stock"]
        with (
                config.patch({"triton.tlx_mode": "force"}),
                mock.patch.object(
                    flex_attention_templates,
                    "_use_amd_flex_template",
                    return_value=True,
                ),
                mock.patch.object(
                    flex_attention_templates,
                    "gfx950_flex_attention_backward_template",
                    self._Template(),
                ),
        ):
            flex_attention_templates.append_tlx_flex(
                choices,
                [self._stock_backward_config()],
                self._backward_inputs(),
                [object(), object(), object(), []],
                object(),
                {
                    "bwd_BLOCK_M1": 32,
                    "bwd_BLOCK_N1": 128,
                    "bwd_BLOCK_M2": 128,
                    "bwd_BLOCK_N2": 32,
                    "bwd_num_warps": 4,
                },
                128,
                128,
                mutated_inputs=[],
            )

        self.assertEqual(1, len(choices))
        self.assertEqual(4, choices[0]["num_warps"])

    def test_flex_backward_registration_emits_effective_gfx950_backend_options(self):
        """Advertised options must match what the gfx950 backend compiles."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        choices = ["stock"]
        with (
                config.patch({"triton.tlx_mode": "force"}),
                mock.patch.object(
                    flex_attention_templates,
                    "_use_amd_flex_template",
                    return_value=True,
                ),
                mock.patch.object(
                    flex_attention_templates,
                    "gfx950_flex_attention_backward_template",
                    self._Template(),
                ),
        ):
            flex_attention_templates.append_tlx_flex(
                choices,
                [self._backward_config()],
                self._backward_inputs(),
                [object(), object(), object(), []],
                object(),
                {},
                128,
                128,
                mutated_inputs=[],
            )

        self.assertEqual(1, len(choices))
        choice = choices[0]
        self.assertEqual(16, choice["matrix_instr_nonkdim"])
        self.assertEqual(1, choice["kpack"])
        self.assertEqual(0, choice["waves_per_eu"])

    def test_flex_backward_registration_preserves_full_mutation_contract(self):
        """Preserve captured mutations and deduplicate equivalent choices."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        inputs = self._backward_inputs()
        mutated_inputs = [inputs[6], inputs[7], object()]
        choices = ["stock"]
        with (
                config.patch({"triton.tlx_mode": "force"}),
                mock.patch.object(
                    flex_attention_templates,
                    "_use_amd_flex_template",
                    return_value=True,
                ),
                mock.patch.object(
                    flex_attention_templates,
                    "gfx950_flex_attention_backward_template",
                    self._Template(),
                    create=True,
                ),
        ):
            result = flex_attention_templates.append_tlx_flex(
                choices,
                [self._backward_config(), self._backward_config()],
                inputs,
                [object(), object(), object(), []],
                object(),
                {
                    "fwd_BLOCK_M": 64,
                    "bwd_BLOCK_M1": 32,
                    "BLOCKS_ARE_CONTIGUOUS": False,
                },
                128,
                128,
                mutated_inputs=mutated_inputs,
            )

        self.assertIs(result, choices)
        self.assertEqual(1, len(choices))
        choice = choices[0]
        self.assertIs(choice["input_nodes"], inputs)
        self.assertIs(choice["mutated_inputs"], mutated_inputs)
        self.assertEqual([2, 8, 512, 128, 1, 512], choice["call_sizes"])
        self.assertEqual(1, choice["num_stages"])
        self.assertEqual(4, choice["num_warps"])
        self.assertFalse(choice["USE_TMA"])
        self.assertEqual(32, choice["BLOCK_M1"])
        self.assertEqual(128, choice["BLOCK_N1"])
        self.assertEqual(128, choice["BLOCK_M2"])
        self.assertEqual(32, choice["BLOCK_N2"])
        self.assertNotIn("fwd_BLOCK_M", choice)
        self.assertNotIn("bwd_BLOCK_M1", choice)

    def test_flex_backward_registration_requires_complete_mutation_contract(self):
        """Missing captured mutations must retain the stock fallback."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        choices = ["stock"]
        with (
                config.patch({"triton.tlx_mode": "force"}),
                mock.patch.object(
                    flex_attention_templates,
                    "_use_amd_flex_template",
                    return_value=True,
                ),
                mock.patch.object(
                    flex_attention_templates,
                    "gfx950_flex_attention_backward_template",
                    self._Template(),
                ),
        ):
            flex_attention_templates.append_tlx_flex(
                choices,
                [self._backward_config()],
                self._backward_inputs(),
                [object(), object(), object(), []],
                object(),
                {},
                128,
                128,
                mutated_inputs=None,
            )

        self.assertEqual(["stock"], choices)

    def test_flex_backward_arch_uses_shared_exact_gfx950_target(self):
        from triton.language.extra.tlx.inductor import flex_attention_templates

        for key, expected in (
            ("gfx950", True),
            ("gfx942", False),
            ("gfx1250", False),
            ("sm100", False),
            ("", False),
        ):
            with (
                    self.subTest(key=key),
                    mock.patch.object(
                        flex_attention_templates,
                        "current_target",
                        return_value=type("Target", (), {"key": key})(),
                    ),
            ):
                self.assertEqual(
                    expected,
                    flex_attention_templates._use_amd_flex_template(),
                )

    def test_flex_backward_registration_keeps_stock_for_unsupported_input(self):
        """Force mode must not clear the only choice when TLX is ineligible."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        for dtype, head_dim in (
            (torch.float16, 128),
            (torch.bfloat16, 96),
        ):
            with self.subTest(dtype=dtype, head_dim=head_dim):
                choices = ["stock"]
                with (
                        config.patch({"triton.tlx_mode": "force"}),
                        mock.patch.object(
                            flex_attention_templates,
                            "_use_amd_flex_template",
                            return_value=True,
                        ),
                        mock.patch.object(
                            flex_attention_templates,
                            "gfx950_flex_attention_backward_template",
                            self._Template(),
                            create=True,
                        ),
                ):
                    result = (flex_attention_templates.append_tlx_flex(
                        choices,
                        [self._backward_config()],
                        self._backward_inputs(dtype, head_dim),
                        [object(), object(), object(), []],
                        object(),
                        {},
                        128,
                        128,
                        mutated_inputs=[],
                    ))
                self.assertIs(result, choices)
                self.assertEqual(["stock"], choices)

    def test_flex_backward_registration_rejects_16_row_lds_transpose(self):
        """The gfx950 TLX local-transpose inference cannot represent 16 rows."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        choices = ["stock"]
        with (
                config.patch({"triton.tlx_mode": "force"}),
                mock.patch.object(
                    flex_attention_templates,
                    "_use_amd_flex_template",
                    return_value=True,
                ),
                mock.patch.object(
                    flex_attention_templates,
                    "gfx950_flex_attention_backward_template",
                    self._Template(),
                ),
        ):
            flex_attention_templates.append_tlx_flex(
                choices,
                [self._stock_backward_config()],
                self._backward_inputs(),
                [object(), object(), object(), []],
                object(),
                {"bwd_BLOCK_M1": 16, "bwd_BLOCK_N2": 16},
                128,
                128,
                mutated_inputs=[],
            )

        self.assertEqual(["stock"], choices)

    def test_flex_backward_empty_configs_never_fall_through_to_forward(self):
        """A malformed backward payload must retain the stock fallback."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        choices = ["stock"]
        with (
                config.patch({"triton.tlx_mode": "force"}),
                mock.patch.object(
                    flex_attention_templates,
                    "_use_amd_flex_template",
                    return_value=True,
                ),
        ):
            flex_attention_templates.append_tlx_flex(
                choices,
                [],
                self._backward_inputs(),
                [object(), object(), object(), []],
                object(),
                {},
                128,
                128,
                mutated_inputs=[],
            )

        self.assertEqual(["stock"], choices)

    def test_flex_backward_force_replaces_stock_only_after_choice_appends(self):
        """Rejected candidates and unsafe effective overrides keep stock."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        for template, kernel_options in (
            (self._RejectingTemplate(), {}),
            (self._Template(), {"bwd_BLOCK_M1": 16}),
            (self._Template(), {"bwd_num_warps": 8}),
            (self._Template(), {"SPARSE_Q_BLOCK_SIZE": 96}),
            (self._Template(), {"SPARSE_KV_BLOCK_SIZE": 0}),
        ):
            with self.subTest(template=type(template).__name__, kernel_options=kernel_options):
                choices = ["stock"]
                with (
                        config.patch({"triton.tlx_mode": "force"}),
                        mock.patch.object(
                            flex_attention_templates,
                            "_use_amd_flex_template",
                            return_value=True,
                        ),
                        mock.patch.object(
                            flex_attention_templates,
                            "gfx950_flex_attention_backward_template",
                            template,
                        ),
                ):
                    flex_attention_templates.append_tlx_flex(
                        choices,
                        [self._backward_config()],
                        self._backward_inputs(),
                        [object(), object(), object(), []],
                        object(),
                        kernel_options,
                        128,
                        128,
                        mutated_inputs=[],
                    )

                self.assertEqual(["stock"], choices)

    def test_flex_backward_registration_only_offers_native_k32_tile(self):
        """Scheduled MFMA candidates must use the reviewed native fragments."""
        from triton.language.extra.tlx.inductor import flex_attention_templates

        choices = ["stock"]
        with (
                config.patch({"triton.tlx_mode": "force"}),
                mock.patch.object(
                    flex_attention_templates,
                    "_use_amd_flex_template",
                    return_value=True,
                ),
                mock.patch.object(
                    flex_attention_templates,
                    "gfx950_flex_attention_backward_template",
                    self._Template(),
                ),
        ):
            flex_attention_templates.append_tlx_flex(
                choices,
                [self._stock_backward_config()],
                self._backward_inputs(),
                [object(), object(), object(), []],
                object(),
                {"bwd_BLOCK_N2": 64},
                128,
                128,
                mutated_inputs=[],
            )

        self.assertEqual(["stock"], choices)

    def test_flex_backward_template_exposes_stock_abi_grid(self):
        """A forward grid on the backward template silently misowns dK/dV."""
        from torch._inductor.select_algorithm import TritonTemplate
        from triton.language.extra.tlx.inductor import flex_attention_templates

        template = (flex_attention_templates.gfx950_flex_attention_backward_template)
        self.assertIsInstance(template, TritonTemplate)
        self.assertEqual("tlx_gfx950_flex_attention_bwd", template.name)
        self.assertEqual(
            (36, 2, 1),
            template.grid(
                2,
                8,
                512,
                128,
                1,
                512,
                {"BLOCK_M2": 128, "BLOCK_N1": 128},
            ),
        )


@instantiate_parametrized_tests
class TestFlexAttention(TestCase):
    """AMD (gfx950/MI350) FlexAttention Inductor template.

    Exercises torch.compile(flex_attention) under tlx_mode and asserts the
    tlx_gfx950_flex_attention template is selected and numerically correct across
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

    def _run_backward(self, fn, q, k, v):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        with config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": False,
        }):
            out, code = run_fw_bw_and_get_code(lambda: torch.compile(fn)(q, k, v))
        return out, (q.grad, k.grad, v.grad), "\n".join(code)

    def _assert_tlx_backward_generated(self, code):
        self.assertIn("# TLX_TEMPLATE: gfx950_flex_attention_bwd", code)
        self.assertIn("tlx.amd_scheduled_mfma", code)
        self.assertIn("tlx.amd_mfma_commit", code)

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
        self.assertIn("tlx_gfx950_flex_attention", code)

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
        self.assertIn("tlx_gfx950_flex_attention", code)

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
        self.assertIn("tlx_gfx950_flex_attention", code)

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
        self.assertIn("tlx_gfx950_flex_attention", "\n".join(code))

    @unittest.skipIf(not is_gfx950(), "Need AMD MI350X (gfx950)")
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not flex_backward_choices_hook_available(),
        "torch lacks the FlexAttention backward choices-hook contract",
    )
    @parametrize("head_dim", (64, 128))
    def test_flex_backward_dense(self, head_dim):
        from torch.nn.attention.flex_attention import flex_attention

        B, H, N, D = 1, 2, (257 if head_dim == 64 else 256), head_dim
        sm = 1.0 / (D**0.5)
        q, k, v = self._qkv(B, H, N, D, torch.bfloat16)
        fn = lambda q, k, v: flex_attention(q, k, v, scale=sm)  # noqa: E731
        out, grads, code = self._run_backward(fn, q, k, v)

        ref_q = q.detach().requires_grad_(True)
        ref_k = k.detach().requires_grad_(True)
        ref_v = v.detach().requires_grad_(True)
        ref = fn(ref_q, ref_k, ref_v)
        ref.sum().backward()

        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
        for actual, expected in zip(grads, (ref_q.grad, ref_k.grad, ref_v.grad)):
            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
        self._assert_tlx_backward_generated(code)

    @unittest.skipIf(not is_gfx950(), "Need AMD MI350X (gfx950)")
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not flex_backward_choices_hook_available(),
        "torch lacks the FlexAttention backward choices-hook contract",
    )
    @parametrize(
        "case",
        ((128, "causal"), (128, "sliding"), (128, "striped"), (64, "striped")),
        name_fn=lambda case: f"d{case[0]}_{case[1]}",
    )
    def test_flex_backward_partial_block_mask_gqa(self, case):
        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        D, mask_kind = case
        B, Hq, Hkv, N = 1, 4, 1, 256
        sm = 1.0 / (D**0.5)
        torch.manual_seed(0)
        q = torch.randn(B, Hq, N, D, device=GPU_TYPE, dtype=torch.bfloat16)
        k = torch.randn(B, Hkv, N, D, device=GPU_TYPE, dtype=torch.bfloat16)
        v = torch.randn(B, Hkv, N, D, device=GPU_TYPE, dtype=torch.bfloat16)

        def mask_mod(b, h, m, n):
            if mask_kind == "causal":
                return m >= n
            if mask_kind == "sliding":
                return (m >= n) & (m - n <= 160)
            return (m == n) | ((m >= n) & ((n // 64) % 3 != 1))

        block_mask = create_block_mask(
            mask_mod,
            B,
            Hq,
            N,
            N,
            device=GPU_TYPE,
        )
        score_mod = lambda score, b, h, m, n: score * 0.7  # noqa: E731
        fn = lambda q, k, v: flex_attention(  # noqa: E731
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=sm,
            enable_gqa=True,
        )
        out, grads, code = self._run_backward(fn, q, k, v)

        ref_q = q.detach().requires_grad_(True)
        ref_k = k.detach().requires_grad_(True)
        ref_v = v.detach().requires_grad_(True)
        ref = fn(ref_q, ref_k, ref_v)
        ref.sum().backward()

        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
        for actual, expected in zip(grads, (ref_q.grad, ref_k.grad, ref_v.grad)):
            torch.testing.assert_close(actual, expected, atol=6e-2, rtol=6e-2)
        self._assert_tlx_backward_generated(code)

    @unittest.skipIf(not is_gfx950(), "Need AMD MI350X (gfx950)")
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not flex_backward_choices_hook_available(),
        "torch lacks the FlexAttention backward choices-hook contract",
    )
    def test_flex_backward_captured_score_mod_gradient(self):
        from torch.nn.attention.flex_attention import flex_attention

        B, H, N, D = 1, 2, 128, 128
        sm = 1.0 / (D**0.5)
        q, k, v = self._qkv(B, H, N, D, torch.bfloat16)
        bias = torch.randn(H, N, N, device=GPU_TYPE, dtype=torch.bfloat16)

        def fn(q, k, v, bias):

            def score_mod(score, b, h, m, n):
                return score + bias[h, m, n]

            return flex_attention(q, k, v, score_mod=score_mod, scale=sm)

        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        bias = bias.detach().requires_grad_(True)
        with config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "max_autotune": True,
        }):
            out, code = run_fw_bw_and_get_code(lambda: torch.compile(fn)(q, k, v, bias))

        ref_q = q.detach().requires_grad_(True)
        ref_k = k.detach().requires_grad_(True)
        ref_v = v.detach().requires_grad_(True)
        ref_bias = bias.detach().requires_grad_(True)
        ref = fn(ref_q, ref_k, ref_v, ref_bias)
        ref.sum().backward()

        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
        for actual, expected in zip(
            (q.grad, k.grad, v.grad, bias.grad),
            (ref_q.grad, ref_k.grad, ref_v.grad, ref_bias.grad),
        ):
            torch.testing.assert_close(actual, expected, atol=7e-2, rtol=7e-2)
        code = "\n".join(code)
        self._assert_tlx_backward_generated(code)


class TestResourceModel(TestCase):
    """Pure arithmetic against GPU arch to calc HW limiter
    """

    @staticmethod
    def _tile(**overrides):
        from triton.language.extra.tlx.hw.resources import BlackwellWSGemmConfig

        kwargs = dict(
            block_m=128,
            block_n=128,
            block_k=64,
            num_smem_buffers=4,
            num_tmem_buffers=2,
            num_mma_groups=1,
            num_ctas=1,
            epilogue_subtile=2,
        )
        kwargs.update(overrides)
        return BlackwellWSGemmConfig(**kwargs)

    def test_hw_package_does_not_pull_in_inductor(self):
        """tlx.hw must not import torch._inductor.

        Primarily this prevents an import cycle. torch's shim
        (torch/_inductor/template_heuristics/tlx.py) does::

            import triton.language.extra.tlx.inductor.registry

        so the chain is torch._inductor -> tlx.inductor.registry -> tlx.hw. An
        import of torch._inductor from tlx.hw would re-enter a
        partially-initialized torch._inductor.

        Secondarily it keeps tlx.hw shareable with the standalone tutorial
        kernels, which is why the hardware model was split out of tlx/inductor/
        in the first place. Note this is not a torch-free guarantee -- target.py
        imports torch -- and tlx/__init__.py imports neither subpackage, so a
        triton-only consumer is unaffected either way.

        Checked in a subprocess because this module has already imported
        torch._inductor itself.
        """
        import subprocess
        import sys

        script = ("import sys; "
                  "import triton.language.extra.tlx.hw.resources; "
                  "import triton.language.extra.tlx.hw.target; "
                  "leaked = [m for m in sys.modules if m.startswith('torch._inductor')]; "
                  "print(len(leaked))")
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout.strip(),
            "0",
            "tlx.hw imported torch._inductor; keep the hardware model free of "
            "Inductor so the tutorials can share it",
        )

    # --- Blackwell WS GEMM model ----------------------------------------

    def test_blackwell_smem_matches_hand_computed(self):
        from triton.language.extra.tlx.hw.resources import BLACKWELL_WS_GEMM

        tile = self._tile()
        # A: 128*64*2*4, B: 64*128*2*4, epilog: 128*(128/2)*2, barriers: 4*1*8
        expected = 65536 + 65536 + 16384 + 32
        self.assertEqual(BLACKWELL_WS_GEMM.estimate_smem(tile, charge_epilogue=True), expected)
        self.assertEqual(
            BLACKWELL_WS_GEMM.estimate_smem(tile, charge_epilogue=False),
            expected - 16384,
        )

    def test_blackwell_smem_two_cta_halves_b_and_doubles_barriers(self):
        from triton.language.extra.tlx.hw.resources import BLACKWELL_WS_GEMM

        one = self._tile(block_m=256, num_mma_groups=2, num_ctas=1)
        two = self._tile(block_m=256, num_mma_groups=2, num_ctas=2)
        # B is split across the CTA pair; each CTA keeps its own barriers.
        self.assertEqual(
            BLACKWELL_WS_GEMM.estimate_smem(two, charge_epilogue=True),
            BLACKWELL_WS_GEMM.estimate_smem(one, charge_epilogue=True) - 32768 + 64,
        )

    def test_blackwell_smem_split_k_adds_fp32_workspace(self):
        from triton.language.extra.tlx.hw.resources import BLACKWELL_WS_GEMM

        tile = self._tile()
        base = BLACKWELL_WS_GEMM.estimate_smem(tile, charge_epilogue=True, split_k=1)
        # (128/1) * (128/2) * 4 bytes * max(num_mma_groups, 2)
        self.assertEqual(
            BLACKWELL_WS_GEMM.estimate_smem(tile, charge_epilogue=True, split_k=4),
            base + 128 * 64 * 4 * 2,
        )

    def test_blackwell_tmem_is_columns_not_bytes(self):
        """TMEM is 128-lane granular, so BLOCK_M must not appear in the model."""
        from triton.language.extra.tlx.hw.resources import BLACKWELL_WS_GEMM

        cols = BLACKWELL_WS_GEMM.estimate_tmem_columns
        self.assertEqual(
            cols(self._tile(block_m=256, num_mma_groups=2)),
            cols(self._tile(block_m=64, num_mma_groups=2)),
        )
        self.assertEqual(cols(self._tile()), 128 * 2 * 1)
        # num_mma_groups multiplies the column count.
        self.assertEqual(cols(self._tile(num_mma_groups=2)), 128 * 2 * 2)

    def test_blackwell_tmem_limit_rejects_oversized_accumulator(self):
        from triton.language.extra.tlx.hw.resources import (
            BLACKWELL_WS_GEMM,
            DeviceLimits,
        )

        limits = DeviceLimits.for_arch("sm100")
        # 256 columns * 2 buffers * 2 groups = 1024 > 512, but SMEM still fits,
        # so this isolates the TMEM check.
        tile = self._tile(block_n=256, num_smem_buffers=2, num_mma_groups=2)
        self.assertLess(
            BLACKWELL_WS_GEMM.estimate_smem(tile, charge_epilogue=True),
            limits.on_chip_bytes,
        )
        self.assertFalse(BLACKWELL_WS_GEMM.validate(tile, charge_epilogue=True, limits=limits))
        self.assertTrue(BLACKWELL_WS_GEMM.validate(tile, charge_epilogue=True, check_tmem=False, limits=limits))

    def test_blackwell_tile_rules(self):
        from triton.language.extra.tlx.hw.resources import BLACKWELL_WS_GEMM

        rules = BLACKWELL_WS_GEMM.check_tile_rules
        self.assertTrue(rules(self._tile()))
        # Rule 1: more than 128 rows per MMA group.
        self.assertFalse(rules(self._tile(block_m=256)))
        # Rule 1b: pair-CTA MMA requires exactly 128 rows per group.
        self.assertFalse(rules(self._tile(block_m=128, num_mma_groups=2, num_ctas=2)))
        self.assertTrue(rules(self._tile(block_m=256, num_mma_groups=2, num_ctas=2)))
        # Rule 2: EPILOGUE_SUBTILE must divide BLOCK_N.
        self.assertFalse(rules(self._tile(block_n=192, epilogue_subtile=128)))

    def test_blackwell_smem_margin_is_applied(self):
        from triton.language.extra.tlx.hw.resources import (
            BLACKWELL_WS_GEMM,
            DeviceLimits,
        )

        limits = DeviceLimits.for_arch("sm100")
        tile = self._tile()
        smem = BLACKWELL_WS_GEMM.estimate_smem(tile, charge_epilogue=True)
        slack = limits.on_chip_bytes - smem
        self.assertGreater(slack, 0, "pick a tile that still fits without margin")
        self.assertTrue(BLACKWELL_WS_GEMM.validate(tile, charge_epilogue=True, smem_margin=slack, limits=limits))
        self.assertFalse(BLACKWELL_WS_GEMM.validate(tile, charge_epilogue=True, smem_margin=slack + 1, limits=limits))

    # --- AMD warp-pipe model (budgets pinned at gfx950) -------------------

    def test_gfx950_warppipe_lds_matches_documented_budget(self):
        """Cross-check against the hand-verified figures on WARPPIPE_CONFIGS."""
        from triton.language.extra.tlx.hw.resources import (
            AMD_WARP_PIPE,
            AmdWarpPipeConfig,
        )

        for block_m, block_n, block_k, num_buffers, want_kb in (
            (128, 256, 64, 2, 96),
            (128, 256, 32, 3, 72),
            (128, 256, 64, 3, 144),
        ):
            cfg = AmdWarpPipeConfig(block_m, block_n, block_k, num_buffers)
            self.assertEqual(
                AMD_WARP_PIPE.estimate_smem(cfg),
                want_kb * 1024,
                f"({block_m}x{block_n}x{block_k}, NB{num_buffers})",
            )

    def test_gfx950_warppipe_register_path_allocates_no_lds(self):
        from triton.language.extra.tlx.hw.resources import (
            AMD_WARP_PIPE,
            AmdWarpPipeConfig,
        )

        cfg = AmdWarpPipeConfig(128, 256, 64, 3, use_async=False)
        self.assertEqual(AMD_WARP_PIPE.estimate_smem(cfg), 0)

    def test_gfx950_warppipe_has_no_tensor_memory(self):
        from triton.language.extra.tlx.hw.resources import (
            AMD_WARP_PIPE,
            AmdWarpPipeConfig,
        )

        self.assertEqual(AMD_WARP_PIPE.estimate_tmem_columns(AmdWarpPipeConfig(128, 256, 64, 3)), 0)

    def test_every_shipped_warppipe_config_fits_gfx950(self):
        """The tuned AMD pool must not contain a config the model rejects."""
        from triton.language.extra.tlx.inductor.registry import (
            Gfx950AddMMWarpPipeConfigHeuristic as H, )
        from triton.language.extra.tlx.hw.resources import (
            AMD_WARP_PIPE,
            DeviceLimits,
            AmdWarpPipeConfig,
        )

        limits = DeviceLimits.for_arch("gfx950")
        for block_m, block_n, block_k, _gm, _nw, num_buffers in H.WARPPIPE_CONFIGS:
            cfg = AmdWarpPipeConfig(block_m, block_n, block_k, num_buffers)
            self.assertTrue(
                AMD_WARP_PIPE.validate(cfg, limits=limits),
                f"shipped config ({block_m}x{block_n}x{block_k}, "
                f"NB{num_buffers}) = {AMD_WARP_PIPE.estimate_smem(cfg)} B "
                f"exceeds the gfx950 LDS budget of {limits.on_chip_bytes} B",
            )

    def test_models_are_registered_for_every_tlx_template(self):
        from triton.language.extra.tlx.inductor import mm_templates
        from triton.language.extra.tlx.hw.resources import MODELS

        for name in (
                "blackwell_gemm_ws_template",
                "gfx950_addmm_warppipe_template",
                "gfx950_addmm_persistent_warppipe_template",
                "gfx950_bmm_warppipe_template",
        ):
            uid = getattr(mm_templates, name).uid
            self.assertIn(uid, MODELS, f"{name} ({uid}) has no resource model")


class TestCandidateScorer(TestCase):
    """The fallback candidate scorer.

    The scorer and its ``_CANDIDATES`` table are specific to the Blackwell
    warp-specialized GEMM template -- the configs carry NUM_TMEM_BUFFERS and
    pair-CTA counts, neither of which exists on the AMD warp-pipe. The pure
    functions are exercised everywhere by pinning Blackwell limits explicitly;
    only the end-to-end lookup, which reads the live target, is device-gated.
    """

    #: Shapes covering both saturated and undersaturated regimes.
    _SHAPES = [(m, n, k) for m in (128, 1024, 4096, 16384) for n in (64, 256, 4096) for k in (512, 4096, 8192)]

    def test_candidates_are_blackwell_shaped(self):
        """Guard the assumption the rest of this class rests on."""
        from triton.language.extra.tlx.inductor.registry import _CANDIDATES
        from triton.language.extra.tlx.hw.resources import (
            BLACKWELL_WS_GEMM,
            DeviceLimits,
            BlackwellWSGemmConfig,
        )

        limits = DeviceLimits.for_arch("sm100")
        self.assertTrue(_CANDIDATES)
        for cfg in _CANDIDATES:
            tile = BlackwellWSGemmConfig.from_dict(cfg)
            # Every candidate needs tensor memory, so this table cannot be
            # scored against a non-Blackwell target.
            self.assertGreater(BLACKWELL_WS_GEMM.estimate_tmem_columns(tile), 0)
            self.assertLessEqual(BLACKWELL_WS_GEMM.estimate_tmem_columns(tile), limits.tmem_columns)

    def test_scorer_never_returns_a_structurally_invalid_config(self):
        """The scorer must not pick a config the tile rules reject.

        It used to skip the pair-CTA rule that ``_is_config_valid`` enforces,
        so a violating candidate could win here and be rejected downstream.
        Because both of ``get_heuristic_config``'s retry paths re-run this same
        deterministic scorer, the retries returned the identical config and the
        whole lookup fell through to ``None``.
        """
        from triton.language.extra.tlx.inductor.registry import (
            _candidate_scorer_evaluate, )
        from triton.language.extra.tlx.hw.resources import (
            BLACKWELL_WS_GEMM,
            BlackwellWSGemmConfig,
        )

        selected = set()
        for m, n, k in self._SHAPES:
            cfg = _candidate_scorer_evaluate(m, n, k, 148)
            if cfg is None:
                continue
            selected.add((cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["NUM_CTAS"]))
            self.assertTrue(
                BLACKWELL_WS_GEMM.check_tile_rules(BlackwellWSGemmConfig.from_dict(cfg)),
                f"scorer returned a structurally invalid config for "
                f"({m},{n},{k}): {cfg}",
            )
        self.assertTrue(selected, "scorer never selected anything")
        # The pair-CTA violator is still in the candidate table; it is filtered
        # by the scorer rather than deleted, so pin that it never wins.
        self.assertNotIn((128, 64, 2), selected)

    @unittest.skipUnless(is_blackwell(), "the scorer's candidate table is Blackwell-specific")
    def test_pair_cta_candidate_no_longer_strands_the_lookup(self):
        """Shapes that used to fall through to None now get a config.

        Device-gated: get_heuristic_config resolves limits from the live
        target, so on a non-Blackwell part these shapes legitimately produce
        no config.
        """
        from triton.language.extra.tlx.inductor.registry import get_heuristic_config

        # These scored best on the (128,64,128) 2-CTA candidate before the fix,
        # which _is_config_valid then rejected, stranding the whole lookup.
        for m, n, k in ((128, 128, 4096), (128, 1024, 2049), (128, 128, 8192)):
            cfg = get_heuristic_config(m, n, k, num_sms=148, tma_epilogue_store=True)
            self.assertIsNotNone(cfg, f"({m},{n},{k}) still returns None")

    def test_split_k_workspace_gap_is_still_open(self):
        """Documents the remaining scorer/validator divergence.

        The scorer picks ``SPLIT_K`` without charging the fp32 partials
        workspace that ``_is_config_valid`` does charge, so it can still return
        a config that is rejected downstream -- the same failure shape as the
        pair-CTA bug, stranding ~84 of the sweep's shapes at ``None``. Closing
        it changes which split factor real shapes get, so it is deliberately
        left for its own diff. Flip this test when that lands.
        """
        from triton.language.extra.tlx.inductor.registry import (
            _candidate_scorer_evaluate,
            _is_config_valid,
        )

        stranded = [(m, n, k)
                    for m, n, k in self._SHAPES
                    if (cfg := _candidate_scorer_evaluate(m, n, k, 148)) is not None
                    and not _is_config_valid(cfg, tma_epilogue_store=False)]
        self.assertTrue(
            all(_candidate_scorer_evaluate(m, n, k, 148).get("SPLIT_K", 1) > 1 for m, n, k in stranded),
            "a non-split-K config is now stranded; the scorer and "
            "_is_config_valid have diverged in a new way",
        )


class TestTargetDetection(TestCase):
    """Arch detection. No GPU needed."""

    def test_is_rocm_does_not_touch_the_device(self):
        from triton.language.extra.tlx.hw.target import is_rocm

        with mock.patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("queried")):
            self.assertEqual(is_rocm(), torch.version.hip is not None)

    @staticmethod
    def _target(arch_key, *, arch="", capability=None, num_sms=1):
        """Build a Target for a named arch without touching a device."""
        from triton.language.extra.tlx.hw.resources import ARCH_SPECS, has_tmem
        from triton.language.extra.tlx.hw.target import Target

        spec = ARCH_SPECS[arch_key]
        return Target(
            spec=spec,
            is_rocm=spec.vendor == "amd",
            arch=arch,
            capability=capability,
            num_sms=num_sms,
            smem_bytes=getattr(spec, "lds_bytes", None) or getattr(spec, "smem_bytes", 0) or 0,
            tmem_columns=spec.tmem_columns() if has_tmem(spec) else None,
        )

    def test_no_device_resolves_to_no_arch(self):
        """A build host must not look like a B200.

        current_target() still returns a Target so the pure-Python heuristics
        stay callable, but with spec=None: every arch predicate is False and
        has_device says why.
        """
        from triton.language.extra.tlx.hw import target

        target.current_target.cache_clear()
        try:
            with mock.patch.object(torch.cuda, "get_device_properties", side_effect=RuntimeError("no gpu")):
                t = target.current_target()
            self.assertIsNone(t.spec)
            self.assertFalse(t.has_device)
            self.assertEqual(t.key, "")
            self.assertFalse(t.is_blackwell)
            self.assertFalse(t.is_hopper)
            self.assertFalse(t.is_gfx950)
            self.assertFalse(t.has_tmem)
            self.assertIsNone(t.tmem_columns)
            with self.assertRaises(AttributeError):
                t.num_xcds
            # Numbers still usable, from the documented reference arch.
            self.assertEqual(t.num_sms, target.REFERENCE_ARCH.processor_count())
            self.assertEqual(t.smem_bytes, target.REFERENCE_ARCH.on_chip_bytes())
        finally:
            target.current_target.cache_clear()

    # --- arch resolution ---------------------------------------------------

    def test_each_hierarchy_answers_the_vendor_agnostic_accessors(self):
        """Both hierarchies implement on_chip_bytes/processor_count.

        They have no common base, so this duck-typed pair is the only contract
        a vendor-agnostic caller can rely on.
        """
        from triton.language.extra.tlx.hw.resources import ARCH_SPECS

        for key, spec in ARCH_SPECS.items():
            self.assertTrue(callable(spec.on_chip_bytes), key)
            self.assertTrue(callable(spec.processor_count), key)
            if spec.vendor == "nvidia":
                self.assertEqual(spec.on_chip_bytes(), spec.smem_bytes, key)
                self.assertEqual(spec.processor_count(), spec.num_sms, key)
                self.assertFalse(hasattr(spec, "lds_bytes"), key)
                self.assertFalse(hasattr(spec, "num_xcds"), key)
            else:
                self.assertFalse(hasattr(spec, "smem_bytes"), key)
                if hasattr(spec, "lds_bytes"):
                    self.assertEqual(spec.on_chip_bytes(), spec.lds_bytes, key)
                    self.assertEqual(spec.processor_count(), spec.num_cus, key)

    def test_newer_arch_inherits_from_its_predecessor(self):
        """Blackwell is Hopper plus TMEM; CDNA4 is CDNA3 with a bigger LDS."""
        from triton.language.extra.tlx.hw.resources import (
            Gfx942,
            Gfx950,
            Gfx1250,
            Sm90,
            Sm100,
        )

        self.assertTrue(issubclass(Sm100, Sm90))
        self.assertTrue(issubclass(Gfx950, Gfx942))
        # gfx1250 is a separate lineage, so it cannot inherit CDNA4's facts.
        self.assertFalse(issubclass(Gfx1250, Gfx950))
        # The vendors stay disjoint: no shared parent across them.
        self.assertFalse(issubclass(Sm100, Gfx942))
        self.assertFalse(issubclass(Gfx942, Sm90))
        # TMEM is introduced at Blackwell, so it is absent everywhere else --
        # reaching for it is an AttributeError, not a plausible 0.
        from triton.language.extra.tlx.hw.resources import has_tmem

        self.assertTrue(has_tmem(Sm100))
        for spec in (Sm90, Gfx942, Gfx950, Gfx1250):  # Rubin inherits Sm100's
            self.assertFalse(has_tmem(spec), spec.__name__)
            with self.assertRaises(AttributeError):
                spec.tmem_columns()

        # An uncharacterized fact is absent, not None and not inherited.
        for missing in ("lds_bytes", "num_cus"):
            self.assertFalse(hasattr(Gfx1250, missing), missing)
        with self.assertRaises(AttributeError):
            Gfx1250.on_chip_bytes()
        with self.assertRaises(AttributeError):
            Gfx1250.processor_count()

    def test_unrecognized_amd_arch_raises_rather_than_inheriting(self):
        """An unknown gfx must not silently pick up gfx950's LDS and XCDs."""
        from triton.language.extra.tlx.hw import target

        props = mock.Mock(
            gcnArchName="gfx90a:sramecc+",
            multi_processor_count=104,
            shared_memory_per_block_optin=65536,
            major=0,
            minor=0,
        )
        target.current_target.cache_clear()
        try:
            with mock.patch.object(target, "is_rocm", return_value=True), \
                 mock.patch.object(
                     torch.cuda, "get_device_properties", return_value=props
                 ):
                with self.assertRaises(ValueError) as cm:
                    target.current_target()
            self.assertIn("gfx90a", str(cm.exception))
        finally:
            target.current_target.cache_clear()

    def test_arch_gates_are_false_without_a_device(self):
        """The module-level gates must not fire off the CPU-only fallback."""
        from triton.language.extra.tlx.hw import target

        target.current_target.cache_clear()
        try:
            with mock.patch.object(torch.cuda, "get_device_properties", side_effect=RuntimeError("no gpu")):
                self.assertFalse(is_blackwell())
                self.assertFalse(is_gfx950())
        finally:
            target.current_target.cache_clear()

    def test_arch_gates_agree_with_the_live_target(self):
        """One model answers both gates; they cannot both be true."""
        target = current_target()
        self.assertEqual(is_blackwell(), target.has_device and target.is_blackwell)
        self.assertEqual(is_gfx950(), target.has_device and target.is_gfx950)
        self.assertFalse(is_blackwell() and is_gfx950())

    def test_gfx1250_is_not_shadowed_by_a_gfx12_prefix(self):
        """triton._internal_testing documents this exact footgun."""
        from triton.language.extra.tlx.hw.target import _spec_for_rocm

        self.assertEqual(_spec_for_rocm("gfx1250").key, "gfx1250")
        self.assertEqual(_spec_for_rocm("gfx950:sramecc+:xnack-").key, "gfx950")
        self.assertEqual(_spec_for_rocm("gfx942:sramecc+").key, "gfx942")
        self.assertIsNone(_spec_for_rocm("gfx90a"))

    # --- resolved target -------------------------------------------------

    def test_num_xcds_absent_on_nvidia(self):
        """Chiplets are an AMD concept; asking on NVIDIA is a caller bug."""
        for key in ("sm90", "sm100", "sm107"):
            with self.assertRaises(AttributeError):
                self._target(key).num_xcds

    def test_num_xcds_on_amd(self):
        for key in ("gfx942", "gfx950", "gfx1250"):
            self.assertEqual(self._target(key).num_xcds, 8, key)

    def test_arch_predicates_are_mutually_exclusive(self):
        for key in ("sm90", "sm100", "sm107", "gfx942", "gfx950", "gfx1250"):
            t = self._target(key)
            flags = [
                t.is_hopper,
                t.is_blackwell,
                t.is_rubin,
                t.is_gfx942,
                t.is_gfx950,
                t.is_gfx1250,
            ]
            self.assertEqual(sum(flags), 1, f"{key} matched {sum(flags)} predicates")
            self.assertEqual(t.is_rocm, key.startswith("gfx"))
            # Blackwell and everything descended from it.
            self.assertEqual(t.has_tmem, key in ("sm100", "sm107"))

    def test_default_kpack_matches_torch(self):
        self.assertEqual(self._target("sm100").default_kpack(16), 0)
        self.assertEqual(self._target("gfx942").default_kpack(16), 1)
        self.assertEqual(self._target("gfx942").default_kpack(32), 2)
        self.assertEqual(self._target("gfx950").default_kpack(16), 2)

    def test_capability_maps_to_arch(self):
        from triton.language.extra.tlx.hw.target import _spec_for_cuda

        self.assertEqual(_spec_for_cuda((9, 0)).key, "sm90")
        self.assertEqual(_spec_for_cuda((10, 0)).key, "sm100")
        # Rubin has its own placeholder class.
        self.assertEqual(_spec_for_cuda((10, 7)).key, "sm107")
        # Blackwell Ultra / newer resolve forward, not back to Hopper.
        self.assertEqual(_spec_for_cuda((10, 3)).key, "sm100")
        self.assertEqual(_spec_for_cuda((11, 0)).key, "sm100")

    def test_get_heuristic_config_defaults_num_sms_from_target(self):
        from triton.language.extra.tlx.hw import target
        from triton.language.extra.tlx.inductor import registry
        from triton.language.extra.tlx.inductor.registry import get_heuristic_config

        stub = self._target("sm100", capability=(10, 0), num_sms=148)
        target.current_target.cache_clear()
        try:
            with mock.patch.object(registry, "current_target", return_value=stub):
                self.assertEqual(
                    get_heuristic_config(4096, 4096, 4096),
                    get_heuristic_config(4096, 4096, 4096, num_sms=148),
                )
        finally:
            target.current_target.cache_clear()


if __name__ == "__main__":
    run_tests()
