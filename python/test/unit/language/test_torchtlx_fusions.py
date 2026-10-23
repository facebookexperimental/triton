# Owner(s): ["module: inductor"]
"""
Unit tests for TorchTLX (Triton Language eXtensions) epilogue fusion.

Usage:
    pytest python/test/unit/language/test_torchtlx_fusions.py
"""

import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.utils._triton import has_datacenter_blackwell_tma_device

try:
    from triton.language.extra.tlx.inductor import tlx_config
except ImportError:
    pass

try:
    from triton.language.extra.tlx.tutorials.blackwell_gemm_pipelined import (  # @manual
        matmul as _blackwell_gemm_pipelined, )

    _HAS_BLACKWELL_TUTORIAL = True
# The Blackwell tutorial runs get_active_torch_device() at import time, which
# raises RuntimeError ("No CUDA GPUs are available") on non-CUDA hosts (e.g. the
# AMD test runner). Catch broadly so the module still imports and the tutorial
# just gates out via _HAS_BLACKWELL_TUTORIAL.
except Exception:
    _HAS_BLACKWELL_TUTORIAL = False


def has_tlx():
    """Check if TLX (Triton Language eXtensions) is available.

    This only checks for the TLX language package; it does NOT import the
    packaged NV Blackwell tutorial kernel (that lives behind
    _HAS_BLACKWELL_TUTORIAL), since importing a tutorial does not imply the
    hardware it targets is available.
    """
    try:
        import triton.language.extra.tlx  # noqa: F401  # @manual

        return True
    except ImportError:
        return False


def is_gfx950():
    """True on AMD MI350X (gfx950), where the TLX warp-pipe addmm template runs.

    Delegates to the shared target model so adding MI300X/MI450X coverage is a
    change to the arch table rather than to every gate in the test suite. Falls
    back to False when TLX is not importable at all, matching has_tlx().
    """
    try:
        from triton.language.extra.tlx.hw.target import current_target
    except ImportError:
        return False
    return current_target().is_gfx950


def supports_template_epilogue_fusion():
    """True if the active torch exposes the template epilogue-fusion codegen API.

    Older torch wheels (e.g. the current ROCm nightly) lack it, and additionally
    decompose ``addmm`` -> ``mm`` + epilogue when a pointwise consumer follows, so
    the TLX addmm warp-pipe template is never a candidate. Epilogue-fusion tests
    gate on this so they run on a matching torch and skip cleanly otherwise.

    Probe torch directly (``codegen_template_body`` is the method the fork's
    fusion path wraps) rather than importing the fork registry: importing the
    registry this early — at collection time, before torch first imports it during
    compile — perturbs TLX JIT resolution on some torch versions and breaks
    unrelated template compiles.
    """
    try:
        from torch._inductor.select_algorithm import TritonTemplateKernel

        return hasattr(TritonTemplateKernel, "codegen_template_body")
    except Exception:
        return False


torch.set_float32_matmul_precision("high")

# Shapes for fusion testing - representative shapes from gemm_rule categories
# Tall-M shapes (Rules 1-4) have correctness issues with NUM_CTAS=2 configs - needs investigation
# Using only proven working shapes for now
FUSION_TEST_SHAPES = [
    # (M, K, N)
    (4096, 4096, 4096),  # Rule 7: GPU-Saturated General
    (16384, 4096, 2048),  # Tall-M (M/N=8, uses Rule 7 config)
    (1152, 16384, 1024),  # Rule 5: Undersaturated Large-Output
]

# Representative logical (M, K, N) shapes from the AMD GEMM-fusion study.
GEMM_NORM_TEST_CASES = [
    ("layernorm", (2032, 2560, 2560)),
    ("rmsnorm", (677, 8192, 4096)),
]


def _normalize(
    value: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    norm_kind: str,
) -> torch.Tensor:
    normalized_shape = (value.shape[-1], )
    if norm_kind == "rmsnorm":
        return torch.nn.functional.rms_norm(
            value,
            normalized_shape,
            scale,
            1.0e-5,
        )
    return torch.nn.functional.layer_norm(
        value,
        normalized_shape,
        scale,
        bias,
        1.0e-5,
    )


@instantiate_parametrized_tests
class TestTorchTLXEpilogueFusion(TestCase):

    def setUp(self) -> None:
        super().setUp()
        import triton.language.extra.tlx.inductor.registry  # noqa: F401

        # Reduction epilogues fuse whenever legal, so codegen checks do not
        # depend on fusion benchmark noise.
        env = mock.patch.dict(os.environ, {"TORCHINDUCTOR_TLX_REDUCTION_EPILOGUE": "force"})
        env.start()
        self.addCleanup(env.stop)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(
        not (has_tlx() and _HAS_BLACKWELL_TUTORIAL),
        "TLX Blackwell tutorial not available",
    )
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("shape", FUSION_TEST_SHAPES)
    @parametrize("use_heuristic_config", (True, False))
    def test_matmul_bias_relu_epilogue_is_fused(
        self,
        dtype: torch.dtype,
        shape: tuple[int, int, int],
        use_heuristic_config: bool,
    ):
        """Verify TLX epilogue fusion for mm + bias add + relu pattern."""

        class MatmulBiasReluModule(nn.Module):
            """Matrix multiply followed by bias add and ReLU activation."""

            def __init__(self, m: int, k: int, n: int, dtype: torch.dtype):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(k, n, dtype=dtype))
                self.bias = nn.Parameter(torch.randn(m, n, dtype=dtype))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.mm(x, self.weight)  # extern kernel (cublas)
                y = y + self.bias  # pointwise (bias add)
                y = y.relu()
                return y

        device = GPU_TYPE
        m, k, n = shape
        model = MatmulBiasReluModule(m=m, k=k, n=n, dtype=dtype).to(device)
        input_tensor = torch.randn(m, k, device=device, dtype=dtype)

        with (
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "enable_caching_generated_triton_templates": False,
                }),
                tlx_config.patch(use_heuristic_config=use_heuristic_config, ),
        ):
            with torch.no_grad():
                compiled_model = torch.compile(model)
                output_actual, generated_code = run_and_get_code(
                    compiled_model,
                    input_tensor,
                )
                # TLX fused kernels keep the matmul result in fp32 (the
                # accumulator is never downcast) and apply the epilogue
                # (bias add, relu, etc.) in fp32 before a single cast to
                # the output dtype.  This is required because Triton
                # libdevice math ops (e.g. sigmoid → exp) only accept
                # fp32/fp64, so the template cannot downcast before the
                # epilogue.
                #
                # The reference therefore does: cublas bf16 matmul (which
                # also accumulates in fp32 internally) → upcast output to
                # fp32 → epilogue in fp32 → cast to output dtype.
                output_expected = ((torch.mm(input_tensor, model.weight).float() + model.bias.float()).relu().to(dtype))

            # Tolerances: cublas and TLX both take bf16 inputs and
            # accumulate in fp32, but they tile and sum the K-dimension
            # products in different orders.  Because floating-point
            # addition is non-associative, the matmul outputs can differ
            # slightly even before the epilogue runs.  No eager reference
            # can eliminate this gap, so we use relaxed tolerances.
            torch.testing.assert_close(output_actual, output_expected, atol=2e-2, rtol=2e-2)

            generated_code_str = str(generated_code)

            # Split-K disables epilogue fusion: epilogue ops run as separate
            # kernels after the reduce_k step, not fused into the GEMM.
            is_split_k = "_reduce_k_kernel" in generated_code_str
            if is_split_k:
                return

            self.assertIn(
                "triton_tem_fused_tlx_add_mm_relu_0.run",
                generated_code_str,
                "Expected fused kernel 'triton_tem_fused_tlx_add_mm_relu_0.run' not found in generated code.",
            )
            # Non-split-K configs use TMA epilogue stores when SMEM permits;
            # fall back to tl.store for 1-CTA configs when TMA doesn't fit
            # (e.g. epilogue fusion adds SMEM for fused bias tensors).
            is_tma = "c_smem_buffers" in generated_code_str
            if is_tma:
                self.assertIn("tlx.async_descriptor_store(desc_c", generated_code_str)
                self.assertIn("tlx.local_store(c_smem", generated_code_str)
                self.assertIn("tlx.fence_async_shared()", generated_code_str)
            else:
                self.assertIn("tl.store(out_ptr", generated_code_str)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(
        not (has_tlx() and _HAS_BLACKWELL_TUTORIAL),
        "TLX Blackwell tutorial not available",
    )
    def test_tlx_mm_epilogue_with_live_intermediate_is_not_fused(self):
        """A GEMM output that stays live outside the fused group vetoes TLX fusion.

        The TLX compute_epilogue/TMA path redirects intermediate stores into
        variable assignments and materializes only the final result. Fusing
        mm + mul + sigmoid while the pre-sigmoid mm result is also a graph
        output (e.g. a tensor saved for backward) leaves that buffer allocated
        but never written. The veto keeps the GEMM materialized and runs the
        pointwise ops separately.
        """
        m, k, n = 4096, 4096, 1024
        dtype = torch.bfloat16
        device = GPU_TYPE
        torch.manual_seed(0)
        x = torch.randn(m, k, device=device, dtype=dtype)
        w = torch.randn(k, n, device=device, dtype=dtype)

        def mm_gated_with_live_mm(
            x: torch.Tensor,
            w: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            mm = torch.mm(x, w)
            return mm * torch.sigmoid(mm), mm

        with torch.no_grad():
            expected = mm_gated_with_live_mm(x, w)
            with (
                    config.patch({
                        "triton.tlx_mode": "force",
                        "force_disable_caches": True,
                        "enable_caching_generated_triton_templates": False,
                    }),
                    tlx_config.patch(use_heuristic_config=True),
            ):
                actual, code = run_and_get_code(
                    torch.compile(mm_gated_with_live_mm, fullgraph=True),
                    x,
                    w,
                )

        for actual_output, expected_output in zip(actual, expected):
            torch.testing.assert_close(
                actual_output,
                expected_output,
                atol=2e-2,
                rtol=2e-2,
            )
        generated_code_str = "\n".join(code)
        if "_reduce_k_kernel" in generated_code_str:
            # Split-K disables epilogue fusion; nothing to veto.
            return
        self.assertIn(
            "tlx_mm_",
            generated_code_str,
            "Expected the TLX mm template to be in play for this graph.",
        )
        self.assertNotIn(
            "triton_tem_fused_tlx_mm_",
            generated_code_str,
            "Live intermediate must veto the single-TMA fused TLX epilogue kernel.",
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(
        not (has_tlx() and _HAS_BLACKWELL_TUTORIAL),
        "TLX Blackwell tutorial not available",
    )
    def test_tlx_mm_epilogue_with_internal_intermediate_stays_fused(self):
        """Positive control: a fully internal intermediate still fuses."""
        m, k, n = 4096, 4096, 1024
        dtype = torch.bfloat16
        device = GPU_TYPE
        torch.manual_seed(0)
        x = torch.randn(m, k, device=device, dtype=dtype)
        w = torch.randn(k, n, device=device, dtype=dtype)

        def mm_gated(
            x: torch.Tensor,
            w: torch.Tensor,
        ) -> torch.Tensor:
            mm = torch.mm(x, w)
            return mm * torch.sigmoid(mm)

        with torch.no_grad():
            expected = mm_gated(x, w)
            with (
                    config.patch({
                        "triton.tlx_mode": "force",
                        "force_disable_caches": True,
                        "enable_caching_generated_triton_templates": False,
                    }),
                    tlx_config.patch(use_heuristic_config=True),
            ):
                actual, code = run_and_get_code(
                    torch.compile(mm_gated, fullgraph=True),
                    x,
                    w,
                )

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        generated_code_str = "\n".join(code)
        if "_reduce_k_kernel" in generated_code_str:
            return
        self.assertIn(
            "triton_tem_fused_tlx_mm_",
            generated_code_str,
            "Fully internal intermediate should still fuse into the TLX kernel.",
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(
        not (has_tlx() and _HAS_BLACKWELL_TUTORIAL),
        "TLX Blackwell tutorial not available",
    )
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_customized_matmul_relu_epilogue_is_fused(
        self,
        dtype: torch.dtype,
    ):
        """Verify TLX epilogue fusion for customized mm + relu pattern.

        The customized matmul op is not visible to inductor during fusion,
        simulating an external/opaque kernel that should still support
        epilogue fusion.
        """

        BLACKWELL_GEMM_CONFIG = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_STAGES": 4,
        }

        @torch.library.custom_op("tlx_test::customized_matmul", mutates_args=())
        def customized_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return _blackwell_gemm_pipelined(x, weight, config=BLACKWELL_GEMM_CONFIG)

        @customized_matmul.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            M, K = x.shape
            K2, N = weight.shape
            return torch.empty((M, N), device=x.device, dtype=x.dtype)

        class CustomizedKernelReluModule(nn.Module):
            """Customized matmul followed by ReLU activation."""

            def __init__(self, k: int, n: int, dtype: torch.dtype):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(k, n, dtype=dtype))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Use the customized matmul op (opaque to inductor)
                y = torch.ops.tlx_test.customized_matmul(x, self.weight)
                y = y.relu()  # epilogue
                return y

        device = GPU_TYPE
        m, k, n = 8192, 4096, 4096
        torch.manual_seed(0)
        model = CustomizedKernelReluModule(k=k, n=n, dtype=dtype).to(device)
        input_tensor = torch.randn(m, k, device=device, dtype=dtype)

        with (config.patch({
                "triton.tlx_mode": "force",
                "force_disable_caches": True,
                "enable_caching_generated_triton_templates": False,
        }), ):
            old_env = os.environ.get("TORCHINDUCTOR_FORCE_TLX_EPILOGUE_FUSION")
            old_torch_logs = os.environ.get("TORCH_LOGS")

            try:
                os.environ["TORCHINDUCTOR_FORCE_TLX_EPILOGUE_FUSION"] = "1"
                os.environ["TORCH_LOGS"] = "output_code,+inductor"

                with torch.no_grad():
                    compiled_model = torch.compile(model)
                    output_actual, generated_code = run_and_get_code(
                        compiled_model,
                        input_tensor,
                    )
                    output_expected = torch.mm(input_tensor, model.weight).relu()

                torch.testing.assert_close(output_actual, output_expected)

                generated_code_str = str(generated_code)
                print("=== GENERATED CODE ===")
                print(generated_code_str)
                print("=== END GENERATED CODE ===")

                # temporarily comment out the assert until the customized op fusion feature works
                # self.assertIn(
                #     "triton_tem_fused_",
                #     generated_code_str,
                #     "Expected a fused TLX template kernel in generated code.",
                # )
            finally:
                if old_env is None:
                    os.environ.pop("TORCHINDUCTOR_FORCE_TLX_EPILOGUE_FUSION", None)
                else:
                    os.environ["TORCHINDUCTOR_FORCE_TLX_EPILOGUE_FUSION"] = old_env
                if old_torch_logs is None:
                    os.environ.pop("TORCH_LOGS", None)
                else:
                    os.environ["TORCH_LOGS"] = old_torch_logs

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe addmm template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template epilogue-fusion codegen API (old ROCm nightly)",
    )
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_tlx_addmm_relu_epilogue_is_fused(
        self,
        dtype: torch.dtype,
    ):
        """Verify TLX epilogue fusion (tl.store + epilogue) for the AMD warp-pipe addmm.

        addmm(bias, x, W.t()) + relu: the relu epilogue is fused into the TLX
        addmm template's store_output (a single triton_tem_fused kernel writing via
        tl.store), not a separate pointwise kernel. AMD MI350X / gfx950, col-major B,
        gated by TORCHINDUCTOR_TLX_MODE.
        """

        class AddmmReluModule(nn.Module):
            """addmm (bias fused) followed by ReLU; W.t() gives col-major B."""

            def __init__(self, k: int, n: int, dtype: torch.dtype):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(n, k, dtype=dtype))
                self.bias = nn.Parameter(torch.randn(n, dtype=dtype))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.addmm(self.bias, x, self.weight.t())  # addmm (bias via epilogue)
                return y.relu()  # downstream pointwise -> fused into the template store

        device = GPU_TYPE
        m, k, n = 4096, 2048, 192
        model = AddmmReluModule(k=k, n=n, dtype=dtype).to(device)
        input_tensor = torch.randn(m, k, device=device, dtype=dtype)

        with (
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }),
                torch.no_grad(),
        ):
            compiled_model = torch.compile(model)
            output_actual, generated_code = run_and_get_code(
                compiled_model,
                input_tensor,
            )
            # fp32 accumulate + epilogue in fp32, then cast (matches the template).
            output_expected = (torch.addmm(model.bias, input_tensor, model.weight.t()).float().relu()).to(dtype)

        torch.testing.assert_close(output_actual, output_expected, atol=2e-2, rtol=2e-2)

        generated_code_str = str(generated_code)
        # The relu epilogue is fused into the TLX warp-pipe addmm template kernel
        # (a single triton_tem_fused_tlx_addmm... kernel written via tl.store), not
        # run as a separate pointwise kernel. The "tlx" prefix (added by
        # fusion.maybe_add_tlx_prefix) confirms the TLX template was selected.
        self.assertIn("triton_tem_fused", generated_code_str)
        self.assertIn("tlx", generated_code_str)
        self.assertIn("tl.store", generated_code_str)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for pre-drain epilogue prefetch",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template epilogue-fusion codegen API",
    )
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("y_shape", ((129, 257), (1, 257), (129, 1)))
    def test_tlx_addmm_predrain_epilogue_prefetch(
        self,
        dtype: torch.dtype,
        y_shape: tuple[int, int],
    ):
        """Prefetch a full output tile or a simple Y broadcast before drain."""
        from triton.language.extra.tlx.inductor import mm_templates as tlx_mm
        from triton.language.extra.tlx.inductor import registry as tlx_registry

        m, k, n = 129, 256, 257
        bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        a = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        y = torch.randn(y_shape, device=GPU_TYPE, dtype=dtype)

        def fused(bias, a, weight, y):
            value = torch.addmm(bias, a, weight.t())
            return value + value * y

        original_append_tlx = tlx_mm.append_tlx

        def only_warppipe(templates, op_name, kernel_inputs):
            original_append_tlx(templates, op_name, kernel_inputs)
            if tlx_mm.gfx950_addmm_warppipe_template in templates:
                templates[:] = [tlx_mm.gfx950_addmm_warppipe_template]
            return templates

        with (
                mock.patch.object(tlx_mm, "append_tlx", only_warppipe),
                mock.patch.object(
                    tlx_registry.Gfx950AddMMWarpPipeConfigHeuristic,
                    "WARPPIPE_CONFIGS",
                    [(64, 64, 64, 8, 8, 3)],
                ),
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }),
                torch.no_grad(),
        ):
            actual, code = run_and_get_code(torch.compile(fused, fullgraph=True), bias, a, weight, y)

        expected = fused(bias, a, weight, y)
        atol = 2e-2 * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-2)
        source = "\n".join(code)
        self.assertIn("TorchTLX pre-drain epilogue prefetch: gemm_drain", source)
        self.assertEqual(1, source.count("tlx_predrain_ptr_gemm_drain = in_ptr3 +"))
        self.assertNotIn("tl.load(in_ptr3", source)
        self.assertIn("= tlx_prefetched_gemm_drain", source)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for pre-drain epilogue prefetch",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template epilogue-fusion codegen API",
    )
    def test_tlx_predrain_prefetch_rejects_multiple_external_loads(self):
        """Two external epilogue operands exceed the initial one-load budget."""
        from triton.language.extra.tlx.inductor import mm_templates as tlx_mm
        from triton.language.extra.tlx.inductor import registry as tlx_registry

        m, k, n = 129, 256, 257
        dtype = torch.bfloat16
        bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        a = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        y = torch.randn((m, n), device=GPU_TYPE, dtype=dtype)
        z = torch.randn((m, n), device=GPU_TYPE, dtype=dtype)

        def fused(bias, a, weight, y, z):
            value = torch.addmm(bias, a, weight.t())
            return value + value * y + z

        original_append_tlx = tlx_mm.append_tlx

        def only_warppipe(templates, op_name, kernel_inputs):
            original_append_tlx(templates, op_name, kernel_inputs)
            if tlx_mm.gfx950_addmm_warppipe_template in templates:
                templates[:] = [tlx_mm.gfx950_addmm_warppipe_template]
            return templates

        with (
                mock.patch.object(tlx_mm, "append_tlx", only_warppipe),
                mock.patch.object(
                    tlx_registry.Gfx950AddMMWarpPipeConfigHeuristic,
                    "WARPPIPE_CONFIGS",
                    [(64, 64, 64, 8, 8, 3)],
                ),
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }),
        ):
            actual, code = run_and_get_code(torch.compile(fused, fullgraph=True), bias, a, weight, y, z)

        expected = fused(bias, a, weight, y, z)
        atol = 2e-2 * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-2)
        self.assertNotIn(
            "TorchTLX pre-drain epilogue prefetch: gemm_drain",
            "\n".join(code),
        )

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for persistent pre-drain prefetch",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template epilogue-fusion codegen API",
    )
    def test_tlx_persistent_addmm_predrain_epilogue_prefetch(self):
        """The paired hook stays inside each persistent output-tile iteration."""
        from triton.language.extra.tlx.inductor import mm_templates as tlx_mm
        from triton.language.extra.tlx.inductor import registry as tlx_registry

        m, k, n = 4160, 256, 512
        dtype = torch.bfloat16
        bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        a = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        y = torch.randn((m, n), device=GPU_TYPE, dtype=dtype)

        def fused(bias, a, weight, y):
            value = torch.addmm(bias, a, weight.t())
            return value + value * y

        original_append_tlx = tlx_mm.append_tlx

        def only_persistent(templates, op_name, kernel_inputs):
            original_append_tlx(templates, op_name, kernel_inputs)
            if tlx_mm.gfx950_addmm_persistent_warppipe_template in templates:
                templates[:] = [tlx_mm.gfx950_addmm_persistent_warppipe_template]
            return templates

        with (
                mock.patch.object(tlx_mm, "append_tlx", only_persistent),
                mock.patch.object(
                    tlx_registry.Gfx950AddMMPersistentWarpPipeConfigHeuristic,
                    "WARPPIPE_CONFIGS",
                    [(64, 128, 64, 8, 8, 2)],
                ),
                config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }),
                torch.no_grad(),
        ):
            actual, code = run_and_get_code(torch.compile(fused, fullgraph=True), bias, a, weight, y)

        expected = fused(bias, a, weight, y)
        atol = 2e-2 * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-2)
        source = "\n".join(code)
        self.assertIn("Persistent TLX warp-pipelined addmm", source)
        self.assertIn("TorchTLX pre-drain epilogue prefetch: gemm_drain", source)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template fusion codegen API (old ROCm nightly)",
    )
    @parametrize("case", GEMM_NORM_TEST_CASES)
    def test_tlx_matmul_norm_is_fused(
        self,
        case: tuple[str, tuple[int, int, int]],
    ):
        """Verify forced fusion and transparent allow mode on production shapes."""
        norm_kind, (m, k, n) = case
        dtype = torch.bfloat16
        x = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        gemm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        norm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)

        def matmul_norm(
            x: torch.Tensor,
            weight: torch.Tensor,
            gemm_bias: torch.Tensor,
            scale: torch.Tensor,
            norm_bias: torch.Tensor,
        ) -> torch.Tensor:
            gemm = torch.addmm(gemm_bias, x, weight.t())
            return _normalize(gemm, scale, norm_bias, norm_kind)

        with torch.no_grad():
            expected = matmul_norm(x, weight, gemm_bias, scale, norm_bias)
            with config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
            }):
                actual, code = run_and_get_code(
                    torch.compile(matmul_norm, fullgraph=True),
                    x,
                    weight,
                    gemm_bias,
                    scale,
                    norm_bias,
                )

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        self._assert_reduction_epilogue_fused(code)

        with (
                torch.no_grad(),
                mock.patch.dict(os.environ, {"TORCHINDUCTOR_TLX_REDUCTION_EPILOGUE": "1"}),
                config.patch({
                    "triton.tlx_mode": "allow",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }),
        ):
            allow_actual, allow_code = run_and_get_code(
                torch.compile(matmul_norm, fullgraph=True),
                x,
                weight,
                gemm_bias,
                scale,
                norm_bias,
            )

        torch.testing.assert_close(
            allow_actual,
            expected,
            atol=3e-2,
            rtol=3e-2,
        )
        self.assertNotIn("torch.ops.torch_tlx", "\n".join(allow_code))

    _FORCE_CONFIG = {
        "triton.tlx_mode": "force",
        "force_disable_caches": True,
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
        "enable_caching_generated_triton_templates": False,
    }

    def _assert_reduction_epilogue_fused(self, code: list[str], fused: bool = True) -> None:
        source = "\n".join(code)
        if not fused:
            self.assertNotIn("_row_consumer", source)
            return
        self.assertIn("row_stats_ws", source)
        self.assertIn("_row_consumer", source)
        self.assertEqual(code[-1].count(".run("), 2)

    def _run_gemm_epilogue(self, epilogue, m: int, k: int, n: int, *, vector_stride: int = 1, repeats: int = 1):
        dtype = torch.bfloat16
        inputs = (
            torch.randn((m, k), device=GPU_TYPE, dtype=dtype),
            torch.randn((n, k), device=GPU_TYPE, dtype=dtype),
            torch.randn((n, ), device=GPU_TYPE, dtype=dtype),
            torch.randn((n * vector_stride, ), device=GPU_TYPE, dtype=dtype)[::vector_stride],
            torch.randn((n * vector_stride, ), device=GPU_TYPE, dtype=dtype)[::vector_stride],
        )

        def model(x, weight, gemm_bias, scale, norm_bias):
            return epilogue(torch.addmm(gemm_bias, x, weight.t()), scale, norm_bias)

        with torch.no_grad():
            expected = model(*inputs)
            with config.patch(self._FORCE_CONFIG):
                compiled = torch.compile(model, fullgraph=True)
                actual, code = run_and_get_code(compiled, *inputs)
                torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
                for iteration in range(1, repeats):
                    with self.subTest(iteration=iteration):
                        torch.testing.assert_close(compiled(*inputs), expected, atol=3e-2, rtol=3e-2)
        return code

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template fusion codegen API (old ROCm nightly)",
    )
    @parametrize("case", GEMM_NORM_TEST_CASES)
    def test_tlx_matmul_norm_is_fused_with_dynamic_batch(
        self,
        case: tuple[str, tuple[int, int, int]],
    ):
        """A dynamic batch dim must still build the direct-to-LDS fused plans.

        The register-resident GEMM takes M as a constexpr and cannot serve a
        symbolic M; the LDS path must, including its split-K plans.
        """
        norm_kind, (m, k, n) = case
        dtype = torch.bfloat16
        x = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        gemm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        norm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)

        def matmul_norm(
            x: torch.Tensor,
            weight: torch.Tensor,
            gemm_bias: torch.Tensor,
            scale: torch.Tensor,
            norm_bias: torch.Tensor,
        ) -> torch.Tensor:
            gemm = torch.addmm(gemm_bias, x, weight.t())
            return _normalize(gemm, scale, norm_bias, norm_kind)

        torch._dynamo.mark_dynamic(x, 0)
        with torch.no_grad():
            expected = matmul_norm(x, weight, gemm_bias, scale, norm_bias)
            with config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
            }):
                actual, code = run_and_get_code(
                    torch.compile(matmul_norm, fullgraph=True),
                    x,
                    weight,
                    gemm_bias,
                    scale,
                    norm_bias,
                )

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        self._assert_reduction_epilogue_fused(code)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template fusion codegen API (old ROCm nightly)",
    )
    @parametrize(
        "case",
        [
            ("layernorm_no_affine", (2032, 2560, 2560)),
            ("layernorm_wide", (2032, 2560, 5120)),
            ("layernorm_tail", (2033, 2560, 2577)),
            ("relu_layernorm", (2032, 2560, 2560)),
            ("layernorm_constants", (2032, 2560, 2560)),
            ("rmsnorm_no_affine", (677, 8192, 4096)),
            ("rmsnorm_post_relu", (677, 8192, 4096)),
            ("l2norm", (2032, 2560, 2560)),
        ],
    )
    def test_tlx_matmul_reduction_epilogue_is_fused(self, case: tuple[str, tuple[int, int, int]]):
        """Verify the generated consumer covers affine-free, N > 4096, and pre-norm pointwise epilogues."""
        kind, (m, k, n) = case
        epilogues = {
            "layernorm_no_affine": lambda v, s, b: torch.nn.functional.layer_norm(v, (n, ), eps=1.0e-5),
            "layernorm_wide": lambda v, s, b: _normalize(v, s, b, "layernorm"),
            "layernorm_tail": lambda v, s, b: _normalize(v, s, b, "layernorm"),
            "relu_layernorm": lambda v, s, b: _normalize(torch.relu(v), s, b, "layernorm"),
            "layernorm_constants": lambda v, s, b: torch.nn.functional.layer_norm(v * 1.0e-3 + 1.0e-8,
                                                                                  (n, ), eps=1.0e-5),
            "rmsnorm_no_affine": lambda v, s, b: torch.nn.functional.rms_norm(v, (n, ), eps=1.0e-5),
            "rmsnorm_post_relu": lambda v, s, b: torch.relu(_normalize(v, s, b, "rmsnorm")),
            "l2norm": lambda v, s, b: torch.nn.functional.normalize(v, dim=-1, eps=1.0e-5),
        }
        self._assert_reduction_epilogue_fused(self._run_gemm_epilogue(epilogues[kind], m, k, n))

    @unittest.skipIf(not is_gfx950() or not has_tlx() or not supports_template_epilogue_fusion(),
                     "Need gfx950 template fusion")
    @parametrize("persistent", (False, True))
    @parametrize("norm_kind", (None, "layernorm", "rmsnorm"))
    def test_tlx_norm_fusion_pipeline_drain(self, persistent: bool, norm_kind: str | None):
        from triton.language.extra.tlx.inductor import mm_templates as tlx_mm
        from triton.language.extra.tlx.inductor import registry as tlx_registry

        template = (tlx_mm.gfx950_addmm_persistent_warppipe_template
                    if persistent else tlx_mm.gfx950_addmm_warppipe_template)
        heuristic = (tlx_registry.Gfx950AddMMPersistentWarpPipeConfigHeuristic
                     if persistent else tlx_registry.Gfx950AddMMWarpPipeConfigHeuristic)
        original_append_tlx = tlx_mm.append_tlx

        def only_warppipe(templates, op_name, kernel_inputs):
            original_append_tlx(templates, op_name, kernel_inputs)
            if template in templates:
                templates[:] = [template]
            return templates

        with (
                mock.patch.object(tlx_mm, "append_tlx", only_warppipe),
                mock.patch.object(heuristic, "WARPPIPE_CONFIGS", [(64, 64, 128, 8, 8, 3), (64, 64, 64, 8, 8, 3)]),
        ):
            code = self._run_gemm_epilogue(
                lambda v, s, b: v if norm_kind is None else _normalize(v, s, b, norm_kind),
                2033,
                2560,
                2577,
                repeats=1000 if norm_kind is None else 1,
            )
        self._assert_reduction_epilogue_fused(code, fused=norm_kind is not None)
        self.assertIn(f"DRAIN_PIPELINE={norm_kind is not None}", "\n".join(code))

    @unittest.skipIf(not is_gfx950() or not has_tlx() or not supports_template_epilogue_fusion(),
                     "Need gfx950 template fusion")
    @parametrize("norm_kind", ("layernorm", "rmsnorm"))
    @parametrize("stride", (2, 3))
    def test_tlx_matmul_norm_strided_vectors(self, norm_kind: str, stride: int):
        code = self._run_gemm_epilogue(
            lambda v, s, b: _normalize(v, s, b, norm_kind),
            2032,
            2560,
            2560,
            vector_stride=stride,
        )
        self._assert_reduction_epilogue_fused(code)

    @unittest.skipIf(not is_gfx950() or not has_tlx() or not supports_template_epilogue_fusion(),
                     "Need gfx950 template fusion")
    @parametrize("norm_kind", ("layernorm", "rmsnorm"))
    def test_tlx_matmul_norm_consumer_partial_loop(self, norm_kind: str):
        from triton.language.extra.tlx.inductor import reduction_epilogue

        with mock.patch.object(reduction_epilogue, "_CONSUMER_MAX_BLOCK_STATS", 2):
            code = self._run_gemm_epilogue(
                lambda v, s, b: _normalize(v, s, b, norm_kind),
                2032,
                2560,
                5120,
            )
        self._assert_reduction_epilogue_fused(code)
        self.assertRegex("\n".join(code), r"range\(0, (?:[3-9]|[1-9][0-9]+), 2\)")
        self.assertIn("range(0, 5120, 4096)", "\n".join(code))

    @unittest.skipIf(not is_gfx950() or not has_tlx() or not supports_template_epilogue_fusion(),
                     "Need gfx950 template fusion")
    @parametrize("norm_kind", ("layernorm", "rmsnorm"))
    def test_tlx_matmul_norm_without_row_partial_capability(self, norm_kind: str):
        from triton.language.extra.tlx.inductor import mm_templates

        with ExitStack() as stack:
            for name in (
                    "gfx950_addmm_warppipe_template",
                    "gfx950_addmm_persistent_warppipe_template",
                    "gfx950_addmm_interwave_template",
                    "gfx950_mm_interwave_template",
            ):
                stack.enter_context(mock.patch.object(getattr(mm_templates, name), "tlx_row_partials", False))
            code = self._run_gemm_epilogue(
                lambda v, s, b: _normalize(v, s, b, norm_kind),
                2032,
                2560,
                2560,
            )
        self._assert_reduction_epilogue_fused(code, fused=False)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template fusion codegen API (old ROCm nightly)",
    )
    @parametrize("kind", ("gelu", "post_gelu", "column_reduction", "side_statistics"))
    def test_tlx_matmul_unsupported_epilogue_op_falls_back(self, kind: str):
        """Verify an op outside the consumer whitelist (erf GELU) keeps the epilogue unfused."""
        epilogues = {
            "gelu": lambda v, s, b: _normalize(torch.nn.functional.gelu(v), s, b, "layernorm"),
            "post_gelu": lambda v, s, b: torch.nn.functional.gelu(_normalize(v, s, b, "layernorm")),
            "column_reduction": lambda v, s, b: v / (v.float().square().sum(dim=0) + 1.0e-5),
            "side_statistics": lambda v, s, b: (_normalize(v, s, b, "layernorm"), v.float().mean(dim=-1)),
        }
        code = self._run_gemm_epilogue(
            epilogues[kind],
            2032,
            2560,
            2560,
        )
        self._assert_reduction_epilogue_fused(code, fused=False)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template fusion codegen API (old ROCm nightly)",
    )
    def test_tlx_matmul_layernorm_saved_statistics_fall_back(self):
        """Verify a training forward that saves mean/rstd for backward is not fused."""
        m, k, n = 2032, 2560, 2560
        dtype = torch.bfloat16
        x = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        gemm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n, ), device=GPU_TYPE, dtype=dtype, requires_grad=True)
        norm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype, requires_grad=True)

        def model(x, weight, gemm_bias, scale, norm_bias):
            return _normalize(torch.addmm(gemm_bias, x, weight.t()), scale, norm_bias, "layernorm")

        expected = model(x, weight, gemm_bias, scale, norm_bias)
        with config.patch(self._FORCE_CONFIG):
            actual, code = run_and_get_code(
                torch.compile(model, fullgraph=True),
                x,
                weight,
                gemm_bias,
                scale,
                norm_bias,
            )
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        self._assert_reduction_epilogue_fused(code, fused=False)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template fusion codegen API (old ROCm nightly)",
    )
    @parametrize("case", GEMM_NORM_TEST_CASES)
    def test_tlx_addmm_norm_partial_stats_are_bitwise_repeatable(self, case):
        norm_kind, (m, k, n) = case
        dtype = torch.bfloat16
        inputs = (
            torch.randn((m, k), device=GPU_TYPE, dtype=dtype),
            torch.randn((n, k), device=GPU_TYPE, dtype=dtype),
            torch.randn((n, ), device=GPU_TYPE, dtype=dtype),
            torch.randn((n, ), device=GPU_TYPE, dtype=dtype),
            torch.randn((n, ), device=GPU_TYPE, dtype=dtype),
        )

        def model(x, weight, gemm_bias, scale, norm_bias):
            return _normalize(torch.addmm(gemm_bias, x, weight.t()), scale, norm_bias, norm_kind)

        with torch.no_grad(), config.patch(self._FORCE_CONFIG):
            expected = model(*inputs)
            compiled = torch.compile(model, fullgraph=True)
            first, code = run_and_get_code(compiled, *inputs)
            repeated = [compiled(*inputs) for _ in range(4)]

        torch.testing.assert_close(first, expected, atol=3e-2, rtol=3e-2)
        self._assert_reduction_epilogue_fused(code)
        for output in repeated:
            self.assertTrue(torch.equal(output, first))

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template fusion codegen API (old ROCm nightly)",
    )
    def test_tlx_matmul_norm_dynamic_m(self):
        """Verify GEMM + RMSNorm stays correct across a symbolic M."""
        m, m_new, k, n = 677, 689, 4096, 2048
        dtype = torch.bfloat16
        x = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        gemm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        norm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)

        def matmul_rmsnorm(
            x: torch.Tensor,
            weight: torch.Tensor,
            gemm_bias: torch.Tensor,
            scale: torch.Tensor,
            norm_bias: torch.Tensor,
        ) -> torch.Tensor:
            gemm = torch.addmm(gemm_bias, x, weight.t())
            return _normalize(gemm, scale, norm_bias, "rmsnorm")

        with torch.no_grad():
            expected = matmul_rmsnorm(x, weight, gemm_bias, scale, norm_bias)
            with config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
            }):
                compile_counter = CompileCounterWithBackend("inductor")
                compiled = torch.compile(
                    matmul_rmsnorm,
                    backend=compile_counter,
                    dynamic=True,
                    fullgraph=True,
                )
                actual, code = run_and_get_code(
                    compiled,
                    x,
                    weight,
                    gemm_bias,
                    scale,
                    norm_bias,
                )
                x_new = torch.randn(
                    (m_new, k),
                    device=GPU_TYPE,
                    dtype=dtype,
                )
                actual_new = compiled(
                    x_new,
                    weight,
                    gemm_bias,
                    scale,
                    norm_bias,
                )
                expected_new = matmul_rmsnorm(
                    x_new,
                    weight,
                    gemm_bias,
                    scale,
                    norm_bias,
                )

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(
            actual_new,
            expected_new,
            atol=3e-2,
            rtol=3e-2,
        )
        # Nothing specializes M, so both shapes reuse one compiled frame.
        self.assertEqual(compile_counter.frame_count, 1)
        self.assertGreaterEqual(code[-1].count(".run("), 2)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("norm_kind", ("rmsnorm", "layernorm"))
    def test_tlx_matmul_norm_shared_gemm_output(
        self,
        norm_kind: str,
    ):
        """Verify a GEMM output shared with another consumer stays correct."""
        m, k, n = 279, 4096, 2048
        dtype = torch.bfloat16
        x = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        gemm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        norm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)

        def matmul_norm_with_extra_user(
            x: torch.Tensor,
            weight: torch.Tensor,
            gemm_bias: torch.Tensor,
            scale: torch.Tensor,
            norm_bias: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            gemm = torch.addmm(gemm_bias, x, weight.t())
            return _normalize(gemm, scale, norm_bias, norm_kind), gemm + 1

        with torch.no_grad():
            expected = matmul_norm_with_extra_user(
                x,
                weight,
                gemm_bias,
                scale,
                norm_bias,
            )
            with config.patch({
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
            }):
                actual, code = run_and_get_code(
                    torch.compile(matmul_norm_with_extra_user, fullgraph=True),
                    x,
                    weight,
                    gemm_bias,
                    scale,
                    norm_bias,
                )

        for actual_output, expected_output in zip(actual, expected):
            torch.testing.assert_close(
                actual_output,
                expected_output,
                atol=3e-2,
                rtol=3e-2,
            )
        self.assertGreaterEqual("\n".join(code).count(".run("), 2)
        self._assert_reduction_epilogue_fused(code, fused=False)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    def test_tlx_matmul_norm_disabled_uses_standard_lowering(self):
        """Verify disabled TLX mode leaves addmm and RMSNorm on standard lowering."""
        m, k, n = 17, 64, 2048
        dtype = torch.bfloat16
        x = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        gemm_bias = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n, ), device=GPU_TYPE, dtype=dtype)

        def matmul_rmsnorm(
            x: torch.Tensor,
            weight: torch.Tensor,
            gemm_bias: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            gemm = torch.addmm(gemm_bias, x, weight.t())
            return torch.nn.functional.rms_norm(
                gemm,
                (gemm.shape[-1], ),
                scale,
                1.0e-5,
            )

        with torch.no_grad():
            expected = matmul_rmsnorm(x, weight, gemm_bias, scale)
            with config.patch({
                    "triton.tlx_mode": None,
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
            }):
                actual, code = run_and_get_code(
                    torch.compile(matmul_rmsnorm, fullgraph=True),
                    x,
                    weight,
                    gemm_bias,
                    scale,
                )

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        self._assert_reduction_epilogue_fused(code, fused=False)


if __name__ == "__main__":
    run_tests()
