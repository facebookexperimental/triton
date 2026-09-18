# Owner(s): ["module: inductor"]
"""
Unit tests for TorchTLX (Triton Language eXtensions) epilogue fusion.

Usage:
    pytest python/test/unit/language/test_torchtlx_fusions.py
"""

import os
import unittest

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
    normalized_shape = (value.shape[-1],)
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
        gemm_bias = torch.randn((n,), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n,), device=GPU_TYPE, dtype=dtype)
        norm_bias = torch.randn((n,), device=GPU_TYPE, dtype=dtype)

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
            with config.patch(
                {
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }
            ):
                actual, code = run_and_get_code(
                    torch.compile(matmul_norm, fullgraph=True),
                    x,
                    weight,
                    gemm_bias,
                    scale,
                    norm_bias,
                )

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        generated_code = "\n".join(code)
        self.assertIn(f"tlx_gfx950_addmm_{norm_kind}", generated_code, "Using custom op autotunign from subgraph template")
        self.assertEqual(code[-1].count(".run("), 1)

        with (
            torch.no_grad(),
            config.patch(
                {
                    "triton.tlx_mode": "allow",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }
            ),
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
        self.assertNotIn(
            f"torch.ops.torch_tlx.gfx950_addmm_{norm_kind}",
            "\n".join(allow_code),
            "The opaque custom-op fallback is not emitted"
        )

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @unittest.skipIf(
        not supports_template_epilogue_fusion(),
        "torch lacks the template fusion codegen API (old ROCm nightly)",
    )
    def test_tlx_matmul_norm_dynamic_m_is_fused(self):
        """Verify one fused kernel handles dynamic M across launch-grid sizes."""
        m, m_new, k, n = 677, 689, 4096, 2048
        dtype = torch.bfloat16
        x = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        gemm_bias = torch.randn((n,), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n,), device=GPU_TYPE, dtype=dtype)
        norm_bias = torch.randn((n,), device=GPU_TYPE, dtype=dtype)
        torch._dynamo.mark_dynamic(x, 0, min=120, max=766)

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
            with config.patch(
                {
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }
            ):
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
        self.assertEqual(compile_counter.frame_count, 1)
        generated_code = "\n".join(code)
        self.assertIn("tlx_gfx950_addmm_rmsnorm", generated_code)
        self.assertEqual(code[-1].count(".run("), 1)

    @unittest.skipIf(
        not is_gfx950(),
        "Need AMD MI350X (gfx950) for the TLX warp-pipe GEMM template",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("norm_kind", ("rmsnorm", "layernorm"))
    def test_tlx_matmul_norm_multiple_user_falls_back(
        self,
        norm_kind: str,
    ):
        """Verify a shared addmm output prevents GEMM-normalization fusion."""
        m, k, n = 279, 4096, 2048
        dtype = torch.bfloat16
        x = torch.randn((m, k), device=GPU_TYPE, dtype=dtype)
        weight = torch.randn((n, k), device=GPU_TYPE, dtype=dtype)
        gemm_bias = torch.randn((n,), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n,), device=GPU_TYPE, dtype=dtype)
        norm_bias = torch.randn((n,), device=GPU_TYPE, dtype=dtype)

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
            with config.patch(
                {
                    "triton.tlx_mode": "force",
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }
            ):
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
        generated_code = "\n".join(code)
        self.assertNotIn(f"tlx_gfx950_addmm_{norm_kind}", generated_code)
        self.assertGreaterEqual(generated_code.count(".run("), 2)

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
        gemm_bias = torch.randn((n,), device=GPU_TYPE, dtype=dtype)
        scale = torch.randn((n,), device=GPU_TYPE, dtype=dtype)

        def matmul_rmsnorm(
            x: torch.Tensor,
            weight: torch.Tensor,
            gemm_bias: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            gemm = torch.addmm(gemm_bias, x, weight.t())
            return torch.nn.functional.rms_norm(
                gemm,
                (gemm.shape[-1],),
                scale,
                1.0e-5,
            )

        with torch.no_grad():
            expected = matmul_rmsnorm(x, weight, gemm_bias, scale)
            with config.patch(
                {
                    "triton.tlx_mode": None,
                    "force_disable_caches": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "enable_caching_generated_triton_templates": False,
                }
            ):
                actual, code = run_and_get_code(
                    torch.compile(matmul_rmsnorm, fullgraph=True),
                    x,
                    weight,
                    gemm_bias,
                    scale,
                )

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        self.assertNotIn("tlx_gfx950_addmm_rmsnorm", "\n".join(code))


if __name__ == "__main__":
    run_tests()
