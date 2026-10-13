# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

import unittest
from unittest import TestCase

import torch
import triton
import triton.language as tl


def is_cuda():
    return torch.cuda.is_available() and triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hopper_or_newer():
    return is_cuda() and triton.runtime.driver.active.get_current_target().arch >= 90

def _get_device():
    return triton.runtime.driver.active.get_active_torch_device()

_CLUSTER_SCALE = tl.constexpr(1.0)

class TestCtasPerCgaAutotunerSteadyState(TestCase):
    """Autotuner steady-state path must preserve ctas_per_cga.

    For single-config autotune, _last_key is never set (only the multi-config
    branch sets it), so _seed_key is always None. After the first call seeds
    the C cache (adding None to _fc_seeded), every subsequent call goes through
    the steady-state path: self.fn[evaluated_grid](*full_args), which does NOT
    pass the config's compilation options.

    When __getitem__ returns the lambda fallback (e.g. because used_global_vals
    is non-empty), run() is called without ctas_per_cga. Without the
    _fc_meta_kwargs fallback in JITFunction.run(), recompilation drops the
    cluster config and cuLaunchKernelEx fails with error 912.
    """

    @unittest.skipUnless(is_hopper_or_newer(), "ctas_per_cga is NVIDIA-only and requires Hopper (sm90) or newer")
    def test_second_call_preserves_cluster_config(self):

        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 128},
                    num_warps=4,
                    num_stages=1,
                    ctas_per_cga=(2, 1, 1),
                ),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel_2cta(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            scale = _CLUSTER_SCALE  # noqa: F841  — forces used_global_vals
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, (x + y).to(tl.float32), mask=mask)

        device = _get_device()
        n = 1024

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_SIZE"]), )

        # Call 1: seeds the C cache (_seed_key=None not in _fc_seeded).
        x1 = torch.randn(n, device=device, dtype=torch.float32)
        y1 = torch.randn(n, device=device, dtype=torch.float32)
        o1 = torch.empty(n, device=device, dtype=torch.float32)
        add_kernel_2cta[grid](x1, y1, o1, n)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(o1, x1 + y1))

        # Call 2: steady-state path (_seed_key=None already in _fc_seeded).
        # Without the _fc_meta_kwargs fix this recompiles without
        # ctas_per_cga and fails with CUDA error 912.
        x2 = torch.randn(n, device=device, dtype=torch.float32)
        y2 = torch.randn(n, device=device, dtype=torch.float32)
        o2 = torch.empty(n, device=device, dtype=torch.float32)
        add_kernel_2cta[grid](x2, y2, o2, n)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(o2, x2 + y2))
