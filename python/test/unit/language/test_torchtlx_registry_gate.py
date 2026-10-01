# Owner(s): ["module: inductor"]
"""Import-gate tests for the torchTLX inductor integration.

Each subprocess case runs in a fresh interpreter: module import is
process-global, so the enabled/disabled paths cannot be covered in one process.
CPU-only; no GPU or autotune required.
"""

import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from triton.language.extra.tlx.inductor import tlx_config


def has_tlx() -> bool:
    try:
        import triton.language.extra.tlx  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipIf(not has_tlx(), "TLX not available")
class TestTLXRegistryGate(TestCase):
    def _run_probe(self, stmt: str, env_overrides: dict[str, str]) -> str:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("TRITON_ENABLE_TORCH_TLX", "TORCHINDUCTOR_TLX_MODE")
        }
        env.update(env_overrides)
        proc = subprocess.run(
            [sys.executable, "-c", stmt],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        return proc.stdout.strip()

    def test_disabled_by_default(self) -> None:
        # Bare import stays cheap: no impl load, no torch import. Explicit
        # attribute access loads on demand (see
        # test_disabled_attr_access_loads_on_demand) and is not exercised here.
        out = self._run_probe(
            "from triton.language.extra.tlx.inductor import registry;"
            "import sys;"
            "print(registry.tlx_only_cuda_options);"
            "print('registry_impl' in ','.join(sys.modules));"
            "print('torch' in sys.modules)",
            {},
        )
        self.assertEqual(out, "[]\nFalse\nFalse")

    def test_disabled_shared_list_identity(self) -> None:
        out = self._run_probe(
            "from triton.language.extra.tlx.inductor import registry;"
            "from triton.language.extra.tlx.inductor import tlx_config;"
            "print(registry.tlx_only_cuda_options is tlx_config.tlx_only_cuda_options);"
            "print(registry.tlx_only_cuda_options)",
            {},
        )
        self.assertEqual(out, "True\n[]")

    def test_disabled_attr_access_loads_on_demand(self) -> None:
        out = self._run_probe(
            "from triton.language.extra.tlx.inductor import registry\n"
            "import sys\n"
            "print(registry.tlx_only_cuda_options)\n"
            "print(callable(registry.get_heuristic_config))\n"
            "print('triton.language.extra.tlx.inductor.registry_impl' in sys.modules)\n"
            "print(registry.tlx_only_cuda_options)\n"
            "from triton.language.extra.tlx.inductor import tlx_config\n"
            "print(registry.tlx_only_cuda_options is tlx_config.tlx_only_cuda_options)",
            {},
        )
        self.assertEqual(
            out, "[]\nTrue\nTrue\n['ctas_per_cga']\nTrue"
        )

    def test_disabled_leaves_inductor_untouched(self) -> None:
        out = self._run_probe(
            "from triton.language.extra.tlx.inductor import registry;"
            "from torch._inductor import config;"
            "from torch._inductor.select_algorithm import TritonTemplateKernel;"
            "print(getattr(config, 'inductor_choices_class', None));"
            "print(TritonTemplateKernel.__init__.__module__);"
            "from torch._inductor.utils import tlx_only_cuda_options as f;"
            "print(f())",
            {},
        )
        self.assertEqual(out, "None\ntorch._inductor.select_algorithm\n[]")

    def test_enabled_via_master_switch(self) -> None:
        out = self._run_probe(
            "from triton.language.extra.tlx.inductor import registry;"
            "import sys;"
            "print(registry.tlx_only_cuda_options);"
            "print('registry_impl' in ','.join(sys.modules));"
            "print(callable(registry.get_heuristic_config));"
            "from torch._inductor import config;"
            "print(config.inductor_choices_class is not None)",
            {"TRITON_ENABLE_TORCH_TLX": "1"},
        )
        self.assertEqual(out, "['ctas_per_cga']\nTrue\nTrue\nTrue")

    def test_enabled_via_tlx_mode_env(self) -> None:
        out = self._run_probe(
            "from triton.language.extra.tlx.inductor import registry;"
            "print(registry.tlx_only_cuda_options)",
            {"TORCHINDUCTOR_TLX_MODE": "allow"},
        )
        self.assertEqual(out, "['ctas_per_cga']")

    def test_enabled_via_config_patch(self) -> None:
        out = self._run_probe(
            "from torch._inductor import config\n"
            "with config.patch({'triton.tlx_mode': 'force'}):\n"
            "    from triton.language.extra.tlx.inductor import registry\n"
            "    print(registry.tlx_only_cuda_options)",
            {},
        )
        self.assertEqual(out, "['ctas_per_cga']")

    def test_late_enable_repairs_option_list_in_place(self) -> None:
        out = self._run_probe(
            "import os;"
            "from triton.language.extra.tlx.inductor import tlx_config;"
            "from torch._inductor.utils import tlx_only_cuda_options as f;"
            "print(f());"
            "os.environ['TRITON_ENABLE_TORCH_TLX'] = '1';"
            "from triton.language.extra.tlx.inductor import registry;"
            "print(registry.tlx_only_cuda_options);"
            "print(f());"
            "print(f() is tlx_config.tlx_only_cuda_options)",
            {},
        )
        self.assertEqual(out, "[]\n['ctas_per_cga']\n['ctas_per_cga']\nTrue")

    def test_knob_default_off(self) -> None:
        out = self._run_probe(
            "from triton import knobs;"
            "print(knobs.tlx.enable_torch_tlx)",
            {},
        )
        self.assertEqual(out, "False")

    def test_knob_on_via_env(self) -> None:
        out = self._run_probe(
            "from triton import knobs;"
            "print(knobs.tlx.enable_torch_tlx)",
            {"TRITON_ENABLE_TORCH_TLX": "1"},
        )
        self.assertEqual(out, "True")

    def test_torch_tlx_enabled_decision(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("TRITON_ENABLE_TORCH_TLX", None)
            os.environ.pop("TORCHINDUCTOR_TLX_MODE", None)
            self.assertFalse(tlx_config.torch_tlx_enabled())
            for truthy in ("1", "true", "TRUE", "yes", "on", "y"):
                os.environ["TRITON_ENABLE_TORCH_TLX"] = truthy
                self.assertTrue(
                    tlx_config.torch_tlx_enabled(), msg=f"switch={truthy!r}"
                )
            for falsy in ("0", "false", "", "2", "yes please"):
                os.environ["TRITON_ENABLE_TORCH_TLX"] = falsy
                self.assertFalse(
                    tlx_config.torch_tlx_enabled(), msg=f"falsy={falsy!r}"
                )
            del os.environ["TRITON_ENABLE_TORCH_TLX"]
            os.environ["TORCHINDUCTOR_TLX_MODE"] = "force"
            self.assertTrue(tlx_config.torch_tlx_enabled())
            os.environ["TORCHINDUCTOR_TLX_MODE"] = "default"
            self.assertFalse(tlx_config.torch_tlx_enabled())
            del os.environ["TORCHINDUCTOR_TLX_MODE"]
            with config.patch({"triton.tlx_mode": "allow"}):
                self.assertTrue(tlx_config.torch_tlx_enabled())


if __name__ == "__main__":
    run_tests()
