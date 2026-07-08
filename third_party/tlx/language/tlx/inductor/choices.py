"""
TLXInductorChoices: InductorChoices subclass that consolidates all TLX
template selection, filtering, and layout logic.

Activated via config.inductor_choices_class set in template_heuristics/tlx.py.
All methods check config.triton.tlx_mode at runtime and fall through to
super() when mode is "default".
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.select_algorithm import ExternKernelChoice

from . import tlx_config

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch._inductor.codegen.common import KernelTemplate
    from torch._inductor.ir import ChoiceCaller
    from torch._inductor.kernel_inputs import KernelInputs
    from torch._inductor.kernel_template_choice import KernelTemplateChoice

log = logging.getLogger(__name__)


class TLXInductorChoices(InductorChoices):
    """
    InductorChoices subclass that handles TLX template injection,
    filtering, and layout fixing.

    In "default" mode: delegates entirely to the parent class.
    In "allow" mode: TLX templates compete with other backends via autotuning.
    In "force" mode: only TLX templates are used.
    """

    def uuid(self) -> str:
        return "TLXInductorChoices"

    def _finalize_template_configs(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        templates: list[KernelTemplate | ExternKernelChoice],
        op_name: str,
        kwarg_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[KernelTemplateChoice]:
        if config.triton.tlx_mode == "force":
            # In force mode, keep only TLX templates
            return [
                c for uid, gen in template_choices.items() if "tlx_" in uid for c in gen
            ]

        return super()._finalize_template_configs(
            template_choices, kernel_inputs, templates, op_name, kwarg_overrides
        )

    def customize_fused_kernel_name(self, fused_name: str, src_code: str) -> str:
        from .fusion import maybe_add_tlx_prefix

        return maybe_add_tlx_prefix(fused_name, src_code)

    def override_best_choice(
        self,
        best_choice: ChoiceCaller,
        timings: dict[ChoiceCaller, float],
    ) -> ChoiceCaller:
        return maybe_override_best_choice(best_choice, timings)

    def _need_to_fix_layout(
        self,
        adjusted_choices: list[KernelTemplateChoice],
        op_name: str,
    ) -> bool:
        if config.triton.tlx_mode == "force":
            return True
        return super()._need_to_fix_layout(adjusted_choices, op_name)

    # Ops that use prefix_args (bias passed separately from mat1/mat2).
    # The blackwell WS template only supports plain mm (2 named kernel args: A, B),
    # so skip TLX injection for these ops. addmm IS supported: the AMD warp-pipe
    # template handles the bias epilogue (prefix_args=1), injected via append_tlx.
    _UNSUPPORTED_OPS = frozenset({"baddbmm"})

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        templates: list[KernelTemplate | ExternKernelChoice],
        op_name: str,
        kwarg_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[ChoiceCaller]:
        tlx_mode = config.triton.tlx_mode
        if tlx_mode == "default" or op_name in self._UNSUPPORTED_OPS:
            return super().get_template_configs(
                kernel_inputs, templates, op_name, kwarg_overrides
            )

        from .mm_templates import append_tlx

        templates = list(templates)
        kwarg_overrides = kwarg_overrides or {}
        append_tlx(templates, op_name)

        input_tensors = kernel_inputs.nodes()
        if len(input_tensors) < 2:
            raise ValueError(f"Need at least 2 input tensors, got {len(input_tensors)}")

        template_choices = {
            t.uid: self.get_ktc(
                kernel_inputs, t, op_name, kwarg_overrides.get(t.uid, {})
            )
            for t in templates
        }

        # Second pass: Adjust the template choices
        adjusted_choices = self._finalize_template_configs(
            template_choices,
            kernel_inputs,
            templates,
            op_name,
            kwarg_overrides,
        )

        # Fix layouts: force mode fixes all, allow mode fixes non-extern only
        fix_all = self._need_to_fix_layout(adjusted_choices, op_name)
        if fix_all or tlx_mode == "allow":
            fixed_layout = kernel_inputs.output_layout(flexible=False)
            for ktc in adjusted_choices:
                needs_fix = fix_all or not isinstance(ktc.template, ExternKernelChoice)
                if needs_fix:
                    ktc.layout = fixed_layout
                    if hasattr(ktc, "_choice"):
                        del ktc._choice

        # Third pass: Convert to ChoiceCaller objects
        return [ktc.choice for ktc in adjusted_choices if ktc.choice is not None]

    def append_flex_attention_choices(
        self,
        choices: list[Any],
        configs: list[Any],
        input_nodes: list[Any],
        subgraphs: list[Any],
        layout: Any,
        kernel_options: dict[str, Any],
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
    ) -> list[Any]:
        if config.triton.tlx_mode == "default":
            return super().append_flex_attention_choices(
                choices,
                configs,
                input_nodes,
                subgraphs,
                layout,
                kernel_options,
                sparse_q_block_size,
                sparse_kv_block_size,
            )

        from .flex_attention_templates import (
            append_tlx_flex_attention_choice,
        )

        append_tlx_flex_attention_choice(
            choices,
            configs,
            input_nodes,
            subgraphs,
            layout,
            kernel_options,
            sparse_q_block_size,
            sparse_kv_block_size,
        )
        return choices


def maybe_override_best_choice(
    best_choice: ChoiceCaller,
    timings: dict[ChoiceCaller, float],
) -> ChoiceCaller:
    """
    In TLX "allow" mode, prefer extern kernels (cublas) unless a TLX template
    beats them by a meaningful margin. Short autotuning benchmarks are noisy
    and TLX can narrowly "win" but run slower in steady-state.

    Returns the (possibly overridden) best choice.
    """
    from .fusion import _is_tlx_choice

    if config.triton.tlx_mode != "allow" or not _is_tlx_choice(best_choice):
        return best_choice

    from torch._inductor.select_algorithm import ExternKernelCaller

    extern_choices = [c for c in timings if isinstance(c, ExternKernelCaller)]
    if not extern_choices:
        return best_choice

    tlx_threshold = tlx_config.allow_min_speedup

    best_extern_time = min(timings[c] for c in extern_choices)
    best_time = timings[best_choice]
    speedup = best_extern_time / best_time if best_time > 0 else 0
    if speedup < tlx_threshold:
        log.debug(
            "TLX choice %s speedup %.2fx < threshold %.2fx over extern, using extern",
            best_choice.name,
            speedup,
            tlx_threshold,
        )
        return min(extern_choices, key=lambda c: timings[c])

    return best_choice
