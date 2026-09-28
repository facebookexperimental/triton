"""Autotuned semantic-subgraph integration helpers for TorchTLX."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._inductor import config
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, ir_node_to_tensor
from torch._inductor.virtualized import V
from torch.utils._pytree import tree_flatten


class _MultiOutputSubgraphTemplate(SubgraphTemplate):
    """SubgraphTemplate variant that validates and tunes tensor pytrees.

    Inductor's current custom-op autotuner assumes a single tensor output even
    though decomposition graphs and their benchmark callables can return
    tuples.  The output passed to the algorithm selector is only a scratch
    allocation for subgraph choices, so use the first tensor leaf as that
    representative while retaining and validating the complete output ABI.
    The selected graph is inlined and returns the original pytree.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._output_signatures: list[tuple[str, tuple[tuple[object, ...], ...]]] = []

    @staticmethod
    def _layout_signature(layout: FixedLayout) -> tuple[object, ...]:
        return (
            layout.device,
            layout.dtype,
            tuple(map(str, layout.size)),
            tuple(map(str, layout.stride)),
        )

    def _infer_custom_op_layout(
        self,
        input_nodes: list[Buffer],
        function_decomposition: Callable[..., Any],
        kwargs: dict[str, Any],
        default_impl: Callable[..., Any] | None = None,
        input_gen_fns: dict[int, Callable[[Any], torch.Tensor]] | None = None,
    ) -> FixedLayout:
        del default_impl
        self._validate_non_tensor_kwargs(kwargs)

        with V.fake_mode:
            example_inputs = []
            for i, inp in enumerate(input_nodes):
                if input_gen_fns and i in input_gen_fns:
                    fake_tensor = input_gen_fns[i](inp)
                else:
                    raw_shape = inp.get_size()
                    concrete_shape = V.graph.sizevars.optimization_hints(raw_shape)
                    raw_stride = inp.get_stride()
                    concrete_stride = V.graph.sizevars.optimization_hints(raw_stride)
                    fake_tensor = torch.empty_strided(
                        concrete_shape,
                        concrete_stride,
                        dtype=inp.get_dtype(),
                        device=inp.get_device(),
                    )
                example_inputs.append(fake_tensor)

            output = functools.partial(function_decomposition, **kwargs)(
                *example_inputs
            )

        leaves, spec = tree_flatten(output)
        if not leaves or any(not isinstance(leaf, torch.Tensor) for leaf in leaves):
            raise AssertionError(
                "TorchTLX subgraph candidates must return a tensor or a pytree "
                "containing only tensors"
            )

        layouts = [
            FixedLayout(
                device=leaf.device,
                dtype=leaf.dtype,
                size=leaf.shape,
                stride=leaf.stride(),
            )
            for leaf in leaves
        ]
        self._output_signatures.append(
            (
                repr(spec),
                tuple(self._layout_signature(layout) for layout in layouts),
            )
        )
        return layouts[0]

    def _validate_layout_equivalence(
        self,
        op_name: str,
        decompositions: list[Callable[..., Any]],
        layouts: list[FixedLayout],
    ) -> None:
        super()._validate_layout_equivalence(op_name, decompositions, layouts)
        reference = self._output_signatures[0]
        for index, signature in enumerate(self._output_signatures[1:], start=1):
            if signature != reference:
                raise AssertionError(
                    f"Output mismatch in custom op '{op_name}': decomposition "
                    f"'{decompositions[index].__name__}' does not match "
                    f"'{decompositions[0].__name__}'"
                )


def _autotune_without_opaque_fallback(
    *,
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    non_tensor_args: list[dict[str, Any]],
    op_overload: torch._ops.OpOverload,
    user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None,
    config_patches_list: list[dict[str, Any]],
    benchmark_with_cudagraphs: bool,
) -> Any:
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    from torch._inductor.kernel import custom_op as custom_op_module

    input_gen_fns: dict[int, Callable[[Any], torch.Tensor]] = {}
    if user_input_gen_fns:
        input_gen_fns = custom_op_module._adapt_user_input_gen_fns(
            inputs, op_overload, user_input_gen_fns
        )

    template = _MultiOutputSubgraphTemplate(name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=decompositions,
        input_nodes=list(inputs),
        non_tensor_args=non_tensor_args,
        input_gen_fns=input_gen_fns or None,
        config_patches_list=config_patches_list,
    )
    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    _, winning_choice = custom_op_module.autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=list(inputs),
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
        is_collective=custom_op_module._detect_collective_ops(choices),
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
        return_multi_template=False,
    )
    if winning_choice is None or winning_choice.gm is None:
        raise AssertionError("TorchTLX subgraph autotuning must select a decomposition")

    operations_before = len(V.graph.operations)
    # Lower the selected FX graph under its own mode.  Applying patches only
    # after inlining is too late for lowerings that inspect config while they
    # create choices (notably generic TorchTLX GEMM epilogue fusion).
    with config.patch(winning_choice.config_patches or {}):
        result = inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)
    if winning_choice.config_patches:
        custom_op_module._apply_config_patches_recursive(
            V.graph.operations[operations_before:],
            winning_choice.config_patches,
        )
    return result


def register_tlx_subgraph_autotuning(
    custom_op: Any,
    *,
    name: str,
    tlx_configs: Sequence[Any],
    aten_impl: Callable[..., Any],
    input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    benchmark_with_cudagraphs: bool = False,
) -> None:
    """Register mode-aware autotuning for a semantic TorchTLX custom op.

    ``None`` lowers only through ``aten_impl``; ``allow`` benchmarks that
    implementation against all TLX candidates; and ``force`` admits only TLX
    candidates.  All choices are decomposed subgraphs, avoiding the opaque
    eager fallback and supporting tuple/list tensor outputs.
    """
    from torch._inductor.kernel import custom_op as custom_op_module
    from torch._inductor.kernel.custom_op import CustomOpConfig
    from torch._inductor.lowering import user_lowerings
    from torch._library.custom_ops import CustomOpDef

    if isinstance(custom_op, CustomOpDef):
        op_overload = custom_op._opoverload
    elif isinstance(custom_op, torch._ops.OpOverload):
        op_overload = custom_op
    else:
        raise TypeError(
            f"custom_op must be a CustomOpDef or OpOverload, got {type(custom_op)}"
        )

    candidates = list(tlx_configs)
    if not candidates:
        raise ValueError("at least one TLX subgraph config is required")
    if any(not isinstance(candidate, CustomOpConfig) for candidate in candidates):
        raise TypeError("tlx_configs must contain only CustomOpConfig objects")
    if any(candidate.decomposition is None for candidate in candidates):
        raise ValueError("every TLX subgraph config must provide a decomposition")

    # Keep the control candidate independent from the ambient ``allow`` mode.
    # Otherwise generic TorchTLX lowering may rewrite the ATen decomposition,
    # causing the autotuner to compare two TLX candidates instead of ATen vs TLX.
    aten_config = CustomOpConfig(
        aten_impl,
        config_patches={"triton.tlx_mode": None},
    )

    def make_lowering(configs: list[Any], lowering_name: str):
        @functools.wraps(op_overload)
        def lowering(*args: Any, **kwargs: Any) -> Any:
            tensor_inputs, runtime_kwargs = custom_op_module._extract_tensor_inputs(
                args, kwargs, op_overload
            )
            decompositions, non_tensor_args, config_patches_list = (
                custom_op_module._prepare_configs_and_decompositions(
                    configs,
                    None,
                    tensor_inputs,
                    op_overload,
                    op_overload,
                    runtime_kwargs,
                    lowering_name,
                )
            )
            result = _autotune_without_opaque_fallback(
                name=lowering_name,
                decompositions=decompositions,
                inputs=tensor_inputs,
                non_tensor_args=non_tensor_args,
                op_overload=op_overload,
                user_input_gen_fns=input_gen_fns,
                config_patches_list=config_patches_list,
                benchmark_with_cudagraphs=benchmark_with_cudagraphs,
            )
            custom_op_module.validate_ir(result)
            return result

        return lowering

    off_lowering = make_lowering([aten_config], f"{name}_off")
    allow_lowering = make_lowering([*candidates, aten_config], f"{name}_allow")
    force_lowering = make_lowering(candidates, name)

    @functools.wraps(op_overload)
    def mode_aware_lowering(*args: Any, **kwargs: Any) -> Any:
        if config.triton.tlx_mode == "force":
            return force_lowering(*args, **kwargs)
        if config.triton.tlx_mode == "allow":
            return allow_lowering(*args, **kwargs)
        return off_lowering(*args, **kwargs)

    user_lowerings[op_overload] = mode_aware_lowering
