"""End-to-end pipeline entrypoint for the new TLX Wave converter."""

from dataclasses import dataclass

from . import barrier_order
from . import canonicalize
from . import emission
from . import facts
from . import op_conversion
from . import source_import
from . import tokens
from . import types
from . import verifier


@dataclass(frozen=True)
class ConversionOutput:
    source_program: object
    type_layout_program: object
    fact_program: object
    token_program: object
    target_program: object
    emitted_module: emission.EmittedWaveModule


def convert_ttgir_to_wave(
    mod,
    *,
    kernel_name=None,
    verify=True,
    compiler_membar_barriers=(),
    enable_split_barriers=False,
    enable_multi_wave_specialization=False,
    waves_per_eu=0,
):
    source_program = source_import.import_source_program(
        mod,
        kernel_name=kernel_name,
        compiler_membar_barriers=compiler_membar_barriers,
    )
    type_layout_program = types.convert_source_program(source_program)
    fact_program = facts.analyze_facts(source_program, type_layout_program)
    token_program = tokens.build_token_program(source_program, type_layout_program)
    target_program = op_conversion.convert_ops(
        source_program,
        type_layout_program,
        fact_program,
        token_program,
    )
    target_program = canonicalize.canonicalize_target_program(target_program)
    target_program = canonicalize.eliminate_redundant_compiler_membar_barriers(
        target_program
    )
    target_program = barrier_order.thread_full_barrier_issue_order(
        target_program
    )
    if verify:
        verifier.verify_target_program(
            target_program,
            source_program=source_program,
            fact_program=fact_program,
            token_program=token_program,
        )
    target_program = canonicalize.eliminate_dead_target_ops(target_program)
    emitted_module = emission.emit_wave_module(
        target_program,
        fact_program,
        enable_split_barriers=enable_split_barriers,
        enable_multi_wave_specialization=enable_multi_wave_specialization,
        waves_per_eu=waves_per_eu,
    )
    return ConversionOutput(
        source_program,
        type_layout_program,
        fact_program,
        token_program,
        target_program,
        emitted_module,
    )
