"""Post-register-allocation AMDGCN hazard repair."""

import re

_SCHEDULED_MFMA_MARKER = "; triton_amd_scheduled_mfma"
_AMDGPU_REGISTER_RE = re.compile(r"\b([va])\[(\d+):(\d+)\]|\b([va])(\d+)\b")
_AMDGPU_REGISTER_OPERAND_RE = re.compile(r"([va])(?:\[(\d+):(\d+)\]|(\d+))")
_AMDGPU_UNLABELED_BLOCK_RE = re.compile(r"^\s*;\s*%bb\.\d+:")
_GFX950_SCHEDULED_MFMA_OPCODES = frozenset({
    "v_mfma_f32_16x16x32_bf16",
    "v_mfma_f32_16x16x32_f16",
    "v_mfma_f32_32x32x16_bf16",
    "v_mfma_f32_32x32x16_f16",
})
_GFX950_MFMA_RESULT_WAIT_STATES = {
    "v_mfma_f32_16x16x32_bf16": 12,
    "v_mfma_f32_16x16x32_f16": 12,
    "v_mfma_f32_32x32x16_bf16": 20,
    "v_mfma_f32_32x32x16_f16": 20,
}


def _amdgcn_instruction(line):
    code = line.partition(";")[0].strip()
    if not code or code.startswith(".") or code.endswith(":"):
        return None
    fields = code.split(None, 1)
    return fields[0], fields[1] if len(fields) == 2 else ""


def _amdgcn_registers(text):
    registers = set()
    for match in _AMDGPU_REGISTER_RE.finditer(text):
        if match.group(1):
            register_class = match.group(1)
            first = int(match.group(2))
            last = int(match.group(3))
            registers.update((register_class, index) for index in range(first, last + 1))
        else:
            registers.add((match.group(4), int(match.group(5))))
    return registers


def _amdgcn_register_operand(text):
    """Parse one complete VGPR/AGPR operand, rejecting unfamiliar syntax."""
    match = _AMDGPU_REGISTER_OPERAND_RE.fullmatch(text.strip())
    if match is None:
        return None
    register_class = match.group(1)
    if match.group(4) is not None:
        return {(register_class, int(match.group(4)))}
    first = int(match.group(2))
    last = int(match.group(3))
    if last < first:
        return None
    return {(register_class, index) for index in range(first, last + 1)}


def _amdgcn_wait_states(instruction):
    opcode, operands = instruction
    if opcode == "s_nop":
        # gfx950 only decodes SIMM16[3:0], so larger immediates wrap instead
        # of waiting for more than 16 states.
        return (int(operands.split()[0], 0) & 0xF) + 1
    return 1


def _wait_states_since(history, limit, is_hazard):
    wait_states = 0
    for instruction in reversed(history):
        if is_hazard(instruction):
            return wait_states
        wait_states += _amdgcn_wait_states(instruction)
        if wait_states >= limit:
            return limit
    # History is reset at each basic-block boundary, so reaching the beginning
    # means no predecessor distance is assumed.
    return wait_states


def _is_legacy_valu_not_dot(instruction):
    opcode, _ = instruction
    return opcode.startswith("v_") and not opcode.startswith(("v_mfma", "v_smfmac", "v_wmma", "v_dot"))


def _legacy_valu_may_write(instruction, source_registers):
    if not _is_legacy_valu_not_dot(instruction):
        return False
    opcode, operands = instruction
    operand_fields = operands.split(",")
    # These swap instructions define both operands. LLVM may introduce them
    # while shrinking after register allocation, so treating only operand 0 as
    # a destination can miss a source-register hazard on operand 1.
    destination_count = (2 if opcode.startswith((
        "v_swap_b32",
        "v_permlane16_swap_b32",
        "v_permlane32_swap_b32",
    )) else 1)
    destination_registers = _amdgcn_registers(",".join(operand_fields[:destination_count]))
    # Be conservative for unfamiliar VALU syntax in the short hazard window.
    return not destination_registers or bool(destination_registers & source_registers)


def _valu_writes_exec(instruction):
    opcode, _ = instruction
    return opcode.startswith("v_") and "cmpx" in opcode


def _accvgpr_write_may_write(instruction, registers):
    opcode, operands = instruction
    if not opcode.startswith(("v_accvgpr_write", "v_accvgpr_mov")):
        return False
    destination, _, _ = operands.partition(",")
    destination_registers = _amdgcn_registers(destination)
    return not destination_registers or bool(destination_registers & registers)


def _advance_outstanding_mfma_results(outstanding, wait_states):
    """Advance and retire marked MFMA result-read hazards in place."""
    for register, remaining in tuple(outstanding.items()):
        remaining -= wait_states
        if remaining <= 0:
            del outstanding[register]
        else:
            outstanding[register] = remaining


def _is_amdgcn_basic_block_label(line):
    # LLVM prints an untargeted fallthrough MBB as only `; %bb.N:`. It still
    # starts a distinct CFG node even though no branch can name it.
    if _AMDGPU_UNLABELED_BLOCK_RE.match(line):
        return True
    label = line.partition(";")[0].strip()
    if not label.endswith(":"):
        return False
    return label.startswith(".LBB") or not label.startswith(".")


def _amdgcn_basic_block_label(line):
    if _AMDGPU_UNLABELED_BLOCK_RE.match(line):
        return None
    if not _is_amdgcn_basic_block_label(line):
        return None
    return line.partition(";")[0].strip()[:-1]


def _amdgcn_basic_blocks(amdgcn):
    blocks = []
    current = []
    for line in amdgcn.splitlines():
        if _is_amdgcn_basic_block_label(line) and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _amdgcn_block_successors(blocks):
    labels = {}
    for index, block in enumerate(blocks):
        for line in block:
            if (label := _amdgcn_basic_block_label(line)) is not None:
                labels[label] = index
                break

    successors = []
    for index, block in enumerate(blocks):
        block_successors = []
        terminates = False
        for line_index, line in enumerate(block):
            instruction = _amdgcn_instruction(line)
            if instruction is None:
                continue
            opcode, operands = instruction
            is_conditional_branch = opcode.startswith("s_cbranch")
            is_unconditional_branch = opcode == "s_branch"
            if is_conditional_branch or is_unconditional_branch:
                for token in re.findall(r"[.$A-Za-z_][\w.$]*", operands):
                    if token in labels:
                        block_successors.append((labels[token], line_index))
                terminates |= is_unconditional_branch
            elif opcode.startswith(("s_endpgm", "s_setpc", "s_swappc")):
                terminates = True
        if not terminates and index + 1 < len(blocks):
            block_successors.append((index + 1, None))
        successors.append(block_successors)
    return successors


def _append_s_nops(output, history, indentation, wait_states):
    while wait_states:
        encoded_wait_states = min(wait_states, 16)
        instruction = ("s_nop", str(encoded_wait_states - 1))
        output.append(f"{indentation}{instruction[0]} {instruction[1]}")
        history.append(instruction)
        wait_states -= encoded_wait_states


def _repair_scheduled_mfma_block(lines, incoming_results):
    output = []
    history = []
    outstanding_results = dict(incoming_results)
    results_after_lines = []
    pending_marker = False
    for line in lines:
        if _SCHEDULED_MFMA_MARKER in line:
            if pending_marker:
                raise ValueError("consecutive scheduled MFMA markers in AMDGCN assembly")
            pending_marker = True
            output.append(line)
            results_after_lines.append(dict(outstanding_results))
            continue

        instruction = _amdgcn_instruction(line)
        marked_destination = None
        source_registers = None
        accumulator_registers = None
        if pending_marker and instruction is not None:
            opcode, operands = instruction
            if opcode not in _GFX950_SCHEDULED_MFMA_OPCODES:
                raise ValueError("scheduled MFMA marker is followed by an unsupported "
                                 f"gfx950 instruction: {opcode}")
            mfma_operands = [operand.strip() for operand in operands.split(",")]
            if len(mfma_operands) != 4:
                raise ValueError("cannot parse marked MFMA operands")
            marked_destination = _amdgcn_register_operand(mfma_operands[0])
            source_a_registers = _amdgcn_register_operand(mfma_operands[1])
            source_b_registers = _amdgcn_register_operand(mfma_operands[2])
            if not marked_destination or not source_a_registers or not source_b_registers:
                raise ValueError("cannot parse marked MFMA source registers")
            destination_classes = {kind for kind, _ in marked_destination}
            if len(destination_classes) != 1:
                raise ValueError("marked MFMA destination must use one register class")
            source_registers = source_a_registers | source_b_registers
            if mfma_operands[3] != "0":
                accumulator_registers = _amdgcn_register_operand(mfma_operands[3])
                if not accumulator_registers:
                    raise ValueError("cannot parse marked MFMA accumulator registers")
                if accumulator_registers != marked_destination:
                    raise ValueError("marked MFMA srcC must be tied to its destination")

        if instruction is not None and outstanding_results:
            if marked_destination is None:
                # Both a read and an overwrite of an outstanding result are
                # hazardous, regardless of its physical register class.
                referenced_registers = _amdgcn_registers(instruction[1])
            else:
                referenced_registers = set(source_registers)
                if accumulator_registers is None:
                    referenced_registers.update(marked_destination)
            early_references = referenced_registers.intersection(outstanding_results)
            if early_references:
                residual_wait_states = max(outstanding_results[register] for register in early_references)
                indentation = line[:len(line) - len(line.lstrip())]
                _append_s_nops(
                    output,
                    history,
                    indentation,
                    residual_wait_states,
                )
                _advance_outstanding_mfma_results(outstanding_results, residual_wait_states)

        if marked_destination is not None:
            source_wait_states = _wait_states_since(
                history,
                2,
                lambda previous: _legacy_valu_may_write(previous, source_registers),
            )
            source_agpr_wait_states = _wait_states_since(
                history,
                3,
                lambda previous: _accvgpr_write_may_write(previous, source_registers),
            )
            accumulator_wait_states = (_wait_states_since(
                history,
                1,
                lambda previous: (_accvgpr_write_may_write(previous, accumulator_registers)
                                  or _legacy_valu_may_write(previous, accumulator_registers)),
            ) if accumulator_registers else 1)
            exec_wait_states = _wait_states_since(history, 4, _valu_writes_exec)
            residual_wait_states = max(
                2 - source_wait_states,
                3 - source_agpr_wait_states,
                1 - accumulator_wait_states,
                4 - exec_wait_states,
                0,
            )
            if residual_wait_states:
                indentation = line[:len(line) - len(line.lstrip())]
                _append_s_nops(
                    output,
                    history,
                    indentation,
                    residual_wait_states,
                )
                _advance_outstanding_mfma_results(outstanding_results, residual_wait_states)
            pending_marker = False

        output.append(line)
        if instruction is not None:
            _advance_outstanding_mfma_results(outstanding_results, _amdgcn_wait_states(instruction))
            if marked_destination is not None:
                result_wait_states = _GFX950_MFMA_RESULT_WAIT_STATES[instruction[0]]
                for register in marked_destination:
                    outstanding_results[register] = result_wait_states
            history.append(instruction)
        results_after_lines.append(dict(outstanding_results))

    if pending_marker:
        raise ValueError("scheduled MFMA marker is not followed by an instruction")
    return output, outstanding_results, results_after_lines


def insert_scheduled_mfma_hazard_nops(amdgcn, arch):
    """Repair marked gfx950 MFMA hazards after scheduling and register allocation."""
    if _SCHEDULED_MFMA_MARKER not in amdgcn:
        return amdgcn
    if arch != "gfx950":
        raise ValueError(f"scheduled MFMA post-RA hazard repair is unsupported on {arch}")

    blocks = _amdgcn_basic_blocks(amdgcn)
    successors = _amdgcn_block_successors(blocks)
    incoming_results = [{} for _ in blocks]
    worklist = list(range(len(blocks)))
    queued = set(worklist)
    while worklist:
        block_index = worklist.pop()
        queued.remove(block_index)
        _, outgoing_results, results_after_lines = _repair_scheduled_mfma_block(blocks[block_index],
                                                                                incoming_results[block_index])
        for successor, branch_line_index in successors[block_index]:
            edge_results = (outgoing_results if branch_line_index is None else results_after_lines[branch_line_index])
            changed = False
            successor_results = incoming_results[successor]
            for register, wait_states in edge_results.items():
                if wait_states > successor_results.get(register, 0):
                    successor_results[register] = wait_states
                    changed = True
            if changed and successor not in queued:
                worklist.append(successor)
                queued.add(successor)

    output = []
    for block, block_results in zip(blocks, incoming_results):
        repaired, _, _ = _repair_scheduled_mfma_block(block, block_results)
        output.extend(repaired)
    return "\n".join(output) + ("\n" if amdgcn.endswith("\n") else "")
