"""LLM-based modulo scheduling for Triton.

When TRITON_USE_LLM_SCHEDULE=1 is set, this module is called during
make_ttgir() to produce schedule annotations (tt.autows, tt.num_stages)
using an LLM instead of the C++ modulo scheduling pass.

The LLM receives the TTGIR text and a system prompt describing the
scheduling rules, and produces a modulo.schedule graph. The graph is
then parsed and the annotations are set on the IR via mod.walk().
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def _get_prompt_file() -> Path:
    return Path(__file__).resolve().parents[1] / "tlx" / "tools" / "scheduling_prompt.md"


def _get_system_prompt() -> str:
    """Load the scheduling system prompt."""
    prompt_file = _get_prompt_file()
    if prompt_file.exists():
        return prompt_file.read_text()
    raise FileNotFoundError(f"Scheduling prompt not found at {prompt_file}")


def _call_llm(ttgir_text: str, system_prompt: str) -> str:
    """Call the LLM with the TTGIR input and return the schedule graph."""
    model = os.environ.get("TRITON_LLM_MODEL", "sonnet")

    user_prompt = ("Given the following TTGIR, produce the modulo.schedule graph "
                   "by following the steps in your instructions.\n\n"
                   "Output ONLY the raw modulo.schedule block. No explanation.\n\n"
                   f"{ttgir_text}")

    # Try Anthropic Python SDK first, fall back to claude CLI
    try:
        return _call_anthropic_sdk(user_prompt, system_prompt, model)
    except ImportError:
        pass

    return _call_claude_cli(user_prompt, system_prompt, model)


def _call_anthropic_sdk(user_prompt: str, system_prompt: str, model: str) -> str:
    """Call via the Anthropic Python SDK."""
    import anthropic

    model_map = {
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-6",
        "haiku": "claude-haiku-4-5-20251001",
    }
    model_id = model_map.get(model, model)

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def _call_claude_cli(user_prompt: str, system_prompt: str, model: str) -> str:
    """Fall back to calling the claude CLI."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as sp_file:
        sp_file.write(system_prompt)
        sp_path = sp_file.name

    try:
        cmd = [
            "claude",
            "--system-prompt",
            sp_path,
            "-p",
            user_prompt,
            "--output-format",
            "text",
            "--model",
            model,
            "--allowedTools",
            "",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout
    finally:
        os.unlink(sp_path)


def _parse_schedule(output: str) -> dict | None:
    """Parse the modulo.schedule block from LLM output.

    Returns a dict with schedule info, or None if parsing fails.
    """
    schedule_match = re.search(r"modulo\.schedule @\w+ \{(.+)\n\}", output, re.DOTALL)
    if not schedule_match:
        return None

    body = schedule_match.group(1)
    result = {"ii": None, "max_stage": None, "nodes": []}

    header = re.search(r"ii = (\d+), max_stage = (\d+)", body)
    if header:
        result["ii"] = int(header.group(1))
        result["max_stage"] = int(header.group(2))

    if result["ii"] is None:
        return None

    # Parse stage/order annotations for MMA ops
    for m in re.finditer(
            r"(ttng\.tc_gen5_mma|ttng\.tc_gen5_mma_scaled|ttng\.warp_group_dot|tt\.dot)"
            r"\s+\{[^}]*cycle: (\d+), cluster: (\d+)",
            body,
    ):
        op_name = m.group(1)
        cycle = int(m.group(2))
        cluster = int(m.group(3))
        stage = cycle // result["ii"]
        result["nodes"].append({
            "op": op_name,
            "stage": stage,
            "order": cluster,
        })

    # Parse num_stages from buffer counts
    buf_counts = []
    for m in re.finditer(r"%buf\d+ = modulo\.alloc SMEM \[(\d+) x", body):
        buf_counts.append(int(m.group(1)))
    if buf_counts:
        result["num_stages"] = max(buf_counts)
    elif result["max_stage"] is not None:
        result["num_stages"] = result["max_stage"] + 1

    return result


def _apply_schedule_to_ir(mod, schedule: dict) -> None:
    """Apply the parsed schedule to the MLIR module.

    Sets tt.autows on MMA ops and tt.num_stages on loops, matching
    the output format of the C++ modulo schedule pass.
    """
    from triton._C.libtriton import ir

    builder = ir.builder(mod.context)
    mma_idx = 0

    def visit(op):
        nonlocal mma_idx
        op_name = op.get_name()

        # Set tt.autows on MMA ops
        if op_name in (
                "ttng.tc_gen5_mma",
                "ttng.tc_gen5_mma_scaled",
                "ttng.warp_group_dot",
                "tt.dot",
        ):
            if mma_idx < len(schedule["nodes"]):
                node = schedule["nodes"][mma_idx]
                autows = json.dumps({
                    "stage": str(node["stage"]),
                    "order": str(node["order"]),
                })
                op.set_attr("tt.autows", builder.get_string_attr(autows))
                mma_idx += 1

        # Set tt.num_stages on scf.for loops
        if op_name == "scf.for" and schedule.get("num_stages"):
            op.set_attr(
                "tt.num_stages",
                builder.get_int32_attr(schedule["num_stages"]),
            )

    mod.walk(visit)


def run_llm_schedule(mod) -> bool:
    """Run LLM-based scheduling on the module.

    Returns True if scheduling was applied, False if it failed.
    """
    try:
        system_prompt = _get_system_prompt()
    except FileNotFoundError as e:
        logger.error("LLM Schedule: %s", e)
        return False

    ttgir_text = str(mod)

    logger.info("LLM Schedule: Calling LLM for schedule graph...")
    try:
        output = _call_llm(ttgir_text, system_prompt)
    except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError) as e:
        logger.error("LLM Schedule: LLM call failed: %s", e)
        return False

    schedule = _parse_schedule(output)
    if schedule is None:
        logger.warning("LLM Schedule: Failed to parse schedule from LLM output")
        if os.environ.get("TRITON_LLM_SCHEDULE_DEBUG"):
            logger.debug("LLM Schedule: Raw output:\n%s", output[:2000])
        return False

    logger.info(
        "LLM Schedule: Parsed: II=%d, max_stage=%s, num_stages=%s, MMA nodes=%d",
        schedule["ii"],
        schedule["max_stage"],
        schedule.get("num_stages"),
        len(schedule["nodes"]),
    )

    _apply_schedule_to_ir(mod, schedule)
    return True
