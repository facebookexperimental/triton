from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol

from .models import KernelOptimizationRequest, PerformanceSummary
from .source import extract_python_source

TLX_PROMPT_PREAMBLE = """You are optimizing a Triton/TLX GPU kernel.

Grouped MXFP8 target: C_g = A_g @ B_g.T for g in [0, G), A_g is E4M3 [M_g, K]
and B_g is E4M3 [N, K], one E8M0 scale per 32 K values. Scales are swizzled
to 5-D [1, rows//128, K//128, 2, 256] for tlx.async_dot_scaled.

References you should follow (prefer grouped_gemm.py:752 for scheduling,
blackwell_gemm_ws_mxfp8.py / fburl.com/code/wdyi24fn for MX):
- Grouped scheduling (genai/msl/ops/kernels/tlx/gemm/grouped_gemm.py:752
  _tlx_grouped_gemm_blackwell): 3 async_tasks (producer TMA / MMA async_dot
  / epilogue TMEM), split_sizes per-group M (_get_group_sizes), flat vs
  persistent dispatch via counter_ptr atomicAdd + tile_id_smem DSMEM +
  remote_shmem_store + fence.proxy.async.shared::cluster, GROUP_SIZE_M
  swizzle via _compute_pid, 1CTA/2CTA (cluster_cta_rank, cta_bars,
  remote_cta_rank), NUM_SMEM_BUFFERS / NUM_TMEM_BUFFERS / EPILOGUE_SUBTILE
  pipeline, _get_tile_grid / _producer_fetch_tile_idx_2cta, tmem_empty/full
  mbarriers.
- MX block-scaled MMA (third_party/tlx/tutorials/blackwell_gemm_ws_mxfp8.py  # fburl.com/code/wdyi24fn):
  BLOCK_SIZE_M==128 required for scaled MMA, tlx.async_dot_scaled on E4M3+E8M0
  with tlx.dtype_of(desc), swizzled scales [1, REP_M, REP_K, 2, 256] via
  TensorDescriptor + async_descriptor_load + barrier_expect_bytes, REP_K=
  BLOCK_K//32//4, tlx.local_trans on B, tmem empty/full + smem empty/full
  phases via get_bufidx_phase, tlx.tcgen05_commit.

Optimization axes to tune (pick 1-2 per candidate, keep harness verifiable):
- GROUP_SIZE_M in {1, 4, 8, 64} — swizzle order of M vs N tiles.
- NUM_SMEM_BUFFERS in {3, 4, 6} and NUM_TMEM_BUFFERS in {1, 2} — pipeline depth.
- NUM_CTAS in {1, 2} — only use 2 with a persistent counter + DSMEM tile_idx
  path and ctas_per_cga=(2,1,1); remember 2-CTA needs full B-scale reload.
- BLOCK_SIZE_K 128 vs 256 — must stay %128; larger reduces K tiles but grows SMEM.
Promote from the flat-grid tl.dot baseline to TMA + TMEM + tlx.async_dot_scaled
when cases have K>=256 (form is block-scaled MXFP8, scale tiling assumes K%128==0).

TLX essentials:
- `import triton.language as tl` and `import triton.language.extra.tlx as tlx`
- Warp spec: `with tlx.async_tasks(exclusive=True, no_ending_cluster_sync=True, mbarrier_try_wait_suspend_ns=50000): with tlx.async_task("default"): ... with tlx.async_task(num_warps=1, num_regs=24): ...`
- Memory: `tlx.local_alloc(shape, dtype, num_buffers, tlx.storage_kind.tmem)` + `tlx.alloc_barriers`, `tlx.async_descriptor_load`, `tlx.barrier_wait/arrive/expect_bytes`, `tlx.local_slice/load/store`, `tlx.barrier_arrive` + `tcgen05_commit`
- Persistent grouped GEMMs: `tlx.local_alloc((1,), tl.int32, NUM_TILE_BUFFERS)` for tile_id_smem, `tlx.remote_shmem_store` / `local_load` for 2-CTA, `tl.atomic_add(counter_ptr, 1 or 2)` for dispatch.
- Follow https://www.internalfb.com/wiki/Triton_Core/Agentic_Reference/ and third_party/tlx/doc/tlx_barriers.md.
"""


@dataclass(frozen=True)
class CandidateProposal:
    source: str
    summary: str = ""


@dataclass(frozen=True)
class CandidateContext:
    round_index: int
    candidate_index: int
    current_source: str
    current_performance: PerformanceSummary
    previous_diagnostics: tuple[str, ...]


class CandidateProvider(Protocol):
    def propose(
        self,
        request: KernelOptimizationRequest,
        context: CandidateContext,
    ) -> CandidateProposal: ...


@dataclass
class FixedCandidateProvider:
    candidates: list[CandidateProposal]

    def propose(
        self,
        request: KernelOptimizationRequest,
        context: CandidateContext,
    ) -> CandidateProposal:
        del request, context
        if not self.candidates:
            raise RuntimeError("fixed candidate provider is exhausted")
        return self.candidates.pop(0)


@dataclass(frozen=True)
class MockLLMProvider:
    """Deterministic stub for CI — replays canned candidates without a live LLM."""

    canned: tuple[CandidateProposal, ...] = ()
    fallback_source: str | None = None

    def propose(
        self,
        request: KernelOptimizationRequest,
        context: CandidateContext,
    ) -> CandidateProposal:
        del request
        index = (context.round_index - 1) * 10 + context.candidate_index
        if index < len(self.canned):
            return self.canned[index]
        if self.fallback_source is not None:
            return CandidateProposal(source=self.fallback_source, summary="mock-fallback")
        # Default: echo current source so the harness re-evaluates it (dedup will
        # turn the second echo into a deterministic failure rather than a hang).
        return CandidateProposal(source=context.current_source, summary="mock-echo")


@dataclass(frozen=True)
class CodexCandidateProvider:
    model: str = "sonnet"
    timeout_seconds: float = 300.0

    def propose(
        self,
        request: KernelOptimizationRequest,
        context: CandidateContext,
    ) -> CandidateProposal:
        prompt = _build_prompt(request, context)
        try:
            completed = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--skip-git-repo-check",
                    "-m",
                    self.model,
                    "--sandbox",
                    "workspace-write",
                    prompt,
                ],
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "codex binary not found; install it or use --provider mock"
            ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                f"candidate generator exited with code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        source = extract_python_source(completed.stdout)
        if not source.strip():
            raise RuntimeError("candidate generator returned empty source")
        return CandidateProposal(source=source, summary="Codex-generated candidate")


def _build_prompt(
    request: KernelOptimizationRequest,
    context: CandidateContext,
) -> str:
    case_lines = "\n".join(
        f"- {case.case_id}: parameters={dict(case.parameters)}, weight={case.weight}"
        for case in request.cases
    )
    performance_lines = "\n".join(
        f"- {case.case_id}: median_us="
        f"{case.timing.median_us if case.timing else 'unavailable'}, "
        f"p95_us={case.timing.p95_us if case.timing else 'unavailable'}, "
        f"cv={case.timing.coefficient_of_variation if case.timing else 'unavailable'}, "
        f"profile={dict(case.profile)}"
        for case in context.current_performance.cases
    )
    diagnostics = "\n".join(context.previous_diagnostics[-5:]) or "None"
    reference_block = ""
    if getattr(request, "reference_kernel_source", None):
        reference_block = f"\nReference kernel (oracle, do not copy verbatim — use for correctness/performance comparison):\n```python\n{request.reference_kernel_source[:4000]}\n```\n"
    return f"""{TLX_PROMPT_PREAMBLE}{reference_block}
You are proposing one candidate for the closed loop `build -> verify -> benchmark -> profile -> propose -> repeat`.
Return exactly one complete replacement source file. Do not return a diff and do not
claim correctness or performance; an external deterministic harness decides both.
Preserve the public entry points expected by the harness. Make one coherent optimization
that can be diagnosed if it fails.

Target: backend={request.target.backend}, architecture={request.target.architecture}
Round: {context.round_index}, candidate: {context.candidate_index}
Cases:
{case_lines}

Current measurements:
{performance_lines}

Recent failed-candidate diagnostics:
{diagnostics}

Current source:
```python
{context.current_source}
```
"""
