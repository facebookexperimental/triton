"""The vocabulary the harness produces and the baseline consumes.

Kept in its own module so ``measure`` / ``compile`` / ``baseline`` / ``report``
share a vocabulary without depending on each other.

The JSON that ``report`` emits is a stable, versioned artifact: it is what CI
uploads and what a review agent reads *instead of parsing stdout*. Treat it as
an interface -- bump ``SCHEMA_VERSION`` on any incompatible change.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Optional, Sequence

SCHEMA_VERSION = 1


class Status(str, enum.Enum):
    """Per-case outcome. Only ``PASS`` and ``NOISY`` are reachable before the
    guard lands in phase 5; the rest are declared here so the schema is stable
    from the first artifact."""

    PASS = "pass"
    REGRESSED = "regressed"  # slower than baseline by more than the tolerance
    SLOW_COMPILE = "slow_compile"  # t_cold over the absolute cap
    NOISY = "noisy"  # spread over the noise floor; no perf verdict is claimed
    # Latency is dominated by host-side launch cost, so the number describes
    # Python rather than the kernel. Reported, never gated. Distinct from NOISY
    # because the cause and the fix are different: NOISY is the machine,
    # HOST_BOUND is the op's per-call overhead against a too-small shape.
    HOST_BOUND = "host_bound"
    ERROR = "error"  # the case raised


@dataclasses.dataclass(frozen=True)
class Case:
    """One benchmarked point. ``key`` is also the baseline lookup key, so it
    must be stable across runs and readable in a diff."""

    op: str
    arch: str
    dtype: str  # bare torch name, e.g. "float16"
    shape: tuple  # op-defined and JSON-serializable; for mm, (M, N, K, a_rm, b_rm)
    #: Human-readable rendering of ``shape``, supplied by the op module because
    #: only it knows what the tuple means. ``(8192, 8192, 8192, True, False)``
    #: says nothing; "8192x8192x8192 A:row B:col" says what was measured. Falls
    #: back to the raw tuple so a new op need not provide one.
    label: str = ""

    @property
    def key(self) -> str:
        return f"{self.op}/{self.arch}/{self.dtype}/{'x'.join(str(s) for s in self.shape)}"

    @property
    def input(self) -> str:
        return self.label or "x".join(str(s) for s in self.shape)

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "arch": self.arch,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "input": self.input,
            "key": self.key,
        }


@dataclasses.dataclass(frozen=True)
class Stat:
    """A summarized latency measurement, in milliseconds.

    Two dispersion figures, at two different scales, and the distinction is
    load-bearing. ``rel_max_deviation`` is BETWEEN runs and ``rel_idr`` is
    WITHIN one; with a few thousand samples per run the median is far more
    stable than the distribution around it is wide. Measured on B200, mm 8192^3
    has a within-run relative interdecile range of ~6% -- the power-governed
    clock wanders -- while its p50 reproduces between runs to 1.7%. The gate
    reads the first; gating on the second rejected cases whose reported number
    was in fact solid.
    """

    #: The headline value. Reported rather than the median because a mean plus
    #: a coefficient of variation is the conventional way to summarize a
    #: latency distribution, and because the tail matters for a kernel: a
    #: median hides a slow p99 completely.
    mean: float
    #: Coefficient of variation of the pooled samples, ``sd / mean``. The
    #: headline dispersion. Computed after IQR rejection, so it describes the
    #: distribution rather than the worst descheduled iteration.
    cv: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    #: Relative maximum deviation of the replicate means from their median:
    #: ``max|mean_i - median(mean)| / median(mean)``. BETWEEN runs, and what
    #: the gate reads -- it is the uncertainty on the headline number, which
    #: CV (a within-run figure) is not.
    rel_max_deviation: float
    #: Relative interdecile range of the pooled samples, ``(p90 - p10) / p50``.
    #: Robust companion to ``cv``; kept in the artifact, not in the table.
    rel_idr: float
    replicates: int
    n_kept: int
    n_raw: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Result:
    """Everything measured for one case, plus the verdict."""

    case: Case
    status: Status = Status.PASS
    tlx: Optional[Stat] = None
    ref: Optional[Stat] = None
    #: ref.p50 / tlx.p50 -- the enforcing metric, because a ratio measured on
    #: one machine in one run is immune to clock and thermal drift.
    speedup: Optional[float] = None
    tlx_tflops: Optional[float] = None
    ref_tflops: Optional[float] = None
    #: Host-side per-call cost, microseconds. Carried in the artifact because
    #: it is what distinguishes "the kernel got slower" from "the launch path
    #: got slower", and the two have nothing to do with each other.
    tlx_host_us: Optional[float] = None
    ref_host_us: Optional[float] = None
    #: Populated in phase 3.
    t_cold_s: Optional[float] = None
    t_compile_single_s: Optional[float] = None
    n_configs: Optional[int] = None
    #: Free-text, carried into the failure message and the artifact.
    notes: list = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case": self.case.to_dict(),
            "status": self.status.value,
            "tlx": self.tlx.to_dict() if self.tlx else None,
            "ref": self.ref.to_dict() if self.ref else None,
            "speedup": self.speedup,
            "tlx_tflops": self.tlx_tflops,
            "ref_tflops": self.ref_tflops,
            "tlx_host_us": self.tlx_host_us,
            "ref_host_us": self.ref_host_us,
            "t_cold_s": self.t_cold_s,
            "t_compile_single_s": self.t_compile_single_s,
            "n_configs": self.n_configs,
            "notes": list(self.notes),
        }


def artifact(results: Sequence[Result], env: dict[str, Any]) -> dict:
    """The top-level JSON document. ``env`` records what the numbers depend on
    -- GPU, driver, clock-lock state -- because a latency without that context
    is not comparable to anything."""
    return {
        "schema_version": SCHEMA_VERSION,
        "env": env,
        "results": [r.to_dict() for r in results],
    }
