"""The vocabulary the harness produces and the report consumes.

Kept in its own module so ``measure`` / ``compile`` / ``verdict`` / ``report``
share a vocabulary without depending on each other.

The JSON that ``report`` emits is a stable, versioned artifact: it is what CI
uploads and what a review agent reads *instead of parsing stdout*. Treat it as
an interface -- bump ``SCHEMA_VERSION`` on any incompatible change.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Optional, Sequence

#: 2: ``Stat`` is TFLOP/s rather than milliseconds, and the redundant
#: ``Result.tlx_tflops`` / ``ref_tflops`` are gone -- they were
#: ``flop_count / mean_latency``, which ``Stat.mean`` now is.
SCHEMA_VERSION = 2


class Status(str, enum.Enum):
    """Per-case outcome. The four the README defines, and no more.

    Every threshold behind these is absolute, so a verdict depends only on the
    run that produced it -- there is no recorded baseline to compare against.
    That is what lets the suite be a single stateless command.
    """

    OK = "ok"  # everything else
    PIP = "pip"  # perf improvement pending: too slow, or too slow to compile
    NOISY = "noisy"  # CV over the limit; no perf verdict is claimed
    ERROR = "error"  # the case raised, or returned the wrong answer


@dataclasses.dataclass(frozen=True)
class Case:
    """One benchmarked point. ``key`` identifies it in the JSON artifact, so it
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
        # Flattened, because a shape element may be a tuple (mm carries operand
        # strides) and str() on one puts parens and spaces in the artifact key.
        parts = ("_".join(str(x) for x in s) if isinstance(s, (tuple, list)) else str(s) for s in self.shape)
        return f"{self.op}/{self.arch}/{self.dtype}/{'x'.join(parts)}"

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
    """A summarized measurement. ``unit`` says of what; ``tflops`` by default.

    Every value-typed field below is in that unit, and the conversion happened
    per timed iteration, upstream in ``measure`` -- so ``cv`` and the
    percentiles describe the distribution of the reported quantity rather than
    the distribution of something else that was divided into it afterwards.

    **Percentiles are literal.** Over TFLOP/s samples they ascend: ``p99`` is
    the fast tail, beaten by only 1% of iterations, and the slow tail is
    ``min``. This is the inverse of the latency reading, where p99 is the bad
    case, and it is the one thing about this dataclass that is easy to get
    backwards.

    Three dispersion figures at two different scales, and the distinction is
    worth keeping straight. ``cv`` and ``rel_idr`` are WITHIN one run;
    ``rel_max_deviation`` is BETWEEN runs. With a few thousand samples per run
    the median is far more stable than the distribution around it is wide:
    measured on B200, mm 8192^3 has a within-run interdecile range of ~6% --
    the power-governed clock wandering -- while its p50 reproduces between runs
    to 1.7%.

    **The README makes ``cv`` the gate**, so a case is judged on whether the
    machine held still while measuring. Note that reads the wider of the two
    scales: the between-run figures on that same B200 run were 0.0-0.2%, so
    ``rel_max_deviation`` stays computed and stays in the artifact as the
    diagnostic for "did the number reproduce", which is a different question.
    """

    #: The headline value. Reported rather than the median because a mean plus
    #: a coefficient of variation is the conventional way to summarize a
    #: distribution, and because the tail matters for a kernel: a median hides
    #: a slow iteration completely.
    mean: float
    #: Coefficient of variation of the pooled samples, ``sd / mean``. The
    #: headline dispersion. Computed after IQR rejection, so it describes the
    #: distribution rather than the worst descheduled iteration.
    cv: float
    #: Nearest-rank percentiles of the pooled samples, so each is an observed
    #: iteration. In TFLOP/s these ASCEND -- see the class docstring.
    p50: float
    p95: float
    p99: float
    #: In TFLOP/s, ``min`` is the slowest iteration and ``max`` the fastest.
    min: float
    max: float
    #: Relative maximum deviation of the replicate means from their median:
    #: ``max|mean_i - median(mean)| / median(mean)``. BETWEEN runs -- it is the
    #: uncertainty on the headline number, which CV (a within-run figure) is
    #: not. Dimensionless, so unaffected by the unit.
    rel_max_deviation: Optional[float]
    #: Relative interdecile range of the pooled samples, ``(p90 - p10) / p50``.
    #: Robust companion to ``cv``; kept in the artifact, not in the table.
    #: Dimensionless.
    rel_idr: float
    replicates: int
    n_kept: int
    n_raw: int
    #: What the value-typed fields are in. ``tflops`` for anything the report
    #: prints; ``ms`` when ``measure`` was called without a ``flop_count``.
    #: Recorded rather than assumed so a consumer of the JSON never has to
    #: infer the unit from the magnitudes.
    unit: str = "tflops"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Result:
    """Everything measured for one case, plus the verdict."""

    case: Case
    status: Status = Status.OK
    #: Both in TFLOP/s. There is no separate ``tlx_tflops`` field: it used to
    #: be ``flop_count / tlx.mean_latency``, which is exactly what ``tlx.mean``
    #: is now, and two spellings of one number is two things to keep in sync.
    tlx: Optional[Stat] = None
    ref: Optional[Stat] = None
    #: tlx.mean / ref.mean -- the enforcing metric, because a ratio measured on
    #: one machine in one run is immune to clock and thermal drift. Above 1
    #: means TLX is faster, in throughput as it did in latency.
    speedup: Optional[float] = None
    #: Useful FLOPs for one call, supplied by the op module. Kept even though
    #: the throughputs are already derived from it, because it is what lets a
    #: consumer recover a latency, and what says which FLOPs were counted.
    flop_count: Optional[int] = None
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
            "flop_count": self.flop_count,
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
