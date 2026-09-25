from __future__ import annotations

import dataclasses
import enum
from typing import Any, Optional, Sequence

#: 2: ``Stat`` is TFLOP/s rather than milliseconds, and the redundant
#: ``Result.tlx_tflops`` / ``ref_tflops`` are gone -- they were
#: ``flop_count / mean_latency``, which ``Stat.mean`` now is.
#: 3: multi-op. ``Case.direction`` (an op with a backward has two cases per
#: shape), ``Result.extra`` (op-specific derived metrics), and ``env["ref"]``
#: (which reference was raced -- not every op races torch).
SCHEMA_VERSION = 3


class Status(str, enum.Enum):

    OK = "ok"  # everything else
    PIP = "pip"  # perf improvement pending: too slow, or too slow to compile
    NOISY = "noisy"  # CV over the limit; no perf verdict is claimed
    ERROR = "error"  # the case raised, or returned the wrong answer


@dataclasses.dataclass(frozen=True)
class Case:

    op: str
    arch: str
    dtype: str  # bare torch name, e.g. "float16"
    shape: tuple  # op-defined and JSON-serializable; for mm, (M, N, K, a_rm, b_rm)
    #: Human-readable rendering of ``shape``, supplied by the op module because
    #: only it knows what the tuple means. ``(8192, 8192, 8192, True, False)``
    #: says nothing; "8192x8192x8192 A:row B:col" says what was measured. Falls
    #: back to the raw tuple so a new op need not provide one.
    label: str = ""
    #: "fwd" or "bwd". A real field rather than an entry in ``Result.extra``
    #: because it changes the FLOP count, changes which reference is fair, and
    #: has to reach ``key`` -- otherwise an op's forward and backward cases
    #: collide in the artifact and the second silently wins.
    direction: str = "fwd"

    @property
    def key(self) -> str:
        # Flattened, because a shape element may be a tuple (mm carries operand
        # strides) and str() on one puts parens and spaces in the artifact key.
        parts = ("_".join(str(x) for x in s) if isinstance(s, (tuple, list)) else str(s) for s in self.shape)
        key = f"{self.op}/{self.arch}/{self.dtype}/{'x'.join(parts)}"
        # Appended only when there is something to disambiguate, so every key an
        # op with no backward has ever written stays byte-identical and old
        # artifacts remain diffable against new ones.
        return key if self.direction == "fwd" else f"{key}/{self.direction}"

    @property
    def input(self) -> str:
        return self.label or "x".join(str(s) for s in self.shape)

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "arch": self.arch,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "direction": self.direction,
            "input": self.input,
            "key": self.key,
        }


@dataclasses.dataclass(frozen=True)
class Stat:
    # Percentiles are literal over TFLOP/s samples, so they ASCEND: p99 is the FAST
    # tail and the slow tail is `min`. The inverse of the latency reading.

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
    #: The config the launch actually ran, as a compact string. Without it a
    #: number cannot be reproduced or attributed: two runs of the same shape at
    #: `space="full"` may land on different tiles.
    best_config: Optional[str] = None
    #: Free-text, carried into the failure message and the artifact.
    notes: list = dataclasses.field(default_factory=list)
    #: Op-specific derived metrics, JSON-serializable. The dividing line against
    #: ``Case``: ``Case`` holds what the run was ASKED for (shape, dtype, causal,
    #: direction -- inputs you chose), ``extra`` holds what the run FOUND OUT
    #: (which SDPA backend torch dispatched, how many tokens a ragged batch
    #: actually carried). An op declares which of these earn a table column via
    #: its ``EXTRA_COLUMNS``; the rest ride along in the artifact.
    extra: dict = dataclasses.field(default_factory=dict)

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
            "best_config": self.best_config,
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }


def artifact(results: Sequence[Result], env: dict[str, Any]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "env": env,
        "results": [r.to_dict() for r in results],
    }
