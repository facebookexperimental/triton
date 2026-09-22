from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Sequence

MAX_PAIRED_SPEEDUP_SPREAD = 0.05


@dataclasses.dataclass(frozen=True)
class PairedSample:
    before_us: float
    after_us: float
    order: str

    @property
    def speedup(self) -> float:
        return self.before_us / self.after_us


@dataclasses.dataclass(frozen=True)
class PairedSummary:
    before_us: float
    after_us: float
    speedup: float
    speedup_spread: float
    noisy: bool
    samples: tuple[PairedSample, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "before_us": self.before_us,
            "after_us": self.after_us,
            "speedup": self.speedup,
            "speedup_spread": self.speedup_spread,
            "status": "noisy" if self.noisy else "stable",
            "samples": [dataclasses.asdict(sample) | {"speedup": sample.speedup} for sample in self.samples],
        }


def relative_interdecile_range(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two values are required")
    median = statistics.median(values)
    if not median:
        return float("inf")
    deciles = statistics.quantiles(values, n=10, method="inclusive")
    return (deciles[8] - deciles[0]) / median


def summarize_pairs(
    samples: Sequence[PairedSample],
    *,
    max_spread: float = MAX_PAIRED_SPEEDUP_SPREAD,
) -> PairedSummary:
    if len(samples) < 2:
        raise ValueError("at least two paired samples are required")
    before = [sample.before_us for sample in samples]
    after = [sample.after_us for sample in samples]
    speedups = [sample.speedup for sample in samples]
    spread = relative_interdecile_range(speedups)
    return PairedSummary(
        before_us=statistics.median(before),
        after_us=statistics.median(after),
        speedup=statistics.median(speedups),
        speedup_spread=spread,
        noisy=spread > max_spread,
        samples=tuple(samples),
    )
