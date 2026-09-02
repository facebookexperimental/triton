"""Pin a candidate's Triton autotuner to one config, without touching the kernel.

Applied around every ``build``/``verify``/``benchmark``/``profile`` call in the
gfx942 harness, and restored afterwards.

`tlx.ops.mm` at `space="full"` searches 42 configs per (M, N, K). Left alone that makes the
promote/reject gate measure *tile selection* rather than the structural change
the candidate actually made -- two candidates can differ only in which tile the
autotuner happened to pick -- and it makes every evaluation pay the search.

Pinning is done by filtering the `Autotuner` object's own `configs` list, found
by scanning module globals for the type. Doing it that way rather than through a
kernel-side env hook means it keeps working when the agent renames functions,
restructures `_configs()`, or drops a hook it was told to preserve -- none of
which are things a candidate generator can be relied on not to do.

Both the in-process harness and the rocprofv3 child use this, so the traced
dispatch is the same one the gate timed.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any


def _autotuners(module: ModuleType) -> list[Any]:
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:  # pragma: no cover - triton always present in practice
        return []
    return [value for value in vars(module).values() if isinstance(value, Autotuner)]


def _matches(config: Any, pin: dict[str, Any]) -> bool:
    """True when ``config`` agrees with every key in ``pin``.

    `num_warps` / `num_stages` live on the Config object; everything else lives
    in its `kwargs`, so both namespaces are checked.
    """
    for key, wanted in pin.items():
        if key in ("num_warps", "num_stages", "num_ctas"):
            if getattr(config, key, None) != wanted:
                return False
        elif config.kwargs.get(key) != wanted:
            return False
    return True


def pin(module: ModuleType, pin_config: dict[str, Any] | None) -> dict[str, Any]:
    """Restrict every autotuner in ``module`` to configs matching ``pin_config``.

    Returns a report -- never raises. A candidate that cannot be pinned is still
    a candidate; the caller decides what to do about a gate measured unpinned,
    and `pinned` in the report is how it finds out.
    """
    tuners = _autotuners(module)
    report: dict[str, Any] = {
        "autotuners_found": len(tuners),
        "requested": dict(pin_config or {}),
        "pinned": False,
    }
    if not pin_config:
        report["configs_before"] = [len(t.configs) for t in tuners]
        return report
    if not tuners:
        report["note"] = "no triton Autotuner in candidate module; nothing to pin"
        return report

    before: list[int] = []
    after: list[int] = []
    for tuner in tuners:
        before.append(len(tuner.configs))
        kept = [config for config in tuner.configs if _matches(config, pin_config)]
        if kept:
            tuner.configs = kept
            # A stale entry would replay the pre-pin winner and silently defeat
            # the pin, so drop whatever the tuner already decided.
            getattr(tuner, "cache", {}).clear()
        after.append(len(tuner.configs))
    report["configs_before"] = before
    report["configs_after"] = after
    report["pinned"] = all(count == 1 for count in after)
    if not report["pinned"]:
        report["note"] = ("pin_config did not reduce every autotuner to a single config; "
                          "the gate measurement includes an autotune search")
    return report


def restore(module: ModuleType, saved: list[list[Any]]) -> None:
    """Put back the config lists captured by :func:`snapshot`."""
    for tuner, configs in zip(_autotuners(module), saved):
        tuner.configs = configs
        getattr(tuner, "cache", {}).clear()


def snapshot(module: ModuleType) -> list[list[Any]]:
    """Copy every autotuner's config list, so :func:`pin` can be undone."""
    return [list(tuner.configs) for tuner in _autotuners(module)]
