"""Unit tests for environment verification. No GPU required.

The GPU readings below were all taken from a B200 on devgpu006 -- unmanaged,
under ``third_party/tlx/denoise.sh``, and under sustained load -- so these
tests pin the decision logic against hardware that was observed rather than
against invented numbers.
"""

import pytest

from _harness import MAX_CLOCK_IDR, ClockTrace, GpuState, decode_event_reasons, parse_cpulist, relative_interdecile_range  # noqa: E501
from _harness.denoise import DEGRADING_REASONS


def test_decode_event_reasons():
    assert decode_event_reasons(0x1) == ["gpu_idle"]
    assert decode_event_reasons(0x4) == ["sw_power_cap"]
    assert decode_event_reasons(0x44) == ["hw_thermal_slowdown", "sw_power_cap"]
    assert decode_event_reasons(0x0) == []


def test_sw_power_cap_is_not_degrading():
    """Measured: a B200 under an 8192^3 fp16 GEMM reports sw_power_cap
    continuously, with or without ``nvidia-smi -lgc``. Counting it as
    degradation would flag every healthy run, so the signal has to exclude it.
    """
    assert not DEGRADING_REASONS & 0x4
    assert not DEGRADING_REASONS & 0x1  # gpu_idle says nothing about the window
    assert DEGRADING_REASONS & 0x40  # hw_thermal_slowdown does count
    assert DEGRADING_REASONS & 0x8  # hw_slowdown too


def test_no_clock_lock_check_exists():
    """Regression guard for the phase-2 finding.

    An earlier version asserted "the SM clock must be near maximum". On B200
    that is false for every compute-bound run -- the card is power-governed at
    ~830/1965 MHz whether or not the clock is pinned -- so the check fired on
    correct runs. If someone reintroduces it, this should argue back.
    """
    import _harness

    assert not hasattr(_harness, "clocks_locked")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0-3", {0, 1, 2, 3}),
        ("0,2,4", {0, 2, 4}),
        ("0-2,8", {0, 1, 2, 8}),
        # The real node0 cpulist from devgpu006.
        ("0-95,192-287", set(range(0, 96)) | set(range(192, 288))),
        ("", set()),
    ],
)
def test_parse_cpulist(text, expected):
    assert parse_cpulist(text) == expected


def test_gpu_state_to_dict_expands_reason_names():
    state = GpuState(available=True, sm_clock_mhz=1845, max_sm_clock_mhz=1965, event_reasons=0x4)
    d = state.to_dict()
    assert d["event_reason_names"] == ["sw_power_cap"]
    assert d["sm_clock_mhz"] == 1845


def test_gpu_state_unknown_without_nvidia_smi():
    assert GpuState(available=False).event_reason_names == []


def test_relative_interdecile_range_ignores_the_idle_ramp():
    """The real failure this replaced: a 6 s window on a denoised B200 sampled
    120 clocks, ~830 MHz throughout except the handful before the first launch
    at the 990 MHz idle clock. min/max called that a 26% spread and every run
    unstable; deciles see the steady state."""
    steady = [830] * 110
    ramp = [990, 985, 970, 950, 900]
    values = ramp + steady
    assert (max(values) - min(values)) / 830 > 0.15  # what min/max would have said
    assert relative_interdecile_range(values) < MAX_CLOCK_IDR


def test_relative_interdecile_range_still_sees_real_movement():
    drifting = list(range(700, 1000, 3))  # a card sliding across the window
    assert relative_interdecile_range(drifting) > MAX_CLOCK_IDR


def test_relative_interdecile_range_degrades_to_min_max_on_tiny_samples():
    assert relative_interdecile_range([100, 110], median=100) == pytest.approx(0.1)
    assert relative_interdecile_range([]) is None


def test_clock_trace_stability():
    ramping = ClockTrace(samples=21, min_mhz=802, median_mhz=832, max_mhz=990, rel_idr=0.23)
    assert ramping.stable is False

    steady = ClockTrace(samples=120, min_mhz=828, median_mhz=832, max_mhz=990, rel_idr=0.014)
    assert steady.rel_idr < MAX_CLOCK_IDR
    assert steady.stable is True


def test_clock_trace_degradation_overrides_a_tight_spread():
    """A card that thermally slowed but happened to do so smoothly is still
    not a valid measurement."""
    trace = ClockTrace(samples=100, min_mhz=828, median_mhz=832, max_mhz=840, rel_idr=0.01,
                       reasons=("hw_thermal_slowdown", ), degrading=("hw_thermal_slowdown", ))
    assert trace.stable is False


def test_clock_trace_unknown_without_samples():
    assert ClockTrace(samples=0).stable is None
    assert ClockTrace(samples=0).to_dict()["stable"] is None


def test_power_target_matches_the_part():
    """Same table as denoise.sh. A *fixed* cap is the point, not a high one:
    an unpinned cap is one more thing that can differ between two runs being
    compared."""
    from _harness.denoise import AMD, NVIDIA, Device

    assert Device(NVIDIA, 0, "NVIDIA B200").power_target_w == 750
    assert Device(NVIDIA, 0, "NVIDIA H100 80GB HBM3").power_target_w == 700
    assert Device(NVIDIA, 0, "NVIDIA GB200").power_target_w == 1200
    assert Device(AMD, 0, "MI350X").power_target_w == 1000
    assert Device(AMD, 0, "MI355X").power_target_w == 1400
    # Unknown parts get no target rather than a guessed one; the governor then
    # leaves the card's own limit alone.
    assert Device(NVIDIA, 0, "NVIDIA L4").power_target_w is None


def test_visibility_variable_follows_the_vendor():
    from _harness.denoise import AMD, NVIDIA, Device

    assert Device(NVIDIA, 2, "NVIDIA B200").visibility_env == "CUDA_VISIBLE_DEVICES"
    assert Device(AMD, 2, "MI350X").visibility_env == "HIP_VISIBLE_DEVICES"


def test_auto_selection_picks_the_least_used_gpu(monkeypatch):
    """By free memory, not by index: on a shared box a co-tenant is the failure
    mode most likely to go unnoticed, and index 0 is as likely to be busy as
    any other."""
    import _harness.denoise as denoise_mod
    from _harness.denoise import NVIDIA, Device

    fleet = [
        Device(NVIDIA, 0, "NVIDIA B200", memory_used_mib=14178),
        Device(NVIDIA, 1, "NVIDIA B200", memory_used_mib=4),
        Device(NVIDIA, 2, "NVIDIA B200", memory_used_mib=1254),
    ]
    monkeypatch.setattr(denoise_mod, "list_devices", lambda: fleet)
    assert denoise_mod.select_device("auto").index == 1
    assert denoise_mod.select_device("2").index == 2
    with pytest.raises(ValueError):
        denoise_mod.select_device("9")


def test_governor_is_inert_without_a_device():
    """No GPU is a reason to report honestly, not to crash."""
    from _harness.denoise import Governor

    with Governor(None) as g:
        pass
    assert g.applied == []
    assert g.skipped and "no GPU" in g.skipped[0]


def test_governor_can_be_disabled():
    from _harness.denoise import Governor, NVIDIA, Device

    with Governor(Device(NVIDIA, 0, "NVIDIA B200"), enable=False) as g:
        pass
    assert g.applied == []
