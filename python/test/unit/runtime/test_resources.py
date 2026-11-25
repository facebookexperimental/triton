"""Tests for the OutOfResources shared-memory breakdown report.

The compiler records per-allocation shared-memory usage as a JSON string in
the `ttg.shared_memory_breakdown` module attribute; backends copy it verbatim
into kernel metadata, and OutOfResources parses it only on error
creation/display. These tests use real-package imports (no synthetic source
fallback) and CPU-runnable checks only.
"""

import json
import pickle

import pytest

from triton.runtime.errors import OutOfResources, attach_shared_memory_breakdown


def _allocation(kind="explicit", barrier_backed=False, operation="op", offset=0, size=100, location="loc"):
    return {
        "kind": kind,
        "barrier_backed": barrier_backed,
        "operation": operation,
        "offset": offset,
        "size": size,
        "location": location,
    }


def _reservation(kind="alignment", operation="op", offset=0, size=64, location="loc"):
    return {"kind": kind, "operation": operation, "offset": offset, "size": size, "location": location}


def _function(name="kernel", footprint=1024, omitted=0, allocations=None):
    return {
        "name": name,
        "reserved_bytes": footprint,
        "omitted_allocations": omitted,
        "allocations": allocations if allocations is not None else [_allocation(size=100)],
    }


def _report(total=1024, functions=None, reservations=None):
    return json.dumps({
        "version": 1,
        "reserved_bytes": total,
        "functions": functions if functions is not None else [_function(footprint=total)],
        "reservations": reservations if reservations is not None else [],
    })


def test_legacy_message_unchanged():
    err = OutOfResources(100, 50, "shared memory")
    assert str(err) == ("out of resource: shared memory, Required: 100, Hardware limit: 50. "
                        "Reducing block sizes or `num_stages` may help.")


def test_legacy_pickle_roundtrip():
    err = OutOfResources(100, 50, "shared memory")
    revived = pickle.loads(pickle.dumps(err))
    assert (revived.required, revived.limit, revived.name) == (100, 50, "shared memory")
    assert revived.shared_memory_breakdown is None
    assert str(revived) == str(err)


def test_report_pickle_roundtrip():
    raw = _report()
    err = OutOfResources(1024, 512, "shared memory", raw)
    revived = pickle.loads(pickle.dumps(err))
    assert revived.shared_memory_breakdown == raw
    assert revived.breakdown_report is not None
    assert str(revived) == str(err)


@pytest.mark.parametrize("raw", [None, 123, b"bytes", "", "not json", "[1,2]", '{"version": 1}'])
def test_malformed_payload_falls_back_to_legacy(raw):
    err = OutOfResources(1024, 512, "shared memory", raw)
    assert err.breakdown_report is None
    assert str(err) == ("out of resource: shared memory, Required: 1024, Hardware limit: 512. "
                        "Reducing block sizes or `num_stages` may help.")


def test_wrong_version_rejected():
    payload = json.dumps({"version": 2, "reserved_bytes": 1024, "functions": [], "reservations": []})
    assert OutOfResources(1024, 512, "shared memory", payload).breakdown_report is None


def test_bool_version_rejected():
    # bool is a subclass of int; version=True must not pass as version 1.
    payload = json.dumps({"version": True, "reserved_bytes": 1024, "functions": [], "reservations": []})
    assert OutOfResources(1024, 512, "shared memory", payload).breakdown_report is None


@pytest.mark.parametrize("total", [100, 2048, True, "1024", None])
def test_total_must_equal_required(total):
    payload = json.dumps({"version": 1, "reserved_bytes": total, "functions": [], "reservations": []})
    assert OutOfResources(1024, 512, "shared memory", payload).breakdown_report is None


def test_unknown_allocation_kind_rejected():
    payload = _report(functions=[_function(allocations=[_allocation(kind="mystery")])])
    assert OutOfResources(1024, 512, "shared memory", payload).breakdown_report is None


def test_unknown_reservation_kind_rejected():
    payload = _report(reservations=[_reservation(kind="mystery")])
    assert OutOfResources(1024, 512, "shared memory", payload).breakdown_report is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("offset", -1),
        ("size", -1),
        ("offset", True),
        ("size", "x"),
        ("barrier_backed", 1),
        ("operation", 5),
        ("location", None),
    ],
)
def test_bad_allocation_field_types_rejected(field, value):
    kwargs = {field: value}
    payload = _report(functions=[_function(allocations=[_allocation(**kwargs)])])
    assert OutOfResources(1024, 512, "shared memory", payload).breakdown_report is None


def test_allocation_range_checked_against_function_footprint():
    # offset + size (900 + 200) exceeds the function footprint of 1024.
    payload = _report(functions=[_function(footprint=1024, allocations=[_allocation(offset=900, size=200)])])
    assert OutOfResources(1024, 512, "shared memory", payload).breakdown_report is None


def test_function_footprint_cannot_exceed_total():
    payload = _report(total=1024, functions=[_function(footprint=2048)])
    assert OutOfResources(1024, 512, "shared memory", payload).breakdown_report is None


def test_reservation_range_checked_against_total():
    payload = _report(reservations=[_reservation(offset=1000, size=100)])
    assert OutOfResources(1024, 512, "shared memory", payload).breakdown_report is None


def test_overlapping_reused_ranges_are_legal():
    # Two functions each using the full footprint overlap completely through
    # buffer reuse; the summed allocation bytes (2048) exceed the total (1024)
    # and must still validate because footprints combine by max, never by sum.
    big = [_allocation(offset=0, size=1024)]
    payload = _report(
        total=1024,
        functions=[
            _function(name="a", footprint=1024, allocations=big),
            _function(name="b", footprint=1024, allocations=big),
        ],
    )
    err = OutOfResources(1024, 512, "shared memory", payload)
    assert err.breakdown_report is not None
    assert "not additive" in str(err)


@pytest.mark.parametrize("name", ["threads", "tensor memory"])
def test_report_ignored_for_non_shared_memory(name):
    err = OutOfResources(1024, 512, name, _report())
    assert err.breakdown_report is None
    assert "breakdown" not in str(err)


def test_negative_total_rejected_even_if_matching():
    payload = json.dumps({"version": 1, "reserved_bytes": -8, "functions": [], "reservations": []})
    assert OutOfResources(-8, 512, "shared memory", payload).breakdown_report is None


def test_deeply_nested_payload_falls_back_to_legacy():
    payload = "[" * 100000
    err = OutOfResources(1024, 512, "shared memory", payload)
    assert err.breakdown_report is None
    assert "breakdown" not in str(err)


@pytest.mark.parametrize("kind", ["explicit", "scratch", "virtual"])
def test_all_allocation_kinds_listed(kind):
    payload = _report(functions=[_function(allocations=[_allocation(kind=kind)])])
    err = OutOfResources(1024, 512, "shared memory", payload)
    assert f"[{kind}]" in str(err)


@pytest.mark.parametrize("kind", ["alignment", "cluster_barrier", "minimum", "profiling"])
def test_all_reservation_kinds_listed(kind):
    payload = _report(reservations=[_reservation(kind=kind)])
    assert f"reservation [{kind}]" in str(OutOfResources(1024, 512, "shared memory", payload))


def test_profiling_reservation_preserves_kernel_breakdown():
    raw = _report(
        total=136,
        functions=[_function(footprint=1, allocations=[_allocation(size=1)])],
        reservations=[
            _reservation(offset=1, size=7),
            _reservation(kind="profiling", operation="ttg.local_alloc", offset=8, size=128),
        ],
    )
    metadata = json.loads(json.dumps({"shared": 136, "shared_memory_breakdown": raw}))
    err = OutOfResources(metadata["shared"], 128, "shared memory", metadata["shared_memory_breakdown"])
    assert err.breakdown_report is not None
    message = str(err)
    assert "reserved 136 bytes" in message
    assert "kernel [explicit]: 1 bytes at offset 0" in message
    assert "reservation [alignment]: 7 bytes at offset 1" in message
    assert "reservation [profiling]: 128 bytes at offset 8 (op `ttg.local_alloc`" in message
    assert str(pickle.loads(pickle.dumps(err))) == message


def test_virtual_caveat_present():
    payload = _report(functions=[_function(allocations=[_allocation(kind="virtual")])])
    message = str(OutOfResources(1024, 512, "shared memory", payload))
    assert "Virtual ranges reserve callee footprints; do not count callees separately." in message


def test_multiple_roots_ranked_globally():
    small = _allocation(operation="small", size=100)
    large = _allocation(operation="large", size=900)
    payload = _report(
        total=1024,
        functions=[_function(name="first", allocations=[small]),
                   _function(name="second", allocations=[large])],
    )
    message = str(OutOfResources(1024, 512, "shared memory", payload))
    assert "first" in message and "second" in message
    assert message.index("large") < message.index("small")


def test_truncation_bounds_and_aggregates_omitted():
    allocations = [_allocation(operation=f"op{i}", offset=0, size=10) for i in range(12)]
    reservations = [_reservation(operation=f"res{i}", size=8) for i in range(12)]
    payload = _report(functions=[_function(omitted=5, allocations=allocations)], reservations=reservations)
    message = str(OutOfResources(1024, 512, "shared memory", payload))
    assert message.count("op `op") == 10
    assert "... and 7 more allocation(s) omitted" in message
    assert "... and 2 more reservation(s) omitted" in message
    # Aggregate count only: never one omission line per root.
    assert message.count("omitted") == 2


def test_long_fields_capped():
    payload = _report(functions=[_function(allocations=[_allocation(operation="o" * 500, location="l" * 500)])])
    message = str(OutOfResources(1024, 512, "shared memory", payload))
    assert "o" * 500 not in message and "l" * 500 not in message
    assert message.count("...") >= 2
    # Each capped field renders at most 200 chars including the ellipsis.
    assert "o" * 197 + "..." in message
    assert "o" * 198 not in message


def test_control_chars_and_separators_escaped():
    evil = "a\x00b\x1bc\nd,e:f[g]h`i\\j"
    payload = _report(functions=[_function(name=evil, allocations=[_allocation(operation=evil, location=evil)])])
    message = str(OutOfResources(1024, 512, "shared memory", payload))
    # Raw control chars must be gone; the field newline is escaped to a literal
    # backslash-n (message line separators remain real newlines by design).
    for ch in ("\x00", "\x1b"):
        assert ch not in message
    assert "\\x00" in message and "\\x1b" in message and "\\n" in message


@pytest.mark.parametrize(
    "text,escaped",
    [
        ("a\u200bb", "a\\u200bb"),
        ("a\x7f", "a\\x7f"),
        ("a\x00b", "a\\x00b"),
    ],
)
def test_nonprintable_chars_escaped(text, escaped):
    payload = _report(functions=[_function(allocations=[_allocation(operation=text)])])
    message = str(OutOfResources(1024, 512, "shared memory", payload))
    assert escaped in message
    assert text not in message


@pytest.mark.parametrize("text", ["caf\u00e9 \u4e2d\u6587", "emoji \U0001f600"])
def test_printable_unicode_preserved(text):
    payload = _report(functions=[_function(allocations=[_allocation(operation=text)])])
    message = str(OutOfResources(1024, 512, "shared memory", payload))
    assert text in message


def test_no_new_optimization_advice():
    payload = _report()
    message = str(OutOfResources(1024, 512, "shared memory", payload))
    assert message.startswith("out of resource: shared memory, Required: 1024, Hardware limit: 512. "
                              "Reducing block sizes or `num_stages` may help.\n")


def test_metadata_json_roundtrip():
    raw = _report()
    metadata = {"shared": 1024, "shared_memory_breakdown": raw}
    serialized = json.loads(json.dumps(metadata))
    err = OutOfResources(serialized["shared"], 512, "shared memory", serialized.get("shared_memory_breakdown"))
    assert err.breakdown_report is not None
    assert "Shared memory breakdown" in str(err)


class _FakeModule:
    """Minimal stand-in for the MLIR module binding (get_str_attr only)."""

    def __init__(self, value):
        self._value = value

    def get_str_attr(self, _name):
        return self._value


def test_attach_stores_raw_string_verbatim():
    raw = _report()
    metadata = {}
    attach_shared_memory_breakdown(_FakeModule(raw), metadata)
    assert metadata["shared_memory_breakdown"] == raw


@pytest.mark.parametrize("value", [None, "", 123, b"raw"])
def test_attach_ignores_missing_or_non_string(value):
    metadata = {}
    attach_shared_memory_breakdown(_FakeModule(value), metadata)
    assert "shared_memory_breakdown" not in metadata


def test_attach_without_getter_is_noop():
    metadata = {}
    attach_shared_memory_breakdown(object(), metadata)
    assert metadata == {}
