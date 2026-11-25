import json
from typing import Optional

from ..errors import TritonError


class InterpreterError(TritonError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        return self.error_message or ""


class OutOfResources(TritonError):

    def __init__(self, required, limit, name, shared_memory_breakdown=None):
        self.required = required
        self.limit = limit
        self.name = name
        self.shared_memory_breakdown = shared_memory_breakdown
        if name == "shared memory":
            self.breakdown_report = _parse_shared_memory_breakdown(shared_memory_breakdown, required)
        else:
            self.breakdown_report = None

    def __str__(self) -> str:
        message = (f"out of resource: {self.name}, Required: {self.required}, "
                   f"Hardware limit: {self.limit}. Reducing block sizes or `num_stages` may help.")
        if self.breakdown_report is not None:
            message += "\n" + _format_shared_memory_breakdown(self.breakdown_report)
        return message

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        if self.shared_memory_breakdown is None:
            return (type(self), (self.required, self.limit, self.name))
        return (type(self), (self.required, self.limit, self.name, self.shared_memory_breakdown))


# Shared-memory breakdown: the compiler may encode diagnostic JSON in IR;
# backends copy the decoded JSON verbatim into kernel metadata.

_BREAKDOWN_VERSION = 1
_BREAKDOWN_MAX_LISTED_ALLOCATIONS = 10
_BREAKDOWN_MAX_LISTED_RESERVATIONS = 10
_BREAKDOWN_MAX_FIELD_CHARS = 200
_BREAKDOWN_ALLOCATION_KINDS = ("explicit", "scratch", "virtual")
_BREAKDOWN_RESERVATION_KINDS = ("alignment", "cluster_barrier", "minimum", "profiling")


def _check_breakdown_range(offset, size, bound):
    # type(x) is int: bool is a subclass of int but never a valid count/size.
    return type(offset) is int and type(size) is int and 0 <= offset and 0 <= size and offset + size <= bound


def _is_valid_breakdown_entry(entry, bound, kinds, barrier_backed):
    if not isinstance(entry, dict):
        return False
    if entry.get("kind") not in kinds:
        return False
    if barrier_backed and type(entry.get("barrier_backed")) is not bool:
        return False
    if not isinstance(entry.get("operation"), str) or not isinstance(entry.get("location"), str):
        return False
    return _check_breakdown_range(entry.get("offset"), entry.get("size"), bound)


def _parse_shared_memory_breakdown(raw, required):
    """Validate a raw breakdown payload; return the parsed report or None."""
    if not isinstance(raw, str):
        return None
    try:
        report = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    if not isinstance(report, dict):
        return None
    if type(report.get("version")) is not int or report["version"] != _BREAKDOWN_VERSION:
        return None
    total = report.get("reserved_bytes")
    if type(total) is not int or total < 0 or total != required:
        return None
    functions = report.get("functions")
    reservations = report.get("reservations")
    if not isinstance(functions, list) or not isinstance(reservations, list):
        return None
    for func in functions:
        if not isinstance(func, dict) or not isinstance(func.get("name"), str):
            return None
        footprint = func.get("reserved_bytes")
        if type(footprint) is not int or footprint < 0 or footprint > total:
            return None
        omitted = func.get("omitted_allocations")
        if type(omitted) is not int or omitted < 0:
            return None
        allocations = func.get("allocations")
        if not isinstance(allocations, list):
            return None
        if any(not _is_valid_breakdown_entry(a, footprint, _BREAKDOWN_ALLOCATION_KINDS, True) for a in allocations):
            return None
    if any(not _is_valid_breakdown_entry(r, total, _BREAKDOWN_RESERVATION_KINDS, False) for r in reservations):
        return None
    return report


def _escape_breakdown_field(value):
    text = value if isinstance(value, str) else str(value)
    escaped = "".join(ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii") for ch in text)
    if len(escaped) > _BREAKDOWN_MAX_FIELD_CHARS:
        escaped = escaped[:_BREAKDOWN_MAX_FIELD_CHARS - 3] + "..."
    return escaped


def _format_shared_memory_breakdown(report):
    total = report["reserved_bytes"]
    lines = [
        f"Shared memory breakdown: reserved {total} bytes in {len(report['functions'])} function(s). "
        "Allocation ranges may overlap through buffer reuse and sizes are not additive. "
        "Virtual ranges reserve callee footprints; do not count callees separately.",
    ]
    ranked = [(alloc["size"], func["name"], alloc) for func in report["functions"] for alloc in func["allocations"]]
    ranked.sort(key=lambda item: item[0], reverse=True)
    shown = ranked[:_BREAKDOWN_MAX_LISTED_ALLOCATIONS]
    for size, name, alloc in shown:
        barrier = ", barrier-backed" if alloc["barrier_backed"] else ""
        lines.append(f"  {_escape_breakdown_field(name)} [{alloc['kind']}{barrier}]: "
                     f"{size} bytes at offset {alloc['offset']} "
                     f"(op `{_escape_breakdown_field(alloc['operation'])}` "
                     f"at `{_escape_breakdown_field(alloc['location'])}`)")
    omitted = len(ranked) + sum(func["omitted_allocations"] for func in report["functions"]) - len(shown)
    if omitted > 0:
        lines.append(f"  ... and {omitted} more allocation(s) omitted")
    reservations = sorted(report["reservations"], key=lambda entry: entry["size"], reverse=True)
    for res in reservations[:_BREAKDOWN_MAX_LISTED_RESERVATIONS]:
        lines.append(f"  reservation [{res['kind']}]: {res['size']} bytes at offset {res['offset']} "
                     f"(op `{_escape_breakdown_field(res['operation'])}` "
                     f"at `{_escape_breakdown_field(res['location'])}`)")
    omitted_reservations = len(reservations) - min(len(reservations), _BREAKDOWN_MAX_LISTED_RESERVATIONS)
    if omitted_reservations > 0:
        lines.append(f"  ... and {omitted_reservations} more reservation(s) omitted")
    return "\n".join(lines)


def attach_shared_memory_breakdown(src, metadata):
    """Copy decoded compiler diagnostic JSON into metadata."""
    get_breakdown = getattr(src, "get_shared_memory_breakdown", None)
    get_str_attr = getattr(src, "get_str_attr", None)
    if get_breakdown is None and get_str_attr is None:
        return
    try:
        raw = get_breakdown() if get_breakdown is not None else get_str_attr("ttg.shared_memory_breakdown")
    except UnicodeDecodeError:
        return
    if isinstance(raw, str) and raw:
        metadata["shared_memory_breakdown"] = raw


class PTXASError(TritonError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        error_message = self.error_message or ""
        return f"PTXAS error: {error_message}"


class AutotunerError(TritonError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        error_message = self.error_message or ""
        return f"Autotuner error: {error_message}"
