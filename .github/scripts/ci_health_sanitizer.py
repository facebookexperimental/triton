"""Collection-only sanitizer observations from CUTracer's retained artifacts."""

import io
import json
import re
import subprocess
import zipfile

WORKFLOW = "cutracer.yml"
TOOLS = ("memcheck", "racecheck", "synccheck", "initcheck")
SUMMARY = re.compile(r"^=+ (?:ERROR SUMMARY: (\d+) errors?(?! were)|RACECHECK SUMMARY: (\d+) hazards? displayed)")
NOT_PRINTED = re.compile(r"^=+ ERROR SUMMARY: (\d+) errors? were not printed")
LINE = re.compile(r"^=+( *)(.*)$")


def read_log(lines):
    signals = {}
    current = None
    total, omitted, cutoff = None, 0, False
    summaries = []
    summarized = False
    segment_started = False
    unfinished_segment = False

    def identity(text):
        text = re.sub(r"\s*\[\d+ hazards?\]", "", text)
        text = re.split(r"\s+by thread\b", text)[0]
        return re.sub(r"\s+(?:at address|Address) 0x[0-9a-fA-F]+", "", text)

    def finish():
        if not current:
            return
        message = identity(current["message"])
        if "Race reported between" in message:
            message = message.split(" access at ", 1)[0] + " access"
        sites = tuple(identity(site) for site in current["sites"])
        if current["kind"] == "api":
            sites = tuple(re.sub(r"\s*\[0x[0-9a-fA-F]+\]", "", site).strip() for site in sites)
        kind = "peer-barrier" if any("is not located in destination CTA" in line
                                     for line in current["example"]) else current["kind"]
        key = (kind, message, sites)
        signal = signals.setdefault(key,
                                    {**current, "kind": kind, "message": message, "sites": list(sites), "count": 0})
        signal["count"] += 1

    for raw in lines:
        line = raw.rstrip()
        if "Warning: Maximum number of hazards reached" in line:
            finish()
            current = None
            cutoff = True
            continue
        missing = NOT_PRINTED.match(line)
        if missing:
            omitted += int(missing.group(1))
            continue
        summary = SUMMARY.match(line)
        if summary:
            summaries.append(int(next(group for group in summary.groups() if group is not None)))
            summarized = True
        match = LINE.match(line)
        if not match:
            continue
        indent, text = match.groups()
        if len(indent) <= 1:
            finish()
            current = None
            if text.startswith("COMPUTE-SANITIZER"):
                unfinished_segment |= segment_started and not summarized
                segment_started = True
                summarized = False
            if not text or text.startswith(
                ("COMPUTE-SANITIZER", "ERROR SUMMARY", "RACECHECK SUMMARY", "Target application")):
                continue
            segment_started = True
            summarized = False
            kind = "api" if text.startswith("Program hit ") else "tool" if text.startswith(
                "Internal Sanitizer Error") else "device"
            current = {"kind": kind, "message": text, "sites": [], "example": []}
        if current:
            if text.startswith("Host Frame:") or text.startswith("Saved host backtrace"):
                if current["kind"] == "api" and not current["sites"] and text.startswith("Host Frame:"):
                    current["sites"].append(text)
                continue
            if text.startswith("at ") or re.search(r"(?:Read|Write) access at ", text):
                current["sites"].append(text)
            if len(current["example"]) < 8:
                current["example"].append(text)
    finish()
    incomplete_tail = unfinished_segment or (segment_started and not summarized)
    total = sum(summaries) if summaries and not incomplete_tail else None
    return {
        "raw_count": total, "not_printed": None if cutoff or incomplete_tail else omitted, "truncated": cutoff
        or omitted > 0 or incomplete_tail, "signals": list(signals.values())
    }


def op_group(tests):
    if isinstance(tests, str):
        tests = [tests]
    paths = {test.split("::", 1)[0] for test in tests if test.split("::", 1)[0].endswith(".py")}
    ops = set()
    for path in paths:
        match = re.search(r"/tlx_ops/test_(.+)_(?:sm\d+(?:up)?|gfx\d+|any)\.py$", path)
        if match is None:
            return "Unattributed / compiler tests"
        ops.add(match.group(1))
    return next(iter(ops)) if len(ops) == 1 else "Mixed op suites" if ops else "Unattributed / compiler tests"


def archive_data(content, artifact):
    rows = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        result_name = next((name for name in names if name in ("results.json", "sanitize/results.json")), None)
        if result_name is None:
            return rows
        payload = json.loads(archive.read(result_name))
        for result in payload.get("results", []):
            tool = result["tool"]
            if tool not in TOOLS:
                continue
            path = result.get("log", "")
            relative = path.split("/cutracer/", 1)[1] if "/cutracer/" in path else path
            member = relative if relative in names else None
            row = {
                "tool": tool,
                "tests": result.get("tests") or result.get("cmd") or payload.get("cmd", []),
                "elapsed_s": result.get("elapsed_s"),
                "exit": result.get("returncode"),
                "timed_out": result.get("timed_out", False),
                "kernels": result.get("kernels"),
                "artifact": artifact["name"],
                "log": member,
                "raw_count": None,
                "not_printed": None,
                "truncated": False,
                "signals": [],
            }
            if member:
                with io.TextIOWrapper(archive.open(member), errors="replace") as handle:
                    row.update(read_log(handle))
            row["op"] = op_group(row["tests"])
            row["completion"] = "complete" if row["raw_count"] is not None and not row["timed_out"] and row[
                "exit"] == 0 and not row["truncated"] and not any(signal["kind"] == "tool"
                                                                  for signal in row["signals"]) else "incomplete"
            rows.append(row)
    return rows


def collect(run, api, download, repo):
    rows, missing, retry = [], [], False
    artifacts = [
        artifact for page in api(f"repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100")
        for artifact in page["artifacts"] if artifact["name"].startswith("cutracer-sanitizer-")
    ]
    latest = {}
    for artifact in artifacts:
        if artifact["name"] not in latest or artifact["id"] > latest[artifact["name"]]["id"]:
            latest[artifact["name"]] = artifact
    for artifact in latest.values():
        gpu = artifact["name"].rsplit("-", 1)[-1].upper()
        if artifact["expired"]:
            missing.append({"gpu": gpu, "reason": "artifact expired"})
            continue
        try:
            parsed = archive_data(download(artifact["archive_download_url"]), artifact)
        except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
            missing.append({"gpu": gpu, "reason": str(error)})
            retry = True
            continue
        if not parsed:
            missing.append({"gpu": gpu, "reason": "artifact has no sanitizer results"})
        for row in parsed:
            row["gpu"] = gpu
            rows.append(row)
    return {"tools": rows, "missing": missing, "retry": retry}


def signal_attribution(signal, kernels):
    if signal["kind"] not in ("device", "peer-barrier"):
        return "non-kernel diagnostic"
    names = [match.group(1) for site in signal["sites"] if (match := re.search(r"\bat (.+?)\+0x[0-9a-fA-F]+", site))]
    if not kernels or not names:
        return "unknown"
    if all(name in kernels for name in names):
        return "compiled kernel match"
    if any(name in kernels for name in names):
        return "mixed attribution"
    return "unmatched / external"


def observation_summary(observations):
    signatures = {}
    tool_rows = []
    for row in observations:
        for signal in row["signals"]:
            revision = (row.get("sha"), row.get("run_id"))
            attribution = signal_attribution(signal, row.get("kernels"))
            key = (row.get("gpu"), row["tool"], revision, signal["kind"], signal["message"], tuple(signal["sites"]),
                   attribution)
            entry = signatures.setdefault(
                key, {
                    "gpu": row.get("gpu"),
                    "tool": row["tool"],
                    "sha": row.get("sha"),
                    "kind": signal["kind"],
                    "message": signal["message"],
                    "sites": signal["sites"],
                    "example": signal.get("example", []),
                    "printed_reports": 0,
                    "evidence": [],
                    "attribution": "unknown",
                    "attributions": [],
                })
            entry["printed_reports"] += signal["count"]
            entry["evidence"].append(
                {key: row.get(key)
                 for key in ("run_id", "t", "url", "log", "artifact", "tests", "completion")})
            entry["attributions"].append(attribution)
    for tool in TOOLS:
        selected = [row for row in observations if row["tool"] == tool]
        elapsed = [row["elapsed_s"] for row in selected if row.get("elapsed_s") is not None]
        tool_rows.append({
            "tool":
            tool,
            "observations":
            len(selected),
            "complete":
            sum(row["completion"] == "complete" for row in selected),
            "timed_out":
            sum(bool(row.get("timed_out")) for row in selected),
            "truncated":
            sum(bool(row.get("truncated")) for row in selected),
            "missing_log":
            sum(row.get("log") is None for row in selected),
            "unknown_count":
            sum(row["raw_count"] is None for row in selected),
            "raw_known":
            sum(row["raw_count"] for row in selected if row["raw_count"] is not None),
            "unprinted_known":
            sum(row["not_printed"] for row in selected if row.get("not_printed") is not None),
            "unknown_omissions":
            sum(row.get("not_printed") is None for row in selected),
            "timed_observations":
            len(elapsed),
            "elapsed_s":
            sum(elapsed) if elapsed else None,
            "max_elapsed_s":
            max(elapsed) if elapsed else None,
        })
    kinds = ("device", "api", "peer-barrier", "tool")
    entries = sorted(
        signatures.values(), key=lambda item:
        (kinds.index(item["kind"]) if item["kind"] in kinds else 4, -item["printed_reports"], item["message"]))
    for entry in entries:
        attributions = set(entry.pop("attributions"))
        entry["attribution"] = next(iter(attributions)) if len(attributions) == 1 else "mixed attribution"
        entry["observed_logs"] = len({(e["run_id"], e["log"]) for e in entry["evidence"]})
        entry["observed_runs"] = len({e["run_id"] for e in entry["evidence"]})
    tracked = [entry for entry in entries if entry["attribution"] == "compiled kernel match"]
    unknown = [entry for entry in entries if entry["attribution"] in ("unknown", "mixed attribution")]
    excluded = [entry for entry in entries if entry["attribution"] in ("unmatched / external", "non-kernel diagnostic")]
    return {
        "signatures": tracked,
        "unattributed_signatures": len(unknown),
        "excluded_signatures": len(excluded),
        "tracked_printed_reports": sum(entry["printed_reports"] for entry in tracked),
        "kinds": {kind: sum(entry["kind"] == kind
                            for entry in tracked)
                  for kind in kinds},
        "tools": tool_rows,
        "observations": len(observations),
        "complete": sum(row["completion"] == "complete" for row in observations),
        "runs": len({row.get("run_id")
                     for row in observations}),
    }


def view(runs, asof):
    selected = sorted((run for run in runs if run.get("wf") == WORKFLOW and run["created_at"] <= asof),
                      key=lambda run: run["created_at"], reverse=True)[:30]
    rows = []
    for run in selected:
        data = run.get("sanitizer") or {"tools": [], "missing": []}
        rows.append({
            "id": run["id"], "t": run["created_at"], "sha": run.get("head_sha"), "url": run.get("url"), "event":
            run["event"], "status": run.get("status"), "conclusion": run.get("conclusion"), **data
        })
    groups = {}
    for run in rows:
        for observation in run["tools"]:
            op = observation.get("op") or op_group(observation.get("tests", []))
            groups.setdefault(op, []).append(
                {"run_id": run["id"], "t": run["t"], "url": run["url"], "sha": run["sha"], **observation})
    ops = [{"op": op, "observations": groups[op], "summary": observation_summary(groups[op])} for op in sorted(groups)]
    ops.sort(
        key=lambda group: (-group["summary"]["kinds"]["device"], group["op"].startswith("Unattributed"), group["op"]))
    return {
        "runs": rows, "tools": list(TOOLS), "ops": ops, "summary":
        observation_summary([row for group in groups.values() for row in group])
    }
