"""The markdown job summary of a sanitizer run."""

from __future__ import annotations


def render(results: list[dict], kernel: str | None, cmd: list[str]) -> str:
    lines = [
        "## CUTracer compute-sanitizer",
        "",
        f"Kernel: `{kernel or 'all'}`  ",
        f"Command: `{' '.join(cmd)}`",
        "",
        "| Tool | Outcome | Device errors | Host API errors | Test exit |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        count = "-" if result["count"] is None else result["count"]
        exit_code = "timeout" if result["timed_out"] else result["returncode"]
        api_errors = result.get("api_errors", "-")
        lines.append(f"| {result['tool']} | {result['outcome']} | {count} | {api_errors} | {exit_code} |")
    lines += ["", "`none` means the sanitizer reported nothing for what the test ran; it is not proof of no race."]
    for result in results:
        if result["outcome"] != "findings":
            continue
        lines += ["", f"### {result['tool']}", "", "```"]
        lines += result["report"] or ["(no report lines parsed; see the uploaded log)"]
        lines += ["```"]
        for finding in result.get("findings", []):
            marker = "in kernel" if finding["in_kernel"] else "other kernel"
            sites = "; ".join(f"{a['role']} {a['type']} {a['function']} {a['location']}" for a in finding["accesses"])
            lines.append(f"- {finding['severity']} ({marker}): {sites}")
    return "\n".join(lines) + "\n"
