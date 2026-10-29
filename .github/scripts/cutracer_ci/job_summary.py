"""The markdown job summary of a sanitizer run."""

from __future__ import annotations


def render(results: list[dict], kernel: str | None, cmd: list[str]) -> str:
    lines = [
        "## CUTracer compute-sanitizer",
        "",
        f"Kernel: `{kernel or 'all'}`  ",
        f"Command: `{' '.join(cmd)}`",
        "",
        "| Tests | Tool | Outcome | Device errors | Host API errors | Untracked launches | Peer-CTA barrier | Not printed "
        "| Test exit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        count = "-" if result["count"] is None else result["count"]
        exit_code = "timeout" if result["timed_out"] else result["returncode"]
        tests = " ".join(arg for arg in result["cmd"] if arg.split("::")[0].endswith(".py")) or "-"
        lines.append(f"| `{tests}` | {result['tool']} | {result['outcome']} | {count} | "
                     f"{result.get('api_errors', '-')} | {result.get('sanitizer_errors', '-')} | "
                     f"{result.get('peer_barrier', '-')} | {result.get('not_printed', '-')} | {exit_code} |")
    lines += [
        "", "`none` means the sanitizer reported nothing for what the test ran; it is not proof of no race. "
        "A non-zero test exit means some tests did not run to the end, so their kernels may be unchecked. "
        "Peer-CTA barrier reports are 2-CTA TMA signals the PTX ISA allows; they are not counted. "
        "Errors past the print limit are not counted either, and keep the file's kernels from counting as checked."
    ]
    for result in results:
        if result["outcome"] != "findings":
            continue
        lines += ["", f"### {result['tool']}: `{' '.join(result['cmd'])}`", "", "```"]
        lines += result["report"] or ["(no report lines parsed; see the uploaded log)"]
        lines += ["```"]
        for finding in result.get("findings", []):
            marker = "in kernel" if finding["in_kernel"] else "other kernel"
            sites = "; ".join(f"{a['role']} {a['type']} {a['function']} {a['location']}" for a in finding["accesses"])
            lines.append(f"- {finding['severity']} ({marker}): {sites}")
    return "\n".join(lines) + "\n"
