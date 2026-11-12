#!/usr/bin/env python3
"""Compare two tlx_ops_bench_ir_dump.py dumps and report which kernels' generated code changed.

Location metadata, SSA value names and register numbering are normalized away, so only code changes remain. For
each changed kernel the report lists the operation and instruction count deltas per stage and shows the normalized
TTGIR and PTX (or AMDGCN) diffs, hunks with changed operations first. Full diffs of every stage are written next to
the summary.
"""
import argparse
import collections
import difflib
import json
from pathlib import Path
import re

STAGES = ("ttir", "ttgir", "llir", "ptx", "amdgcn", "sass")
MLIR_OP = re.compile(r"^\s*(?:%[^=]*=\s*)?\"?([a-z_]+\.[a-z_][\w.]*)")
SSA = re.compile(r"%[A-Za-z_$.][\w$.-]*|%\d+(?:#\d+)?")
PTX_REG = re.compile(r"%(r|rd|rs|f|fd|p|h|hh|b|c)\d+\b")
PTX_OP = re.compile(r"^\s*(?:@!?%\w+\s+)?([a-z][\w.:]*)\b")
# Triton's dump prefixes each instruction with control codes ("--:-:0:-:1\tLDC ..."); cuobjdump uses "/*0010*/".
SASS_INSN = re.compile(r"^\s*(?:[\w-]+:[\w-]+:[\w-]+:[\w-]+:\d+\s+|/\*[0-9a-f]{4,}\*/\s+)(.*?)\s*;", re.M)
SASS_REG = re.compile(r"\b(U?R|U?P)\d+\b")
GCN_OP = re.compile(r"^\s+([sv]_[\w]+|buffer_\w+|global_\w+|ds_\w+|flat_\w+|scratch_\w+)\b")
EXCERPT_STAGES = ("ttgir", "ptx", "amdgcn")
BOOKKEEPING = {
    "mov", "selp", "not", "setp", "and", "or", "xor", "shl", "shr", "bfe", "prmt", "s_mov", "v_mov", "s_and", "v_and",
    "s_or", "v_or", "v_cndmask", "s_cselect"
}
EXCERPT_LINES = 60
# GitHub rejects comments over 65536 characters.
COMMENT_BUDGET = 55000


def strip_loc(line):
    out, i = [], 0
    while True:
        j = line.find(" loc(", i)
        if j < 0:
            out.append(line[i:])
            return "".join(out)
        out.append(line[i:j])
        depth, k = 0, j + 4
        while k < len(line):
            depth += {"(": 1, ")": -1}.get(line[k], 0)
            k += 1
            if depth == 0:
                break
        i = k


def normalize(stage, text):
    lines = []
    if stage in ("ttir", "ttgir"):
        for line in text.splitlines():
            if line.startswith("#loc") or line.startswith("{-#"):
                if line.startswith("{-#"):
                    break
                continue
            line = SSA.sub("%v", strip_loc(line)).rstrip()
            if line.strip():
                lines.append(line)
    elif stage == "llir":
        for line in text.splitlines():
            if line.startswith("!") or line.startswith("; ") or not line.strip():
                continue
            lines.append(re.sub(r",?\s*!dbg !\d+", "", re.sub(r"%\d+\b", "%v", line)).rstrip())
    elif stage == "ptx":
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(".section") and ".debug" in s:
                break
            if not s or s.startswith(("//", ".loc", ".file")):
                continue
            lines.append(PTX_REG.sub(r"%\1", line).rstrip())
    elif stage == "sass":
        lines = [SASS_REG.sub(r"\1", m.group(1)) for m in SASS_INSN.finditer(text)]
    elif stage == "amdgcn":
        # DWARF sections and .ident lines carry source paths and tool versions, not generated code.
        debug = False
        for line in text.splitlines():
            s = line.split(";")[0].rstrip()
            directive = s.strip()
            if directive.startswith((".section", ".text", ".amdgpu_metadata")):
                debug = directive.startswith(".section") and ".debug" in directive
            if debug:
                continue
            if directive and not directive.startswith((".loc", ".file", ".ident")):
                lines.append(re.sub(r"\b([sv])\d+\b|\b([sv])\[\d+:\d+\]", lambda m: m.group(1) or m.group(2), s))
    return lines


def opcode(stage, line):
    """The operation or instruction a normalized line holds, or None for declarations and labels."""
    if stage in ("ttir", "ttgir"):
        m = MLIR_OP.match(line)
    elif stage == "ptx":
        m = None if line.strip().startswith((".", "$", "{", "}", ")")) else PTX_OP.match(line)
    elif stage == "sass":
        words = re.sub(r"^@!?U?P\w*\s+", "", line).split()
        return words[0] if words else None
    elif stage == "amdgcn":
        m = GCN_OP.match(line)
    else:
        return None
    return m.group(1) if m else None


def histogram(stage, lines):
    return collections.Counter(op for op in (opcode(stage, l) for l in lines) if op)


def excerpt(stage, diff, changed_ops, limit):
    """Up to `limit` lines of the unified diff, hunks that add or remove a changed operation first."""
    hunks = []
    for line in diff[2:]:
        if line.startswith("@@"):
            hunks.append([line])
        elif hunks:
            hunks[-1].append(line)

    def weight(line):
        op = opcode(stage, line)
        if op not in changed_ops:
            return 0
        # Moves, predicates and bit twiddling change with almost any edit; arithmetic and memory ops say more.
        return 0.1 if op.split(".")[0] in BOOKKEEPING or op == "arith.constant" else 1

    hits = lambda h: sum(weight(l[1:]) for l in h[1:] if l[:1] in "+-")
    keep, used = set(), 0
    for i in sorted(range(len(hunks)), key=lambda i: (-hits(hunks[i]), i)):
        if keep and used + len(hunks[i]) > limit:
            break
        keep.add(i)
        used += len(hunks[i])
    shown = [l for i in sorted(keep) for l in hunks[i]][:limit]
    total = sum(len(h) for h in hunks)
    return {"text": "\n".join(shown), "omitted": total - len(shown)}


def deltas(a, b, limit=8):
    d = {k: b[k] - a[k] for k in set(a) | set(b) if b[k] != a[k]}
    return sorted(d.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:limit]


def load(root):
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    kernels = {}
    for c in index["compiles"]:
        kernels.setdefault(c["key"], c)
    return index, kernels


def read_stage(root, entry, stage):
    path = Path(root) / "dump" / entry["dump"] / f"{entry['name']}.{stage}"
    return path.read_text(errors="replace") if path.exists() else None


def pair(base, head):
    pairs = [(base[k], head[k]) for k in base if k in head]
    left = [v for k, v in base.items() if k not in head]
    right = [v for k, v in head.items() if k not in base]
    lb, rb = collections.defaultdict(list), collections.defaultdict(list)
    for v in left:
        lb[(v["case"], v["name"])].append(v)
    for v in right:
        rb[(v["case"], v["name"])].append(v)
    for k in list(lb):
        if len(lb[k]) == 1 and len(rb.get(k, [])) == 1:
            pairs.append((lb.pop(k)[0], rb.pop(k)[0]))
    return pairs, [v for vs in lb.values() for v in vs], [v for vs in rb.values() for v in vs]


def compare(base_root, head_root, out):
    out = Path(out)
    (out / "diffs").mkdir(parents=True, exist_ok=True)
    base_index, base = load(base_root)
    head_index, head = load(head_root)
    pairs, only_base, only_head = pair(base, head)
    groups, unchanged, missing = {}, 0, []
    for n, (b, h) in enumerate(sorted(pairs, key=lambda p: (p[0]["name"], p[0]["key"]))):
        changed, seen = {}, 0
        for stage in STAGES:
            tb, th = read_stage(base_root, b, stage), read_stage(head_root, h, stage)
            if tb is None and th is None:
                continue
            seen += 1
            lb, lh = normalize(stage, tb or ""), normalize(stage, th or "")
            if lb == lh:
                continue
            diff = list(
                difflib.unified_diff(lb, lh, f"base/{b['name']}.{stage}", f"head/{h['name']}.{stage}", lineterm="",
                                     n=3))
            path = out / "diffs" / f"{n:04d}-{b['name']}.{stage}.diff"
            path.write_text("\n".join(diff) + "\n")
            ds = deltas(histogram(stage, lb), histogram(stage, lh))
            changed[stage] = {
                "diff_lines": sum(1 for l in diff[2:] if l[:1] in "+-"), "deltas": ds, "file":
                str(path.relative_to(out))
            }
            if stage in EXCERPT_STAGES:
                changed[stage]["excerpt"] = excerpt(stage, diff, {op for op, _ in ds}, EXCERPT_LINES)
        if not seen:
            missing.append(b["name"])
            continue
        if not changed:
            unchanged += 1
            continue
        sig = json.dumps([b["name"], sorted((s, v["deltas"]) for s, v in changed.items())])
        # The first specialization of a group supplies its diff excerpts; the others share its deltas.
        weight = (1e9 * any(changed.get(s, {}).get("deltas") for s in ("ttir", "ttgir")) +
                  1e6 * sum(abs(d) for _, d in changed.get("sass", {}).get("deltas", [])) +
                  sum(abs(d) for s in ("ptx", "amdgcn") for _, d in changed.get(s, {}).get("deltas", [])))
        g = groups.setdefault(sig, {"kernel": b["name"], "stages": changed, "cases": [], "weight": weight})
        g["cases"].append(b["case"] or h["case"])
    summary = {
        "base": {k: base_index[k]
                 for k in ("triton", "device")}, "head": {k: head_index[k]
                                                          for k in ("triton", "device")}, "compared": len(pairs),
        "unchanged": unchanged, "changed": sum(len(g["cases"]) for g in groups.values()), "groups":
        sorted(groups.values(), key=lambda g: (-g["weight"], -len(g["cases"]), g["kernel"])), "missing_dumps": missing,
        "only_base": sorted({v["name"]
                             for v in only_base}), "only_head": sorted({v["name"]
                                                                        for v in only_head}), "case_errors":
        {
            "base": [c for c in base_index["cases"] if c["status"] != "ok"], "head":
            [c for c in head_index["cases"] if c["status"] != "ok"]
        }
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    text = markdown(summary)
    (out / "summary.md").write_text(text)
    return summary, text


def case_label(case):
    if not case:
        return "?"
    key, direction = case.split("|")
    parts = key.split("/")
    return "/".join(parts[3:]) if len(parts) > 3 else key


def fmt(ds):
    return ", ".join(f"`{k}` {v:+d}" for k, v in ds) or "reordering only"


def markdown(s, max_groups=20):
    lines = [
        "## Kernel IR changes", "",
        f"Base `{s['base']['triton']}`, head `{s['head']['triton']}`. Compared {s['compared']} kernel specializations: "
        f"{s['changed']} changed, {s['unchanged']} unchanged.", ""
    ]
    counted = [g for g in s["groups"] if g["weight"]]
    order_only = sum(len(g["cases"]) for g in s["groups"] if not g["weight"])
    if counted:
        lines += [
            "| Kernel | Cases | TTGIR ops | PTX/AMDGCN instructions | SASS instructions |",
            "| --- | --- | --- | --- | --- |"
        ]
        for g in counted[:max_groups]:
            st = g["stages"]
            asm = st.get("ptx") or st.get("amdgcn")
            cases = ", ".join(f"`{case_label(c)}`" for c in g["cases"][:3])
            if len(g["cases"]) > 3:
                cases += f" and {len(g['cases']) - 3} more"
            cell = lambda v: fmt(v["deltas"]) if v else "unchanged"
            lines.append(
                f"| `{g['kernel']}` | {cases} | {cell(st.get('ttgir'))} | {cell(asm)} | {cell(st.get('sass'))} |")
        if len(counted) > max_groups:
            lines.append(f"\n{len(counted) - max_groups} more groups are in the `summary.json` artifact.")
    if order_only:
        lines.append(f"\n{order_only} more specializations differ only in instruction order, register numbering or "
                     "declarations; their diffs are in the artifact.")
    for side in ("base", "head"):
        if s[f"only_{side}"]:
            lines.append(f"\nOnly in {side}: " + ", ".join(f"`{n}`" for n in s[f"only_{side}"][:20]) + ".")
    if s["missing_dumps"]:
        lines.append(f"\n{len(s['missing_dumps'])} matched kernels have no dumped IR on either side.")
    errors = {k: len(v) for k, v in s["case_errors"].items() if v}
    if errors:
        lines.append("\nCases that failed to compile: " + ", ".join(f"{k} {v}" for k, v in errors.items()) + ".")
    text = "\n".join(lines) + "\n"
    details, shown = [], 0
    for g in counted:
        block = diff_details(g)
        if not block:
            continue
        if len(text) + sum(map(len, details)) + len(block) > COMMENT_BUDGET:
            break
        details.append(block)
        shown += 1
    if details:
        text += "\n### TTGIR and PTX diffs\n\n" + "".join(details)
        if shown < sum(1 for g in counted if diff_details(g)):
            text += "\nThe remaining groups' diffs are in the artifact.\n"
    return text


def diff_details(g):
    """A collapsed block with the normalized TTGIR and PTX (or AMDGCN) diff excerpts of one group."""
    parts = []
    for stage, label in (("ttgir", "TTGIR"), ("ptx", "PTX"), ("amdgcn", "AMDGCN")):
        ex = g["stages"].get(stage, {}).get("excerpt")
        if not ex or not ex["text"]:
            continue
        parts.append(f"**{label}**\n\n```diff\n{ex['text']}\n```\n")
        if ex["omitted"]:
            parts.append(f"{ex['omitted']} more {label} diff lines are in the artifact.\n\n")
    if not parts:
        return ""
    cases = ", ".join(case_label(c)
                      for c in g["cases"][:2]) + (f" and {len(g['cases']) - 2} more" if len(g["cases"]) > 2 else "")
    return f"<details><summary><code>{g['kernel']}</code>: {cases}</summary>\n\n" + "".join(parts) + "</details>\n\n"


def brief(s, limit=3, label="Endpoint IR comparison"):
    """For an issue comment: how many kernels changed, the TTGIR changes of the largest groups, and the TTGIR and
    PTX diffs of the largest one."""
    lines = [f"{label}: {s['changed']} of {s['compared']} kernel specializations changed."]
    ranked = [g for g in s["groups"] if g["stages"].get("ttgir", {}).get("deltas")]
    for g in ranked[:limit]:
        lines.append(f"- `{g['kernel']}` ({len(g['cases'])} case(s)) TTGIR: {fmt(g['stages']['ttgir']['deltas'])}")
    top = ranked[0] if ranked else next((g for g in s["groups"] if g["weight"]), None)
    block = diff_details(top) if top else ""
    return "\n".join(lines) + ("\n\n" + block.rstrip() if block else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    _, text = compare(args.base, args.head, args.out)
    print(text)


if __name__ == "__main__":
    main()
