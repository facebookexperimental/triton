#!/usr/bin/env python3
"""Select the tests affected by a kernel-only change.

On pull requests that touch only TLX kernels (``third_party/tlx/tutorials/``
and ``third_party/tlx/ops/``), most GPU CI suites exercise code the change
cannot influence -- e.g. a Blackwell FA tutorial edit cannot break HSTU. This
script maps the changed files to the tests that import them (by scanning test
imports, not by hardcoded lists) so workflows can skip the rest.

Anything outside the kernel roots -- compiler code, build files, CI config,
unknown paths, or an unreadable/empty changed-files list -- fails open to a
full run. There is deliberately no selection knowledge for compiler changes
yet: those always run everything.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

KERNEL_PREFIXES = ("third_party/tlx/tutorials/", "third_party/tlx/ops/")
TEST_PREFIXES = (
    "python/test/",
    "third_party/tlx/tutorials/testing/",
    "test/",
    "unittest/",
)
TUTORIAL_TEST_BASENAMES = ("test_*.py", "*_test.py")
IGNORED_PREFIXES = (
    "docs/",
    "website/",
    ".agents/",
    ".claude/",
    ".llms/",
    "third_party/tlx/tools/agents/skills/",
)
IGNORED_SUFFIXES = (".md", ".rst")

TUTORIALS_ROOT = "third_party/tlx/tutorials/"
OPS_ROOT = "third_party/tlx/ops/"
OPS_KERNELS_ROOT = "third_party/tlx/ops/kernels/"
CATALOG_REL = "third_party/tlx/ops/_catalog.py"
CORRECTNESS_REL = "third_party/tlx/tutorials/testing/test_correctness.py"

TUTORIAL_IMPORT_ROOT = "triton.language.extra.tlx.tutorials"
OPS_IMPORT_ROOT = "triton.tlx.ops"
OPS_KERNELS_IMPORT_ROOT = "triton.tlx.ops.kernels"

# Test roots scanned for kernel imports. Mirrors what the GPU workflows run.
SCAN_TLX_OPS = "python/test/unit/tlx_ops"
SCAN_LANGUAGE = "python/test/unit/language"
SCAN_RUNTIME = "python/test/unit/runtime"
SCAN_GLUON = "python/test/gluon"
SCAN_TUTORIAL_TESTING = "third_party/tlx/tutorials/testing"

LANGUAGE_BASENAME = "test_tlx_*.py"


def normalize_changed(raw: str, repo: Path) -> str | None:
    """Normalize one changed-files entry to a repo-relative posix path."""
    p = raw.strip().replace("\\", "/")
    # Strip a literal "./" prefix only: lstrip("./") would eat the leading
    # dots of dot-directories (".agents/" -> "agents/") and of "../" escapes.
    p = p[2:] if p.startswith("./") else p
    if not p:
        return None
    # Absolute paths only make sense when they point inside the repo.
    if os.path.isabs(raw.strip()):
        try:
            rel = Path(raw.strip()).resolve().relative_to(repo.resolve())
        except ValueError:
            return None
        p = rel.as_posix()
    # Anything escaping the repo fails open to a full run.
    if p.startswith("../") or "/../" in p:
        return None
    return p


def classify(path: str) -> str:
    """Classify a repo-relative path as ignored/test/kernel/other."""
    if not path:
        return "other"
    for pre in IGNORED_PREFIXES:
        if path.startswith(pre):
            return "ignored"
    if path.endswith(IGNORED_SUFFIXES):
        return "ignored"
    for pre in TEST_PREFIXES:
        if path.startswith(pre):
            return "test"
    if path.startswith(TUTORIALS_ROOT):
        base = path.rsplit("/", 1)[-1]
        for pat in TUTORIAL_TEST_BASENAMES:
            if fnmatch.fnmatch(base, pat):
                return "test"
    for pre in KERNEL_PREFIXES:
        if path.startswith(pre):
            return "kernel"
    return "other"


def parse_catalog(repo: Path) -> tuple[dict[str, set[str]], bool]:
    """Map ops kernel dir name -> public op names from the dispatch catalog.

    Parsed from source (never imported: importing it needs torch). Returns
    (mapping, incomplete): the map is empty when the catalog cannot be parsed,
    and ``incomplete`` is true when any ``OpSpec`` entry was skipped for a
    reason other than pointing outside the ops kernels (e.g. a non-constant
    ``impl``). Callers treat an empty or incomplete catalog as "all ops tests
    affected" rather than failing the whole selection.
    """
    path = repo / CATALOG_REL
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}, True
    mapping: dict[str, set[str]] = {}
    incomplete = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "OpSpec"):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        op = kw.get("op")
        impl = kw.get("impl")
        if not (isinstance(op, ast.Constant) and isinstance(op.value, str)):
            incomplete = True
            continue
        if not (isinstance(impl, ast.Constant) and isinstance(impl.value, str)):
            incomplete = True
            continue
        module = impl.value.split(":", 1)[0]
        prefix = OPS_KERNELS_IMPORT_ROOT + "."
        if not module.startswith(prefix):
            # Deliberate: non-kernel impls (e.g. inductor torch fallbacks)
            # have no kernel dir to attribute, so they are not "missing".
            continue
        kernel_dir = module[len(prefix):].split(".", 1)[0]
        if kernel_dir:
            mapping.setdefault(kernel_dir, set()).add(op.value)
        else:
            incomplete = True
    return mapping, incomplete


class KernelMatchers:
    """What a kernel-only change touches, derived from changed file paths."""

    def __init__(self) -> None:
        self.prefixes: set[str] = set()
        self.op_names: set[str] = set()
        self.all_ops = False
        self.all_tutorials = False
        # Substrings for the TritonBench grep.
        self.tutorial_stems: set[str] = set()
        self.kernel_pkgs: set[str] = set()


def kernel_matchers_for(
    changed_kernels: list[str],
    catalog: dict[str, set[str]],
    catalog_incomplete: bool = False,
) -> KernelMatchers:
    m = KernelMatchers()
    for path in changed_kernels:
        if path.startswith(TUTORIALS_ROOT):
            rel = path[len(TUTORIALS_ROOT):]
            if not rel.endswith(".py"):
                m.all_tutorials = True
                continue
            stem = rel[:-len(".py")]
            parts = stem.split("/")
            if len(parts) == 1:
                m.prefixes.add(f"{TUTORIAL_IMPORT_ROOT}.{parts[0]}")
                m.tutorial_stems.add(parts[0])
            else:
                m.prefixes.add(f"{TUTORIAL_IMPORT_ROOT}.{parts[0]}")
                m.tutorial_stems.add(parts[0])
                m.kernel_pkgs.add("/".join(parts[:2]))
        elif path.startswith(OPS_KERNELS_ROOT):
            rel = path[len(OPS_KERNELS_ROOT):]
            kernel_dir = rel.split("/", 1)[0]
            if kernel_dir.endswith(".py"):
                kernel_dir = kernel_dir[:-len(".py")]
            m.prefixes.add(f"{OPS_KERNELS_IMPORT_ROOT}.{kernel_dir}")
            m.kernel_pkgs.add(f"kernels.{kernel_dir}")
            m.op_names.update(catalog.get(kernel_dir, set()))
        elif path.startswith(OPS_ROOT):
            # Dispatch surface (__init__.py, _catalog.py): every op may route
            # differently, so all ops tests are affected.
            m.all_ops = True
    if (not catalog or catalog_incomplete) and any(p.startswith(OPS_ROOT) for p in changed_kernels):
        # Unparseable or partial catalog: cannot resolve op names, keep the
        # kernel-only fast path for non-ops suites but run all ops tests.
        m.all_ops = True
    return m


def collect_imports(source: str) -> list[tuple[str | None, list[str]]]:
    """Collect (module, names) for every absolute import in the source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[tuple[str | None, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            out.append((node.module or "", [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, []))
    return out


def module_matches(module: str, prefixes: set[str]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def file_imports_ops_root(imports: list[tuple[str | None, list[str]]]) -> bool:
    for module, _names in imports:
        if module == OPS_IMPORT_ROOT or (module or "").startswith(OPS_IMPORT_ROOT + "."):
            return True
    return False


def test_file_affected(
    source: str,
    test_rel: str,
    matchers: KernelMatchers,
    op_pattern: re.Pattern[str] | None,
) -> bool:
    imports = collect_imports(source)
    if matchers.prefixes:
        for module, _names in imports:
            if module and module_matches(module, matchers.prefixes):
                return True
    if matchers.op_names:
        for module, names in imports:
            if module == OPS_IMPORT_ROOT and matchers.op_names.intersection(names):
                return True
        # Attribute-style use (ops.hstu_attn_dev) has no from-import name to
        # match; fall back to an identifier search. Comments can over-match,
        # which only ever runs extra tests.
        if op_pattern is not None and op_pattern.search(source):
            return True
    if matchers.all_ops:
        if test_rel.startswith(SCAN_TLX_OPS + "/") or file_imports_ops_root(imports):
            return True
    if matchers.all_tutorials:
        for module, _names in imports:
            if module == TUTORIAL_IMPORT_ROOT or (module or "").startswith(TUTORIAL_IMPORT_ROOT + "."):
                return True
    return False


def iter_scan_files(repo: Path) -> list[str]:
    """List candidate test files (repo-relative) across the scanned suites."""
    found: list[str] = []

    def add(root: str, basename_glob: str | None = None) -> None:
        base = repo / root
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if basename_glob and not fnmatch.fnmatch(path.name, basename_glob):
                continue
            found.append(path.relative_to(repo).as_posix())

    add(SCAN_TLX_OPS)
    add(SCAN_LANGUAGE, LANGUAGE_BASENAME)
    add(SCAN_RUNTIME)
    add(SCAN_GLUON)
    add(SCAN_TUTORIAL_TESTING)
    tutorials = repo / TUTORIALS_ROOT.rstrip("/")
    if tutorials.is_dir():
        for path in sorted(tutorials.glob("*.py")):
            if any(fnmatch.fnmatch(path.name, pat) for pat in TUTORIAL_TEST_BASENAMES):
                found.append(path.relative_to(repo).as_posix())
    return found


def _bound_names(tree: ast.Module) -> dict[str, str]:
    """Map local names bound by from-imports to their source module."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        module = node.module or ""
        if not module:
            continue
        for a in node.names:
            if a.name == "*":
                continue
            bound[a.asname or a.name] = module
    return bound


def _loaded_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _ops_reaches_file(source: str, matchers: KernelMatchers) -> bool:
    """Whether an ops change reaches a file through an actual ops import."""
    if not (matchers.op_names or matchers.all_ops):
        return False
    for module, names in collect_imports(source):
        if module == OPS_IMPORT_ROOT:
            if matchers.all_ops or matchers.op_names.intersection(names):
                return True
        elif module and module.startswith(OPS_IMPORT_ROOT + "."):
            return True
    return False


def correctness_selection(
    repo: Path,
    matchers: KernelMatchers,
    changed_tests: set[str],
) -> tuple[str, str]:
    """Decide how to run test_correctness.py: ("all"|"skip"|"filtered", k_expr)."""
    if CORRECTNESS_REL in changed_tests or matchers.all_tutorials:
        return ("all", "")
    path = repo / CORRECTNESS_REL
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return ("all", "")
    if _ops_reaches_file(source, matchers):
        # Ops usage cannot be attributed to one test (public-op imports carry
        # no kernel path), so run the whole file. Unreachable today: this file
        # imports no ops.
        return ("all", "")
    if not matchers.prefixes:
        return ("skip", "")
    # A plain `import` of an affected tutorial module cannot be attributed to
    # one test without usage analysis; run the whole file instead.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if module_matches(a.name, matchers.prefixes):
                    return ("all", "")
    bound = _bound_names(tree)

    # A kernel reference often sits one or more hops away from the test that
    # exercises it (test -> _run_blackwell_fa_numeric -> _blackwell_fa_fwd_ws,
    # or test -> Mxfp8Gemm.run_test -> _blackwell_gemm_ws_mxfp8), so resolve
    # module-level names transitively to a fixpoint.
    node_refs: dict[str, set[str]] = {}
    tests: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node_refs[node.name] = _loaded_names(node)
        elif isinstance(node, ast.FunctionDef):
            node_refs[node.name] = _loaded_names(node)
            if node.name.startswith("test_"):
                tests.append(node.name)

    def direct_modules(name: str) -> set[str]:
        return {
            mod
            for mod in (bound[r] for r in node_refs[name] if r in bound)
            if module_matches(mod, matchers.prefixes)
        }

    resolved: dict[str, set[str]] = {name: direct_modules(name) for name in node_refs}
    changed = True
    while changed:
        changed = False
        for name, refs in node_refs.items():
            for ref in refs:
                if ref in resolved and ref != name:
                    before = len(resolved[name])
                    resolved[name] |= resolved[ref]
                    if len(resolved[name]) != before:
                        changed = True
    selected = [t for t in tests if resolved[t]]
    if not selected:
        return ("skip", "")
    return ("filtered", " or ".join(selected))


def tritonbench_roots(repo: Path, extra: list[str]) -> list[Path]:
    candidates = list(extra)
    env_root = os.environ.get("TRITONBENCH_ROOT")
    if env_root:
        candidates.append(env_root)
    gh_workspace = os.environ.get("GITHUB_WORKSPACE")
    if gh_workspace:
        candidates.append(str(Path(gh_workspace) / "tritonbench"))
    candidates.append("/workspace/tritonbench")
    roots: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        p = Path(c)
        if not p.is_absolute():
            p = repo / c
        key = p.as_posix()
        if key not in seen and p.is_dir():
            seen.add(key)
            roots.append(p)
    return roots


def tritonbench_affected(
    matchers: KernelMatchers,
    roots: list[Path],
) -> bool:
    """Whether any TritonBench source references the changed kernels."""
    if matchers.all_ops or matchers.all_tutorials:
        return True
    needles = set(matchers.tutorial_stems) | set(matchers.op_names) | set(matchers.kernel_pkgs)
    needles = {n for n in needles if n}
    if not needles:
        return False
    if not roots:
        # No checkout to scan: fail open.
        return True
    # Build-output dirs are intentionally not listed by name: the scan roots
    # are plain source checkouts, and any extra .py files found only ever run
    # additional tests (fail-open), never skip needed ones.
    skip_dirs = {".git", "__pycache__", ".venv", "node_modules"}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                try:
                    with open(os.path.join(dirpath, fn), encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                if any(n in text for n in needles):
                    return True
    return False


def write_github_output(lines: list[str]) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    with open(out_path, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-files", required=True, help="File with repo-relative changed paths, one per line")
    parser.add_argument("--affected-out", default=None, help="Write affected test files here, one per line")
    parser.add_argument("--tritonbench", action="append", default=[], help="Extra TritonBench root to scan")
    parser.add_argument("--repo", default=str(REPO_ROOT), help="Triton repo root")
    args = parser.parse_args()

    repo = Path(args.repo)

    def full_run(reason: str) -> int:
        print(f"{reason}; running everything", file=sys.stderr)
        if args.affected_out:
            Path(args.affected_out).write_text("", encoding="utf-8")
        write_github_output(["kernel_only=false", "correctness_mode=all", "correctness_k=", "run_tritonbench=true"])
        return 0

    try:
        raw_lines = Path(args.changed_files).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return full_run(f"cannot read changed files: {exc}")
    changed = [n for line in raw_lines if (n := normalize_changed(line, repo))]
    if not changed:
        return full_run("empty changed-files list")

    buckets: dict[str, list[str]] = {"ignored": [], "test": [], "kernel": [], "other": []}
    for path in changed:
        buckets[classify(path)].append(path)

    if buckets["other"]:
        for path in sorted(buckets["other"])[:20]:
            print(f"  non-kernel: {path}")
        return full_run(f"{len(buckets['other'])} non-kernel file(s) changed")

    catalog, catalog_incomplete = parse_catalog(repo)
    matchers = kernel_matchers_for(buckets["kernel"], catalog, catalog_incomplete)
    op_pattern = (re.compile(r"\b(" + "|".join(sorted(re.escape(o) for o in matchers.op_names)) +
                             r")\b") if matchers.op_names else None)

    affected: set[str] = set()
    for test_rel in iter_scan_files(repo):
        if test_rel in buckets["test"]:
            affected.add(test_rel)
            continue
        try:
            source = (repo / test_rel).read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            ast.parse(source)
        except SyntaxError:
            # Unparseable candidate: run it so the failure is loud.
            affected.add(test_rel)
            continue
        if test_file_affected(source, test_rel, matchers, op_pattern):
            affected.add(test_rel)
    # Changed test files outside the scanned suites still count as affected so
    # suite-level "anything affected?" checks stay honest; no suite runs them.
    affected.update(buckets["test"])

    mode, k_expr = correctness_selection(repo, matchers, set(buckets["test"]))
    if mode == "skip":
        affected.discard(CORRECTNESS_REL)
    elif mode in ("all", "filtered"):
        affected.add(CORRECTNESS_REL)

    tb_roots = tritonbench_roots(repo, args.tritonbench)
    run_tb = tritonbench_affected(matchers, tb_roots)

    ordered = sorted(affected)
    if args.affected_out:
        Path(args.affected_out).write_text("".join(p + "\n" for p in ordered), encoding="utf-8")
    else:
        for p in ordered:
            print(p)
    write_github_output([
        "kernel_only=true",
        f"correctness_mode={mode}",
        f"correctness_k={k_expr}",
        f"run_tritonbench={'true' if run_tb else 'false'}",
    ])
    print(f"kernel-only change: {len(buckets['kernel'])} kernel file(s), "
          f"{len(buckets['test'])} test file(s), {len(buckets['ignored'])} ignored")
    print(f"affected test files: {len(ordered)}; correctness: {mode}; tritonbench: {'run' if run_tb else 'skip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
