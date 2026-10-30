"""Plan CPU and GPU test work before allocating runners."""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_FILTERS = REPO_ROOT / ".github/gpu-test-paths.json"
SUITES = REPO_ROOT / ".github/gpu-test-suites.json"
BUMP_PREFIX = "scratch/meta-llvm-bump/"
WORKFLOW = ".github/workflows/gpu-tests.yml"

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
    if path.startswith("python/test/tlx_benchmark/"):
        return "test"
    for pre in TEST_PREFIXES:
        if path.startswith(pre):
            name = path.rsplit("/", 1)[-1]
            if name == "conftest.py" or not (name.startswith("test_") or name.endswith("_test.py")
                                             or name.endswith(".mlir")):
                return "other"
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


def matches(path, pattern):
    # GitHub's * stays within a directory; ** crosses directories, and **/
    # also matches zero directories. These are the only wildcards in our rules.
    parts = []
    i = 0
    while i < len(pattern):
        if pattern[i:i + 3] == "**/":
            parts.append("(?:.*/)?")
            i += 3
        elif pattern[i:i + 2] == "**":
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.fullmatch("".join(parts), path) is not None


def select_platforms(paths, filters):
    selected = {}
    for platform, patterns in filters.items():
        selected[platform] = False
        for path in paths:
            included = False
            for pattern in patterns:
                if matches(path, pattern.removeprefix("!")):
                    included = not pattern.startswith("!")
            if included:
                selected[platform] = True
                break
    return selected


def changed_paths(event_name, event):
    if event_name == "pull_request":
        pr = event["pull_request"]
        revisions = [f"{pr['base']['sha']}...{pr['head']['sha']}"]
    elif event_name == "push" and event["before"].strip("0"):
        revisions = [event["before"], event["after"]]
    else:
        return None  # Schedules, manual runs, and new branches run all suites.
    # Disable rename detection so both old and new paths participate in filters.
    result = subprocess.run(["git", "diff", "--name-only", "--no-renames", "-z", *revisions, "--"], check=True,
                            capture_output=True)
    return result.stdout.decode().rstrip("\0").split("\0") if result.stdout else []


def bump_ref(event_name, event):
    """The Meta LLVM bump branch this event is about, or None."""
    if event_name == "push" and event.get("ref", "").startswith("refs/heads/" + BUMP_PREFIX):
        return event["ref"].removeprefix("refs/heads/")
    pr = event.get("pull_request") if event_name == "pull_request" else None
    if pr and pr["head"]["ref"].startswith(
            BUMP_PREFIX) and pr["head"]["repo"]["full_name"] == pr["base"]["repo"]["full_name"]:
        return pr["head"]["ref"]
    return None


def passed_push_run(runs):
    """URL of a successful push-triggered run of this workflow among runs, or None.

    The bot never pushes to a bump branch once its PR is open, so the PR head is the commit that run tested.
    """
    for run in runs:
        if run["event"] == "push" and run["path"] == WORKFLOW and run["conclusion"] == "success":
            return run["html_url"]
    return None


def fetch_runs(repo, sha):
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/runs?head_sha={sha}&event=push",
                                     headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["workflow_runs"]


def select_kernel_tests(paths, repo, tritonbench=()):
    full = {
        "kernel_only": False, "affected": [], "correctness_mode": "all", "correctness_k": "", "run_tritonbench": True
    }
    if paths is None or not paths:
        return full
    normalized = [normalize_changed(path, repo) for path in paths]
    if any(path is None for path in normalized):
        return full
    buckets = {kind: [p for p in normalized if classify(p) == kind] for kind in ("ignored", "test", "kernel", "other")}
    if buckets["other"]:
        return full
    catalog, incomplete = parse_catalog(repo)
    matchers = kernel_matchers_for(buckets["kernel"], catalog, incomplete)
    op_pattern = re.compile(r"\b(" + "|".join(re.escape(op) for op in sorted(matchers.op_names)) +
                            r")\b") if matchers.op_names else None
    affected = set(buckets["test"])
    for rel in iter_scan_files(repo):
        source = (repo / rel).read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError:
            affected.add(rel)
            continue
        if test_file_affected(source, rel, matchers, op_pattern):
            affected.add(rel)
    mode, expression = correctness_selection(repo, matchers, set(buckets["test"]))
    if mode == "skip":
        affected.discard(CORRECTNESS_REL)
    else:
        affected.add(CORRECTNESS_REL)
    return {
        "kernel_only": True, "affected": sorted(affected), "correctness_mode": mode, "correctness_k": expression,
        "run_tritonbench": tritonbench_affected(matchers, tritonbench_roots(repo, list(tritonbench)))
    }


def cpu_eligible(paths, repo, spec):
    if paths is None:
        paths = []
    if any(p.startswith(tuple(spec["dependencies"])) or p in spec["files"] or p.endswith(".py") for p in paths):
        return False
    if "gluon_skip_files" not in spec:
        return True
    directory = repo / "python/test/gluon"
    actual = {
        str(p.relative_to(repo))
        for p in directory.rglob("*.py")
        if p.name.startswith("test_") or p.name.endswith("_test.py")
    }
    expected = set(spec["files"]) | {"python/test/gluon/" + name for name in spec["gluon_skip_files"]}
    if actual != expected:
        return False
    tree = ast.parse((directory / "conftest.py").read_text())
    settings = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("_VERSION_SKEW_FILES", "_KNOWN_FAIL_SUBSTRINGS")
    }
    ignores = [
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "collect_ignore" for t in n.targets)
    ]
    references = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "collect_ignore"]
    if len(ignores) != 1 or len(references) != 1:
        return False
    if ast.dump(ignores[0]) != ast.dump(ast.parse("sorted(_VERSION_SKEW_FILES)", mode="eval").body):
        return False
    return (settings.get("_VERSION_SKEW_FILES") == set(spec["gluon_skip_files"])
            and settings.get("_KNOWN_FAIL_SUBSTRINGS") == tuple(spec["gluon_skip_tests"]))


def suite_files(repo, patterns):
    files = sorted({
        str(p.relative_to(repo))
        for pattern in patterns
        for p in repo.glob(pattern)
        if (p.is_file() or p.is_dir()) and not p.name.startswith("test_torchtlx_")
    })
    if not files:
        raise ValueError(f"No tests match suite patterns: {patterns}")
    return files


def reviewed_changes(repo, revision, paths):
    try:
        if subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=repo, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode:
            subprocess.run(["git", "fetch", "--no-tags", "origin", revision], cwd=repo, check=True, timeout=60)
        return subprocess.check_output(["git", "diff", "--name-only", "--no-renames", revision, "--", *paths], cwd=repo,
                                       text=True).splitlines()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def test_plan(paths, repo, filters, suites, narrow=True, offload=True):
    platforms = dict.fromkeys(filters, True) if paths is None else select_platforms(paths, filters)
    kernel = select_kernel_tests(paths, repo) if narrow else select_kernel_tests(None, repo)
    affected = set(kernel["affected"])
    gpu, cpu = {}, {}
    cpu_ready = {
        group:
        offload and cpu_eligible(paths, repo, spec)
        and reviewed_changes(repo, suites["reviewed_revision"], spec["dependencies"] + spec["files"]) == []
        for group, spec in suites["cpu"].items()
    }
    for platform, enabled in platforms.items():
        plan = {
            "core": {"files": [], "k": ""}, "correctness": {"files": [], "k": ""}, "codegen": {"files": [], "k": ""},
            "runtime": {"files": [], "k": ""}, "gluon": {"files": [], "k": ""}
        }
        if enabled:
            core = suite_files(repo, suites["platforms"][platform]["core"])
            plan["core"]["files"] = [p for p in core if p in affected] if kernel["kernel_only"] else core
            if kernel["correctness_mode"] != "skip":
                plan["correctness"] = {"files": [CORRECTNESS_REL], "k": kernel["correctness_k"]}
            group = "runtime" if platform == "h100" else "gluon"
            files = suite_files(repo, suites["platforms"][platform][group])
            # These groups historically run their full suite when any file is affected.
            if not kernel["kernel_only"] or any(p.startswith(files[0].rstrip("/") + "/") for p in affected):
                plan[group]["files"] = files
            if platform == "b200" and (not kernel["kernel_only"] or suites["cpu"]["codegen"]["files"][0] in affected):
                plan["codegen"]["files"] = suites["cpu"]["codegen"]["files"]
        for group in ("codegen", "gluon"):
            if cpu_ready[group] and plan[group]["files"]:
                cpu[group] = {
                    "files": suites["cpu"][group]["files"], "k": "", "allowed_skips":
                    suites["cpu"][group]["allowed_skips"]
                }
                plan[group]["files"] = []
        gpu[platform] = plan
    return {
        "gpu": gpu, "cpu": cpu, "kernel": kernel, "build": bool(cpu)
        or any(entry["files"] for plan in gpu.values() for entry in plan.values())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-files")
    parser.add_argument("--affected-out")
    parser.add_argument("--tritonbench", action="append", default=[])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.changed_files:
        try:
            paths = Path(args.changed_files).read_text().splitlines()
        except OSError:
            paths = None
        kernel = select_kernel_tests(paths, repo, args.tritonbench)
        if args.affected_out:
            Path(args.affected_out).write_text("".join(p + "\n" for p in kernel["affected"]))
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key in ("kernel_only", "correctness_mode", "correctness_k", "run_tritonbench"):
                value = kernel[key]
                output.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
        return
    paths = None
    diff_known = True
    event_name = os.environ["GITHUB_EVENT_NAME"]
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        bump = bump_ref(event_name, event)
        if bump and event_name == "push":
            paths = None
        elif bump and passed_push_run(fetch_runs(os.environ["GITHUB_REPOSITORY"],
                                                 event["pull_request"]["head"]["sha"])):
            paths = []
        else:
            paths = changed_paths(event_name, event)
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"::warning::Cannot determine changed paths; running full suites: {exc}")
        diff_known = False
    filters = json.loads(PATH_FILTERS.read_text())
    suites = json.loads(SUITES.read_text())
    try:
        plan = test_plan(paths, repo, filters, suites, narrow=event_name == "pull_request", offload=diff_known)
    except (ValueError, OSError, SyntaxError) as exc:
        print(f"::warning::Cannot select tests; running full GPU suites: {exc}")
        # Configuration or test enumeration failures must not produce an empty success.
        plan = test_plan(None, repo, filters, suites, narrow=False, offload=False)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(json.dumps(plan, indent=2))
    cpu_plan = {group: plan["cpu"].get(group, {"files": [], "k": ""}) for group in ("codegen", "gluon")}
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"sha={sha}\nbuild={str(plan['build']).lower()}\n")
        output.write(f"cpu={str(bool(plan['cpu'])).lower()}\ncpu_plan={json.dumps(cpu_plan, separators=(',', ':'))}\n")
        for platform, steps in plan["gpu"].items():
            output.write(f"{platform}={str(any(s['files'] for s in steps.values())).lower()}\n")
            output.write(f"{platform}_plan={json.dumps(steps, separators=(',', ':'))}\n")


if __name__ == "__main__":
    main()
