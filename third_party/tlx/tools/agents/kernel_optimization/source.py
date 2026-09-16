from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CODE_BLOCK_RE = re.compile(r"```(?P<language>[A-Za-z0-9_+-]*)\s*\n(?P<code>.*?)```", re.DOTALL)
_DIFF_HEADER_RE = re.compile(r"(?m)^--- candidate\.py\n\+\+\+ candidate\.py\n")
_PROTON_ALIAS = "pl"
_PROTON_HELPER_PREFIX = "_tlx_agent_proton_"
INSTRUMENTATION_MAPPING_SCHEMA_VERSION = 1
_MAX_INSTRUMENTATION_PASSES = 8
_MAX_INSTRUMENTATION_NAMES = 128
_MAX_INSTRUMENTATION_TEXT = 256
_DEFAULT_DIAGNOSTIC_PASSES = frozenset({"role", "coarse", "wait", "compute"})
_INSTRUMENTATION_MAPPING_KEYS = frozenset(
    {
        "schema_version",
        "diagnostic_only",
        "instrumentation",
        "expected_kernel",
        "selected_cta",
        "tasks",
        "required_scopes",
        "scope_kinds",
        "wait_scopes",
        "passes",
        "limitations",
    }
)
_INSTRUMENTATION_CONFIG_KEYS = frozenset(
    {"backend", "data", "granularity", "triton_semantic"}
)
_PASS_RECORD_KEYS = frozenset({"expected_regions", "scope_names"})


def extract_python_source(output: str) -> str:
    blocks = list(_CODE_BLOCK_RE.finditer(output))
    python_blocks = [
        match.group("code")
        for match in blocks
        if match.group("language").lower() in {"python", "py"}
    ]
    generic_blocks = [
        match.group("code") for match in blocks if not match.group("language")
    ]
    candidates = python_blocks or generic_blocks or [output]
    valid_sources: list[str] = []
    last_error: ValueError | None = None
    for candidate in candidates:
        source = canonicalize_source(candidate)
        try:
            validate_kernel_source(source)
        except ValueError as error:
            last_error = error
            continue
        valid_sources.append(source)
    if valid_sources:
        return max(valid_sources, key=len)
    assert last_error is not None
    raise last_error


def apply_candidate_diff(output: str, current_source: str) -> str:
    """Apply a model-produced unified diff to an isolated source copy."""
    blocks = list(_CODE_BLOCK_RE.finditer(output))
    diff_blocks = [
        match.group("code")
        for match in blocks
        if match.group("language").lower() in {"diff", "patch"}
    ]
    candidates = diff_blocks or [output]
    patches = [candidate.strip() + "\n" for candidate in candidates if _DIFF_HEADER_RE.search(candidate)]
    if not patches:
        raise ValueError("candidate response does not contain a candidate.py unified diff")
    patch = max(patches, key=len)
    with tempfile.TemporaryDirectory(prefix="tlx-agent-patch-") as directory:
        root = Path(directory)
        candidate_path = root / "candidate.py"
        patch_path = root / "candidate.patch"
        candidate_path.write_text(current_source)
        patch_path.write_text(patch)
        completed = subprocess.run(
            ["patch", "--batch", "--forward", "--silent", "-p0", "-i", str(patch_path)],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            diagnostics = (completed.stderr or completed.stdout).strip()
            raise ValueError(f"candidate diff did not apply cleanly: {diagnostics}")
        source = canonicalize_source(candidate_path.read_text())
    validate_replacement_source(source, current_source)
    return source


def validate_kernel_source(source: str) -> None:
    """Raise ValueError if source is empty or not valid Python."""
    if not source.strip():
        raise ValueError("candidate source is empty")
    try:
        ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"candidate is not valid Python: {error}") from error


def validate_replacement_source(candidate: str, current: str) -> None:
    """Reject responses that are valid Python but not a complete replacement."""
    validate_kernel_source(candidate)
    current_tree = ast.parse(current)
    candidate_tree = ast.parse(candidate)
    node_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    required_names = {node.name for node in current_tree.body if isinstance(node, node_types)}
    candidate_names = {
        node.name for node in candidate_tree.body if isinstance(node, node_types)
    }
    missing_names = sorted(required_names - candidate_names)
    if missing_names:
        preview = ", ".join(missing_names[:8])
        suffix = "..." if len(missing_names) > 8 else ""
        raise ValueError(
            f"candidate is not a complete replacement; missing top-level symbols: "
            f"{preview}{suffix}"
        )
    if len(candidate) < len(current) * 0.8:
        raise ValueError(
            "candidate is not a complete replacement; source is unexpectedly short "
            f"({len(candidate)} bytes versus {len(current)} bytes)"
        )


def validate_candidate_source(candidate: str, current: str) -> None:
    validate_replacement_source(candidate, current)
    tree = ast.parse(candidate)
    for node in ast.walk(tree):
        if _is_any_proton_import(node):
            raise ValueError(
                "candidate source contains diagnostic Proton instrumentation"
            )
        if isinstance(node, ast.Call) and _is_proton_api_call(node):
            raise ValueError(
                "candidate source contains diagnostic Proton instrumentation"
            )
        if isinstance(node, ast.Name) and node.id.startswith(_PROTON_HELPER_PREFIX):
            raise ValueError(
                "candidate source contains diagnostic Proton instrumentation"
            )


def _is_any_proton_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(
            alias.name.startswith("triton.profiler")
            or "proton" in alias.name.split(".")
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module.startswith("triton.profiler") or "proton" in module.split(".")
    return False


def _is_proton_api_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"pl", "proton"}
        and node.func.attr
        in {"enable_semantic", "enter_scope", "exit_scope", "record"}
    )


def canonicalize_diagnostic_task_warps(
    instrumented: str,
    current: str,
    mapping: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Replace model-provided task ordinals with provable physical warp IDs."""
    canonical = copy.deepcopy(dict(mapping))
    tasks = canonical.get("tasks")
    if not isinstance(tasks, Mapping):
        return canonical
    try:
        instrumented_tree = ast.parse(instrumented)
        current_tree = ast.parse(current)
    except SyntaxError:
        return canonical
    inferred = _infer_tlx_scope_warps(
        instrumented_tree,
        current_tree,
        environment or {},
    )
    canonical_tasks: dict[Any, Any] = {}
    for raw_name, raw_task in tasks.items():
        if not isinstance(raw_task, Mapping):
            canonical_tasks[raw_name] = raw_task
            continue
        task = dict(raw_task)
        scope = task.get("scope")
        if isinstance(scope, str) and scope in inferred:
            task["warps"] = list(inferred[scope])
        canonical_tasks[raw_name] = task
    canonical["tasks"] = canonical_tasks
    return canonical


def validate_diagnostic_instrumentation_source(
    instrumented: str,
    current: str,
    mapping: Mapping[str, Any],
    *,
    pass_capabilities: Mapping[str, tuple[str, ...]] | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Validate source-only Proton instrumentation against the original source.

    The validator is intentionally conservative: it strips only a tiny allowlist of
    Proton instrumentation syntax, then requires the remaining AST to match the
    original. Safe equivalent rewrites outside that allowlist may be rejected.
    """
    validate_replacement_source(instrumented, current)
    current_tree = ast.parse(current)
    instrumented_tree = ast.parse(instrumented)
    _validate_top_level_signatures(current_tree, instrumented_tree)
    _validate_instrumentation_mapping(mapping, pass_capabilities or {})
    scope_calls = _collect_proton_scope_calls(instrumented_tree)
    _validate_proton_setup(instrumented_tree)
    if not scope_calls:
        raise ValueError("instrumented source must contain Proton scope calls")
    _validate_scope_balance(instrumented_tree)
    _validate_mapping_references(mapping, {name for _, name, _ in scope_calls})
    _validate_static_tlx_task_warps(
        instrumented_tree,
        current_tree,
        mapping,
        environment or {},
    )
    _validate_only_allowed_instrumentation(current_tree, instrumented_tree)


def canonicalize_source(source: str) -> str:
    return source.strip() + "\n"


def source_diff(
    before: str,
    after: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    return "".join(
        difflib.unified_diff(
            canonicalize_source(before).splitlines(keepends=True),
            canonicalize_source(after).splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def source_digest(source: str) -> str:
    return hashlib.sha256(canonicalize_source(source).encode()).hexdigest()


def _validate_top_level_signatures(
    current_tree: ast.Module,
    instrumented_tree: ast.Module,
) -> None:
    current = _top_level_signatures(current_tree)
    instrumented = _top_level_signatures(instrumented_tree)
    changed = sorted(
        name
        for name, signature in current.items()
        if instrumented.get(name) != signature
    )
    if changed:
        preview = ", ".join(changed[:8])
        suffix = "..." if len(changed) > 8 else ""
        raise ValueError(f"instrumented source changed top-level signatures: {preview}{suffix}")


def _top_level_signatures(tree: ast.Module) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures[node.name] = "|".join(
                (
                    ast.dump(node.args, include_attributes=False),
                    _dump_optional_ast(node.returns),
                )
            )
        elif isinstance(node, ast.ClassDef):
            signatures[node.name] = "|".join(
                (
                    ",".join(ast.dump(base, include_attributes=False) for base in node.bases),
                    ",".join(
                        ast.dump(keyword, include_attributes=False)
                        for keyword in node.keywords
                    ),
                )
            )
    return signatures


def _dump_optional_ast(node: ast.AST | None) -> str:
    return "" if node is None else ast.dump(node, include_attributes=False)


@dataclass(frozen=True)
class _TlxAsyncTask:
    node: ast.With
    is_default: bool
    num_warps: int
    replicate: int
    warp_group_start_id: int | None


def _infer_tlx_scope_warps(
    instrumented_tree: ast.Module,
    current_tree: ast.Module,
    environment: Mapping[str, str],
) -> dict[str, tuple[int, ...]]:
    default_num_warps = _unique_literal_launch_num_warps(
        current_tree,
        environment,
    )
    if default_num_warps is None:
        return {}
    candidates: dict[str, set[tuple[int, ...]]] = {}
    for node in ast.walk(instrumented_tree):
        if not isinstance(node, ast.With) or not _is_tlx_async_tasks_with(node):
            continue
        tasks = _parse_tlx_async_task_block(node, default_num_warps)
        if tasks is None:
            continue
        for task, warps in tasks:
            for scope in _literal_proton_scopes(task.node):
                candidates.setdefault(scope, set()).add(warps)
    return {
        scope: next(iter(warp_sets))
        for scope, warp_sets in candidates.items()
        if len(warp_sets) == 1
    }


def _unique_literal_launch_num_warps(
    tree: ast.Module,
    environment: Mapping[str, str],
) -> int | None:
    constants = _module_string_constants(tree, environment)
    collector = _ReachableTritonConfigCollector(constants, environment)
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            collector.visit_statements(statement.body)
        elif not isinstance(statement, ast.ClassDef):
            collector.visit(statement)
    if not collector.complete or len(collector.num_warps) != 1:
        return None
    return next(iter(collector.num_warps))


def _module_string_constants(
    tree: ast.Module,
    environment: Mapping[str, str],
) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _static_string_value(statement.value, constants, environment)
        if value is not None:
            constants[target.id] = value
    return constants


def _static_string_value(
    node: ast.AST,
    constants: Mapping[str, str],
    environment: Mapping[str, str],
) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if not isinstance(node, ast.Call) or len(node.args) not in {1, 2}:
        return None
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "environ"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "os"
    ):
        return None
    key = _static_string_value(node.args[0], constants, environment)
    default = (
        _static_string_value(node.args[1], constants, environment)
        if len(node.args) == 2
        else None
    )
    if key is None:
        return None
    return environment.get(key, default)


class _ReachableTritonConfigCollector(ast.NodeVisitor):
    def __init__(
        self,
        constants: Mapping[str, str],
        environment: Mapping[str, str],
    ) -> None:
        self.constants = constants
        self.environment = environment
        self.num_warps: set[int] = set()
        self.complete = True

    def visit_statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        condition = _static_boolean_value(
            node.test,
            self.constants,
            self.environment,
        )
        if condition is True:
            self.visit_statements(node.body)
        elif condition is False:
            self.visit_statements(node.orelse)
        else:
            self.visit_statements(node.body)
            self.visit_statements(node.orelse)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_triton_config_call(node) or _is_kernel_launch_call(node):
            value = _literal_positive_int_keyword(node, "num_warps")
            if value is None:
                self.complete = False
            else:
                self.num_warps.add(value)
        self.generic_visit(node)


def _is_kernel_launch_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Subscript) and any(
        keyword.arg == "num_warps" for keyword in node.keywords
    )


def _static_boolean_value(
    node: ast.AST,
    constants: Mapping[str, str],
    environment: Mapping[str, str],
) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    left = _static_string_value(node.left, constants, environment)
    right = _static_string_value(node.comparators[0], constants, environment)
    if left is None or right is None:
        return None
    operator = node.ops[0]
    if isinstance(operator, ast.Eq):
        return left == right
    if isinstance(operator, ast.NotEq):
        return left != right
    return None


def _is_triton_config_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "triton"
        and node.func.attr == "Config"
    )


def _is_tlx_async_tasks_with(node: ast.With) -> bool:
    return len(node.items) == 1 and _is_tlx_call(
        node.items[0].context_expr,
        "async_tasks",
    )


def _is_tlx_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tlx"
        and node.func.attr == name
    )


def _parse_tlx_async_task_block(
    node: ast.With,
    default_num_warps: int,
) -> tuple[tuple[_TlxAsyncTask, tuple[int, ...]], ...] | None:
    tasks: list[_TlxAsyncTask] = []
    for statement in node.body:
        if not isinstance(statement, ast.With) or len(statement.items) != 1:
            return None
        task = _parse_tlx_async_task(statement, default_num_warps)
        if task is None:
            return None
        tasks.append(task)
    if sum(task.is_default for task in tasks) != 1:
        return None
    explicit_starts = [
        task.warp_group_start_id for task in tasks if not task.is_default
    ]
    if any(start is not None for start in explicit_starts):
        if any(start is None for start in explicit_starts):
            return None
        if any(task.is_default and task.replicate != 1 for task in tasks):
            return None
        return tuple(
            (task, _explicit_task_warps(task, default_num_warps))
            for task in tasks
        )
    cursor = default_num_warps
    result: list[tuple[_TlxAsyncTask, tuple[int, ...]]] = []
    for task in tasks:
        if task.is_default:
            warps = list(range(default_num_warps))
            extra_warps = default_num_warps * (task.replicate - 1)
            warps.extend(range(cursor, cursor + extra_warps))
            cursor += extra_warps
        else:
            task_warps = task.num_warps * task.replicate
            warps = list(range(cursor, cursor + task_warps))
            cursor += task_warps
        result.append((task, tuple(warps)))
    return tuple(result)


def _parse_tlx_async_task(
    node: ast.With,
    default_num_warps: int,
) -> _TlxAsyncTask | None:
    call = node.items[0].context_expr
    if not _is_tlx_call(call, "async_task"):
        return None
    assert isinstance(call, ast.Call)
    is_default = (
        len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "default"
    )
    if call.args and not is_default:
        return None
    num_warps = (
        default_num_warps
        if is_default
        else _literal_positive_int_keyword(call, "num_warps")
    )
    replicate = _literal_positive_int_keyword(call, "replicate", default=1)
    has_start_id = any(
        keyword.arg == "warp_group_start_id" for keyword in call.keywords
    )
    start_id = _literal_nonnegative_int_keyword(call, "warp_group_start_id")
    if (
        num_warps is None
        or replicate is None
        or (has_start_id and start_id is None)
    ):
        return None
    return _TlxAsyncTask(
        node=node,
        is_default=is_default,
        num_warps=num_warps,
        replicate=replicate,
        warp_group_start_id=start_id,
    )


def _explicit_task_warps(
    task: _TlxAsyncTask,
    default_num_warps: int,
) -> tuple[int, ...]:
    if task.is_default:
        return tuple(range(default_num_warps))
    assert task.warp_group_start_id is not None
    return tuple(
        range(
            task.warp_group_start_id,
            task.warp_group_start_id + task.num_warps * task.replicate,
        )
    )


def _literal_positive_int_keyword(
    call: ast.Call,
    name: str,
    *,
    default: int | None = None,
) -> int | None:
    value = _literal_int_keyword(call, name)
    if value is None:
        return default
    return value if value > 0 else None


def _literal_nonnegative_int_keyword(
    call: ast.Call,
    name: str,
) -> int | None:
    value = _literal_int_keyword(call, name)
    if value is None:
        return None
    return value if value >= 0 else None


def _literal_int_keyword(call: ast.Call, name: str) -> int | None:
    values = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(values) != 1:
        return None
    value = values[0]
    if not isinstance(value, ast.Constant) or isinstance(value.value, bool):
        return None
    return value.value if isinstance(value.value, int) else None


def _literal_proton_scopes(node: ast.AST) -> set[str]:
    scopes: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not _is_proton_scope_call(child):
            continue
        scope = _literal_scope_name(child)
        if scope is not None:
            scopes.add(scope)
    return scopes


def _validate_static_tlx_task_warps(
    instrumented_tree: ast.Module,
    current_tree: ast.Module,
    mapping: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    inferred = _infer_tlx_scope_warps(
        instrumented_tree,
        current_tree,
        environment,
    )
    tasks = mapping.get("tasks")
    if not inferred or not isinstance(tasks, Mapping):
        return
    for raw_name, raw_task in tasks.items():
        if not isinstance(raw_task, Mapping):
            continue
        scope = raw_task.get("scope")
        if not isinstance(scope, str) or scope not in inferred:
            continue
        actual = tuple(sorted(_parse_warp_list(raw_task.get("warps"), str(raw_name))))
        expected = inferred[scope]
        if actual != expected:
            raise ValueError(
                f"task {str(raw_name)!r} scope {scope!r} maps physical warps "
                f"{list(actual)}, expected {list(expected)} from TLX async-task topology"
            )


def _validate_instrumentation_mapping(
    mapping: Mapping[str, Any],
    pass_capabilities: Mapping[str, tuple[str, ...]],
) -> None:
    unknown_keys = sorted(set(mapping) - _INSTRUMENTATION_MAPPING_KEYS)
    if unknown_keys:
        raise ValueError("instrumentation mapping contains unknown keys: " + ", ".join(unknown_keys))
    if mapping.get("schema_version") != INSTRUMENTATION_MAPPING_SCHEMA_VERSION:
        raise ValueError(
            "instrumentation mapping schema_version must be "
            f"{INSTRUMENTATION_MAPPING_SCHEMA_VERSION}"
        )
    if mapping.get("diagnostic_only") is not True:
        raise ValueError("instrumentation mapping must set diagnostic_only=true")
    _validate_instrumentation_config(mapping.get("instrumentation"))
    _bounded_name(mapping.get("expected_kernel"), "expected_kernel")
    try:
        re.compile(str(mapping.get("expected_kernel")))
    except re.error as error:
        raise ValueError(f"expected_kernel must be a valid regex: {error}") from error
    selected_cta = mapping.get("selected_cta")
    if (
        isinstance(selected_cta, bool)
        or not isinstance(selected_cta, int)
        or selected_cta < 0
    ):
        raise ValueError("selected_cta must be a nonnegative integer")
    _validate_tasks(mapping.get("tasks"))
    _bounded_name_list(mapping.get("required_scopes"), "required_scopes")
    _validate_scope_kinds(mapping)
    _validate_pass_records(mapping.get("passes"), pass_capabilities)
    limitations = mapping.get("limitations", [])
    if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
        raise ValueError("limitations must be a string array")
    if len(limitations) > _MAX_INSTRUMENTATION_NAMES:
        raise ValueError("limitations contains too many entries")
    for item in limitations:
        _bounded_name(item, "limitations entry")


def _validate_instrumentation_config(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("instrumentation must be a JSON object")
    unknown_keys = sorted(set(value) - _INSTRUMENTATION_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError("instrumentation contains unknown keys: " + ", ".join(unknown_keys))
    expected = {
        "backend": "instrumentation",
        "data": "trace",
        "granularity": "warp",
        "triton_semantic": True,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"instrumentation.{key} must be {expected_value!r}")


def _validate_tasks(value: Any) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("tasks must be a nonempty object")
    if len(value) > _MAX_INSTRUMENTATION_NAMES:
        raise ValueError("tasks contains too many entries")
    owned_warps: set[int] = set()
    owned_scopes: set[str] = set()
    for raw_name, raw_task in value.items():
        name = _bounded_name(raw_name, "task name")
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"task {name!r} must be an object")
        unknown_keys = sorted(set(raw_task) - {"scope", "warps"})
        if unknown_keys:
            raise ValueError(f"task {name!r} contains unknown keys: " + ", ".join(unknown_keys))
        scope = _bounded_name(raw_task.get("scope"), f"task {name!r} scope")
        if scope in owned_scopes:
            raise ValueError(f"task scope {scope!r} has more than one owner")
        warps = _parse_warp_list(raw_task.get("warps"), name)
        overlap = owned_warps.intersection(warps)
        if overlap:
            raise ValueError(f"task {name!r} reuses owned warps {sorted(overlap)}")
        owned_scopes.add(scope)
        owned_warps.update(warps)


def _validate_scope_kinds(mapping: Mapping[str, Any]) -> None:
    has_scope_kinds = "scope_kinds" in mapping
    has_wait_scopes = "wait_scopes" in mapping
    if has_scope_kinds == has_wait_scopes:
        raise ValueError("provide exactly one of scope_kinds or wait_scopes")
    if has_wait_scopes:
        _bounded_name_list(mapping.get("wait_scopes"), "wait_scopes")
        return
    scope_kinds = mapping.get("scope_kinds")
    if not isinstance(scope_kinds, Mapping):
        raise ValueError("scope_kinds must be an object")
    if len(scope_kinds) > _MAX_INSTRUMENTATION_NAMES:
        raise ValueError("scope_kinds contains too many entries")
    for raw_scope, raw_kind in scope_kinds.items():
        _bounded_name(raw_scope, "scope_kinds scope")
        _bounded_name(raw_kind, "scope kind")


def _validate_pass_records(
    value: Any,
    pass_capabilities: Mapping[str, tuple[str, ...]],
) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("passes must be a nonempty object")
    if len(value) > _MAX_INSTRUMENTATION_PASSES:
        raise ValueError("passes contains too many entries")
    allowed_passes = set(pass_capabilities) or set(_DEFAULT_DIAGNOSTIC_PASSES)
    for raw_pass_name, raw_record in value.items():
        pass_name = _bounded_name(raw_pass_name, "pass name")
        if pass_name not in allowed_passes:
            raise ValueError(f"unsupported diagnostic pass {pass_name!r}")
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"pass {pass_name!r} must be an object")
        unknown_keys = sorted(set(raw_record) - _PASS_RECORD_KEYS)
        if unknown_keys:
            raise ValueError(f"pass {pass_name!r} contains unknown keys: " + ", ".join(unknown_keys))
        expected_regions = _bounded_name_list(
            raw_record.get("expected_regions"),
            f"pass {pass_name!r} expected_regions",
        )
        if pass_capabilities:
            allowed_regions = set(pass_capabilities.get(pass_name, ()))
            unsupported = sorted(set(expected_regions) - allowed_regions)
            if unsupported:
                raise ValueError(
                    f"pass {pass_name!r} expected_regions are unsupported: "
                    + ", ".join(unsupported)
                )
        _bounded_name_list(
            raw_record.get("scope_names"),
            f"pass {pass_name!r} scope_names",
        )


def _parse_warp_list(value: Any, task_name: str) -> set[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"task {task_name!r} warps must be a nonempty list")
    warps: set[int] = set()
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"task {task_name!r} has a non-integer warp")
        if isinstance(item, int):
            start = end = item
        elif isinstance(item, str) and re.fullmatch(r"\d+-\d+", item):
            start, end = (int(part) for part in item.split("-", 1))
        elif isinstance(item, list) and len(item) == 2:
            start, end = item
        elif isinstance(item, Mapping) and set(item) == {"start", "end"}:
            start, end = item["start"], item["end"]
        else:
            raise ValueError(f"task {task_name!r} has an invalid warp or range")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError(f"task {task_name!r} has an invalid warp range")
        if end - start + 1 > _MAX_INSTRUMENTATION_NAMES:
            raise ValueError(f"task {task_name!r} warp range is too large")
        warps.update(range(start, end + 1))
    if len(warps) > _MAX_INSTRUMENTATION_NAMES:
        raise ValueError(f"task {task_name!r} has too many warps")
    return warps


def _bounded_name_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    if len(value) > _MAX_INSTRUMENTATION_NAMES:
        raise ValueError(f"{field} contains too many entries")
    names = tuple(_bounded_name(item, field) for item in value)
    if len(names) != len(set(names)):
        raise ValueError(f"{field} contains duplicates")
    return names


def _bounded_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    if len(value) > _MAX_INSTRUMENTATION_TEXT:
        raise ValueError(f"{field} exceeds {_MAX_INSTRUMENTATION_TEXT} characters")
    if not value.isascii():
        raise ValueError(f"{field} must contain only ASCII characters")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{field} contains an invalid control character")
    return value


def _collect_proton_scope_calls(
    tree: ast.AST,
) -> tuple[tuple[str, str, str], ...]:
    collector = _ProtonScopeCallCollector()
    collector.visit(tree)
    return tuple(collector.calls)


class _ProtonScopeCallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and _is_proton_scope_call(node):
            name = _literal_scope_name(node)
            if name is None:
                raise ValueError("Proton scope calls must use literal scope names")
            name = _bounded_name(name, "Proton scope name")
            predicate = _proton_scope_predicate(node)
            self.calls.append(
                (
                    node.func.attr,
                    name,
                    ast.dump(predicate, include_attributes=False),
                )
            )
        self.generic_visit(node)


def _proton_scope_predicate(node: ast.Call) -> ast.AST:
    if len(node.args) > 1:
        raise ValueError("Proton scope predicates must use the predicate keyword")
    name_keywords = [keyword for keyword in node.keywords if keyword.arg == "name"]
    if len(node.args) + len(name_keywords) != 1:
        raise ValueError("Proton scope calls must have exactly one name")
    unknown_keywords = [
        keyword.arg
        for keyword in node.keywords
        if keyword.arg not in {"name", "predicate"}
    ]
    if unknown_keywords:
        raise ValueError("Proton scope calls contain unsupported keyword arguments")
    predicates = [
        keyword.value for keyword in node.keywords if keyword.arg == "predicate"
    ]
    if len(predicates) != 1:
        raise ValueError("Proton scope calls must have exactly one predicate")
    predicate = predicates[0]
    if not _is_safe_proton_predicate_expression(predicate):
        raise ValueError(
            "Proton scope predicates must be side-effect-free expressions "
            "using values, comparisons, boolean/arithmetic operators, and "
            "optional tl.program_id"
        )
    if isinstance(predicate, ast.Constant):
        raise ValueError("Proton scope predicates must select a subset of launches")
    return predicate


def _validate_scope_balance(tree: ast.Module) -> None:
    _validate_scope_statement_list(tree.body)


def _validate_scope_statement_list(statements: list[ast.stmt]) -> None:
    stack: list[tuple[str, str]] = []
    for statement in statements:
        scope_call = _statement_scope_call(statement)
        if scope_call is not None:
            kind, name, predicate = scope_call
            if kind == "enter_scope":
                stack.append((name, predicate))
            elif not stack:
                raise ValueError(
                    f"Proton scope calls are not properly nested near {name!r}"
                )
            else:
                entered_name, entered_predicate = stack.pop()
                if entered_name != name or entered_predicate != predicate:
                    raise ValueError(
                        "Proton scope enter/exit names and predicates must match near "
                        f"{name!r}"
                    )
        _validate_nested_scope_statement_lists(statement)
    if stack:
        names = ", ".join(name for name, _ in stack[-8:])
        raise ValueError(
            "Proton scope calls cross a lexical or control-flow boundary: " + names
        )


def _statement_scope_call(
    statement: ast.stmt,
) -> tuple[str, str, str] | None:
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and _is_proton_scope_call(statement.value)
    ):
        return None
    call = statement.value
    assert isinstance(call.func, ast.Attribute)
    name = _literal_scope_name(call)
    if name is None:
        raise ValueError("Proton scope calls must use literal scope names")
    predicate = _proton_scope_predicate(call)
    return (
        call.func.attr,
        name,
        ast.dump(predicate, include_attributes=False),
    )


def _validate_nested_scope_statement_lists(node: ast.AST) -> None:
    for _, value in ast.iter_fields(node):
        if isinstance(value, list):
            if value and all(isinstance(item, ast.stmt) for item in value):
                _validate_scope_statement_list(value)
                continue
            for item in value:
                if isinstance(item, ast.AST):
                    _validate_nested_scope_statement_lists(item)
        elif isinstance(value, ast.AST):
            _validate_nested_scope_statement_lists(value)


def _validate_mapping_references(mapping: Mapping[str, Any], scope_names: set[str]) -> None:
    referenced = set(_bounded_name_list(mapping.get("required_scopes"), "required_scopes"))
    tasks = mapping.get("tasks")
    if isinstance(tasks, Mapping):
        for raw_task in tasks.values():
            if isinstance(raw_task, Mapping):
                referenced.add(str(raw_task.get("scope")))
    scope_kinds = mapping.get("scope_kinds")
    if isinstance(scope_kinds, Mapping):
        referenced.update(str(scope) for scope in scope_kinds)
    wait_scopes = mapping.get("wait_scopes")
    if isinstance(wait_scopes, list):
        referenced.update(_bounded_name_list(wait_scopes, "wait_scopes"))
    passes = mapping.get("passes")
    if isinstance(passes, Mapping):
        for record in passes.values():
            if isinstance(record, Mapping):
                referenced.update(
                    _bounded_name_list(record.get("scope_names"), "pass scope_names")
                )
    missing = sorted(referenced - scope_names)
    if missing:
        raise ValueError("instrumentation mapping references missing scopes: " + ", ".join(missing[:8]))


def _validate_proton_setup(tree: ast.Module) -> None:
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Import) and _is_allowed_proton_import(node)
        or isinstance(node, ast.ImportFrom)
        and _is_allowed_proton_import_from(node)
    ]
    if len(imports) != 1:
        raise ValueError(
            "instrumented source must import triton.profiler.language as pl once"
        )
    enable_calls = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _is_triton_semantic_enablement(node.value)
    ]
    if len(enable_calls) != 1:
        raise ValueError(
            "instrumented source must call pl.enable_semantic('triton') once"
        )
    enable_call = enable_calls[0]
    if len(enable_call.args) != 1 or enable_call.keywords:
        raise ValueError("pl.enable_semantic must only receive 'triton'")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(
            isinstance(target, ast.Name)
            and target.id.startswith(_PROTON_HELPER_PREFIX)
            for target in targets
        ):
            continue
        if not _is_proton_helper_assignment(node):
            raise ValueError("Proton predicate helpers contain unsupported syntax")
        if isinstance(node.value, ast.Constant):
            raise ValueError("Proton predicate helpers must select a subset")


def _validate_only_allowed_instrumentation(
    current_tree: ast.Module,
    instrumented_tree: ast.Module,
) -> None:
    stripped_current = _strip_instrumentation(current_tree)
    stripped_instrumented = _strip_instrumentation(instrumented_tree)
    if ast.dump(stripped_current, include_attributes=False) != ast.dump(
        stripped_instrumented,
        include_attributes=False,
    ):
        raise ValueError(
            "instrumented source changes non-instrumentation AST; only Proton imports, "
            "semantic enablement, predicate helpers, scope calls, and empty guard blocks are allowed"
        )


def _strip_instrumentation(tree: ast.Module) -> ast.Module:
    stripped = copy.deepcopy(tree)
    transformer = _InstrumentationStripper()
    transformed = transformer.visit(stripped)
    assert isinstance(transformed, ast.Module)
    ast.fix_missing_locations(transformed)
    return transformed


class _InstrumentationStripper(ast.NodeTransformer):
    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        return None if _is_allowed_proton_import(node) else node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        return None if _is_allowed_proton_import_from(node) else node

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if isinstance(node.value, ast.Call) and (
            _is_triton_semantic_enablement(node.value) or _is_proton_scope_call(node.value)
        ):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if _is_proton_helper_assignment(node):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if _is_proton_helper_assignment(node):
            return None
        return self.generic_visit(node)

    def visit_If(self, node: ast.If) -> ast.AST | None:
        node = self.generic_visit(node)
        assert isinstance(node, ast.If)
        if not node.body and not node.orelse:
            return None
        return node


def _is_allowed_proton_import(node: ast.Import) -> bool:
    return (
        len(node.names) == 1
        and node.names[0].name == "triton.profiler.language"
        and node.names[0].asname == _PROTON_ALIAS
    )


def _is_allowed_proton_import_from(node: ast.ImportFrom) -> bool:
    return (
        node.level == 0
        and node.module == "triton.profiler"
        and len(node.names) == 1
        and node.names[0].name == "language"
        and node.names[0].asname == _PROTON_ALIAS
    )


def _is_triton_semantic_enablement(node: ast.Call) -> bool:
    if not _is_proton_call(node, "enable_semantic"):
        return False
    return (
        bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "triton"
    )


def _is_proton_scope_call(node: ast.Call) -> bool:
    return _is_proton_call(node, "enter_scope") or _is_proton_call(node, "exit_scope")


def _is_proton_call(node: ast.Call, name: str) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == _PROTON_ALIAS
    )


def _literal_scope_name(node: ast.Call) -> str | None:
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    for keyword in node.keywords:
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    return None


def _is_proton_helper_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    if node.value is None:
        return False
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    if not all(
        isinstance(target, ast.Name) and target.id.startswith(_PROTON_HELPER_PREFIX)
        for target in targets
    ):
        return False
    return _is_safe_proton_predicate_expression(node.value)


def _is_safe_proton_predicate_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (bool, int, float))
    if isinstance(node, ast.BoolOp):
        return isinstance(node.op, (ast.And, ast.Or)) and all(
            _is_safe_proton_predicate_expression(value)
            for value in node.values
        )
    if isinstance(node, ast.BinOp):
        return isinstance(
            node.op,
            (
                ast.Add,
                ast.BitAnd,
                ast.BitOr,
                ast.BitXor,
                ast.FloorDiv,
                ast.Mod,
                ast.Mult,
                ast.Sub,
            ),
        ) and _is_safe_proton_predicate_expression(
            node.left
        ) and _is_safe_proton_predicate_expression(node.right)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.Invert, ast.Not, ast.UAdd, ast.USub)) and (
            _is_safe_proton_predicate_expression(node.operand)
        )
    if isinstance(node, ast.Compare):
        return all(
            isinstance(
                operator,
                (
                    ast.Eq,
                    ast.Gt,
                    ast.GtE,
                    ast.Is,
                    ast.IsNot,
                    ast.Lt,
                    ast.LtE,
                    ast.NotEq,
                ),
            )
            for operator in node.ops
        ) and all(
            _is_safe_proton_predicate_expression(value)
            for value in (node.left, *node.comparators)
        )
    return isinstance(node, ast.Call) and _is_allowed_program_id_call(node)


def _is_allowed_program_id_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "program_id"
        and len(node.args) <= 1
        and all(keyword.arg == "axis" for keyword in node.keywords)
        and all(
            isinstance(value, ast.Constant) and isinstance(value.value, int)
            for value in (
                *node.args,
                *(keyword.value for keyword in node.keywords),
            )
        )
    )
