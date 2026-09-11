"""Structural boundary checks for the TLX Wave converter."""

import ast
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_DEF_USE_ATTRIBUTES = frozenset({
    "defining_op",
    "get_defining_op",
    "get_producer",
    "get_producers",
    "get_uses",
    "get_users",
    "producer",
    "producer_op",
    "use_begin",
    "uses",
    "user_begin",
    "users",
})
_HIDDEN_STATE_TOKENS = frozenset({"frontier", "pending"})
_HIDDEN_TOKEN_QUALIFIERS = frozenset({
    "dependency",
    "hidden",
    "memory",
    "release",
    "scratch",
})
_SELECTION_TOKENS = frozenset({
    "optimal",
    "preference",
    "selected",
    "strategy",
})
_WIDTH_TOKENS = frozenset({"chunk", "tile", "vector", "width"})
_LEGACY_OCCUPANCY_NAMES = frozenset({
    "target_waves",
    "waveamdmachine.target_waves",
    "workgroup_target_waves",
})


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    category: str
    detail: str

    def format(self):
        return f"{self.path}:{self.line}: {self.category}: {self.detail}"


def identifier_tokens(name):
    split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return frozenset(token for token in re.split(r"[^A-Za-z0-9]+", split_camel.lower()) if token)


def bridge_policy_attr(name):
    if name in _LEGACY_OCCUPANCY_NAMES:
        return None
    tokens = identifier_tokens(name)
    if tokens & {"transaction", "transactions", "coalesce", "coalescing"}:
        return "target memory transaction selection"
    if tokens & {"bank", "banks"} and tokens & {"count", "num", "number"}:
        return "target memory-bank selection"
    if tokens & _SELECTION_TOKENS and tokens & _WIDTH_TOKENS:
        return "target width or tile selection"
    if {"target", "waves"} <= tokens:
        return "derived target occupancy"
    if tokens & {"repack", "repacking"}:
        return "bridge-local repacking"
    if {"copy", "layout"} <= tokens:
        return "bridge-local layout copy"
    if {"scratch", "layout"} <= tokens:
        return "bridge-local scratch layout"
    if "shuffle" in tokens and tokens & {"mode", "plan", "steps", "strategy"}:
        return "bridge-local shuffle plan"
    if "plan" in tokens and "layout" in tokens and tokens & _SELECTION_TOKENS:
        return "bridge-local layout selection plan"
    return None


def _physical_mask_payload(name):
    tokens = identifier_tokens(name)
    if tokens & {"scc", "vcc"}:
        return "physical SCC/VCC mask payload"
    if {"physical", "mask"} <= tokens:
        return "physical mask payload"
    if "mask" in tokens and tokens & {"bitset", "registers", "words"}:
        return "physical mask storage"
    if name.startswith("mask_predicate_") or name == "mask_scalar_count":
        return "lowered mask predicate schema"
    return None


def _schema_policy(name):
    detail = _physical_mask_payload(name)
    if detail is not None:
        return "physical-mask-payload", detail
    detail = bridge_policy_attr(name)
    if detail is None:
        return None
    category = "layout-policy-plan" if "bridge-local" in detail else "target-policy"
    return category, detail


def _hidden_state(name):
    tokens = identifier_tokens(name)
    if tokens & _HIDDEN_STATE_TOKENS:
        return "mutable pending/frontier state"
    if "token" in tokens and tokens & _HIDDEN_TOKEN_QUALIFIERS:
        return "hidden memory-token state"
    return None


def _is_dataclass_decorator(decorator):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return (isinstance(target, ast.Name) and target.id == "dataclass") or (isinstance(target, ast.Attribute)
                                                                           and target.attr == "dataclass")


def _is_frozen_dataclass(node):
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or not _is_dataclass_decorator(decorator):
            continue
        for keyword in decorator.keywords:
            if (keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True):
                return True
    return False


def _literal_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _assigned_schema_names(target):
    if isinstance(target, ast.Name):
        return (target.id, )
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
        return (target.attr, )
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(name for element in target.elts for name in _assigned_schema_names(element))
    if isinstance(target, ast.Subscript):
        key = _literal_string(target.slice)
        return () if key is None else (key, )
    return ()


class _StructuralBridgeVisitor(ast.NodeVisitor):

    def __init__(self, path, forbidden_import_prefixes):
        self.path = path
        self.forbidden_import_prefixes = tuple(forbidden_import_prefixes)
        self.violations = []
        self._mutable_class_depth = 0

    def _report(self, node, category, detail):
        self.violations.append(Violation(self.path, int(getattr(node, "lineno", 1)), category, detail))

    def _check_schema_name(self, node, name):
        policy = _schema_policy(name)
        if policy is None:
            return
        category, detail = policy
        self._report(node, category, f"schema name `{name}` encodes {detail}")

    def _check_import(self, node, module):
        if any(module == prefix or module.startswith(prefix + ".") for prefix in self.forbidden_import_prefixes):
            self._report(node, "forbidden-import", f"imports policy implementation `{module}`")

    def visit_Import(self, node):
        for alias in node.names:
            self._check_import(node, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module is not None:
            self._check_import(node, node.module)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr == "owner":
            self._report(node, "emitted-owner-inspection", "inspects an emitted or source IR value owner")
        elif node.attr in _DEF_USE_ATTRIBUTES:
            self._report(node, "def-use-walk", f"uses raw def-use API `{node.attr}`")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in _DEF_USE_ATTRIBUTES:
            self._report(node, "def-use-walk", f"uses raw def-use API `{node.func.id}`")
        for keyword in node.keywords:
            if keyword.arg is not None:
                self._check_schema_name(keyword, keyword.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self._check_schema_name(argument, argument.arg)
        if node.args.vararg is not None:
            self._check_schema_name(node.args.vararg, node.args.vararg.arg)
        if node.args.kwarg is not None:
            self._check_schema_name(node.args.kwarg, node.args.kwarg.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Dict(self, node):
        for key in node.keys:
            name = _literal_string(key)
            if name is not None:
                self._check_schema_name(key, name)
        self.generic_visit(node)

    def _check_assignment(self, target):
        for name in _assigned_schema_names(target):
            self._check_schema_name(target, name)
            detail = _hidden_state(name) if self._mutable_class_depth else None
            if detail is not None:
                self._report(target, "hidden-memory-state", f"mutable field `{name}` encodes {detail}")

    def visit_Assign(self, node):
        for target in node.targets:
            self._check_assignment(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._check_assignment(node.target)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self._check_schema_name(node, node.name)
        is_mutable = not _is_frozen_dataclass(node)
        self._mutable_class_depth += int(is_mutable)
        for statement in node.body:
            self.visit(statement)
        self._mutable_class_depth -= int(is_mutable)


def scan_text(text, *, path="<memory>", forbidden_import_prefixes=()):
    source_path = Path(path)
    try:
        tree = ast.parse(text, filename=str(source_path))
    except SyntaxError as exc:
        return [Violation(source_path, int(exc.lineno or 1), "parse-error", str(exc.msg))]
    visitor = _StructuralBridgeVisitor(source_path, forbidden_import_prefixes)
    visitor.visit(tree)
    return visitor.violations


def scan_paths(paths: Iterable[str | Path], *, forbidden_import_prefixes: Sequence[str] = ()):
    violations = []
    for raw_path in paths:
        path = Path(raw_path)
        sources = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for source in sources:
            if source.suffix != ".py" or not source.is_file():
                continue
            violations.extend(
                scan_text(
                    source.read_text(),
                    path=source,
                    forbidden_import_prefixes=forbidden_import_prefixes,
                ))
    return violations
