"""Guards the nanobind ``std::optional`` argument contract for ``ir.builder`` bindings.

Unlike pybind11, nanobind does NOT let a ``std::optional<T>`` parameter accept
``None`` on its own: the overload dispatcher rejects ``Py_None`` before the type
caster ever runs unless the argument is flagged none-able via ``nb::arg(...).none()``
(or a ``= std::nullopt`` default). nanobind additionally strips ``std::optional``
from top-level argument types when rendering signatures (``remove_opt_mono``), so a
binding that forgets ``.none()`` looks perfectly normal in the docstring and only
fails at runtime with an opaque ``TypeError: incompatible function arguments`` deep
inside kernel compilation.

The pybind11 -> nanobind port dropped ``.none()`` from every ``std::optional``
parameter in ``triton_tlx.cc`` (and from ``create_descriptor_load`` in ``ir.cc``).
This test parses both sources, finds every ``std::optional`` parameter, and asserts
the built extension renders it as ``... | None``, so it also covers bindings added
after this test was written.
"""

import re
from pathlib import Path

import pytest

import triton._C.libtriton as libtriton

REPO_ROOT = Path(__file__).parents[4]

# Both files add methods to the same `ir.builder` nanobind class.
BUILDER_SOURCES = [
    REPO_ROOT / "python" / "src" / "ir.cc",
    REPO_ROOT / "third_party" / "tlx" / "dialect" / "triton_tlx.cc",
]


def _split_top_level(text):
    """Split on commas while ignoring separators nested in <>, () or []."""
    depth, cur, out = 0, "", []
    for ch in text:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def _optional_params_by_method():
    """Map builder method name -> {0-based param position: C++ param name} for ``std::optional``.

    Positions exclude the implicit ``self``. Matching is positional rather than by
    name because a binding may expose a different name than the C++ lambda uses
    (e.g. ``childLoc`` is bound as ``child_loc``).
    """
    result = {}
    for source in BUILDER_SOURCES:
        src = source.read_text()
        # Each binding looks like: .def("name", [](TritonOpBuilder &self, <params>) -> T {
        for m in re.finditer(r'\.def\(\s*"([A-Za-z0-9_]+)"\s*,\s*\[\]\(([^{]*?)\)\s*(?:->|\{)', src, re.S):
            name, params = m.group(1), m.group(2)
            split = _split_top_level(params)
            # Only methods bound onto the `ir.builder` class; skip free functions
            # and bindings on other classes.
            if not split or "TritonOpBuilder" not in split[0]:
                continue
            opts = {}
            for pos, p in enumerate(split[1:]):  # skip the implicit `self`
                if "std::optional" not in p:
                    continue
                ident = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", p.strip())
                opts[pos] = ident[0] if ident else f"#{pos}"
            if opts:
                result[name] = opts
    return result


def _signature(method_name):
    fn = getattr(libtriton.ir.builder, method_name, None)
    assert fn is not None, f"ir.builder has no method {method_name}"
    return (fn.__doc__ or "").splitlines()[0]


def _signature_args(signature):
    """Rendered argument list of a nanobind signature, excluding ``self``."""
    inner = signature[signature.index("(") + 1:signature.rindex(") ->")]
    args = [a.strip() for a in _split_top_level(inner)]
    args = [a for a in args if a and a != "/"]
    assert args and args[0] == "self", f"unexpected signature: {signature}"
    return args[1:]


# nanobind renders a none-able argument as `name: <type> | None`.
_NONE_ABLE = re.compile(r":[^=]*\|\s*None")


@pytest.mark.skipif(not all(p.exists() for p in BUILDER_SOURCES), reason="C++ sources not available")
def test_all_optional_params_are_none_able():
    optional_params = _optional_params_by_method()
    assert optional_params, "failed to parse any std::optional bindings out of the C++ sources"

    errors = []
    for method, opts in sorted(optional_params.items()):
        sig = _signature(method)
        args = _signature_args(sig)
        for pos, cpp_name in sorted(opts.items()):
            assert pos < len(args), f"{method}: parsed param {cpp_name} at {pos} but signature has {len(args)} args"
            if not _NONE_ABLE.search(args[pos]):
                errors.append(f"{method}({cpp_name}) is std::optional but not none-able\n"
                              f"    rendered as: {args[pos]}\n    signature:   {sig}")

    assert not errors, ("nanobind std::optional parameters missing `py::arg(...).none()`:\n\n" + "\n".join(errors))


@pytest.mark.parametrize(
    "method, args",
    [
        # The bindings that regressed in the pybind11 -> nanobind port. The first
        # two took down the whole Blackwell WS GEMM template path.
        ("create_local_alloc", ("alias", "storageAlias")),
        ("create_storage_alias_spec", ("bufferSizeBytes", )),
        ("create_descriptor_load", ("multicast", )),
    ],
)
def test_known_regressed_bindings_accept_none(method, args):
    sig = _signature(method)
    for arg in args:
        rendered = [a for a in _signature_args(sig) if a.startswith(f"{arg}:")]
        assert rendered, f"{method} has no argument named {arg}; got: {sig}"
        assert _NONE_ABLE.search(rendered[0]), f"{method}({arg}) must accept None; got: {sig}"
