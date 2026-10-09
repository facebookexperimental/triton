import os
import re
import subprocess
import sys
from pathlib import Path

from kernel_test_selection import (
    CORRECTNESS_REL,
    KernelMatchers,
    classify,
    collect_imports,
    correctness_selection,
    kernel_matchers_for,
    normalize_changed,
    parse_catalog,
    test_file_affected as file_affected,
    tritonbench_affected,
)

FA = "third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py"
FA_MOD = "triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent"
HSTU_OP = "third_party/tlx/ops/kernels/hstu_attn/sm100.py"
HSTU_TEST = "python/test/unit/tlx_ops/test_hstu_attn_sm100.py"


def matchers(*, prefixes=(), op_names=(), all_ops=False, all_tutorials=False):
    m = KernelMatchers()
    m.prefixes = set(prefixes)
    m.op_names = set(op_names)
    m.all_ops = all_ops
    m.all_tutorials = all_tutorials
    return m


def op_pattern(*names):
    return re.compile(r"\b(" + "|".join(names) + r")\b")


def test_classify_kernel_and_tests():
    assert classify(FA) == "kernel"
    assert classify(HSTU_OP) == "kernel"
    assert classify("third_party/tlx/ops/_catalog.py") == "kernel"
    assert classify("third_party/tlx/tutorials/hstu_self_attn/stubs.py") == "kernel"
    assert classify("third_party/tlx/tutorials/testing/test_correctness.py") == "test"
    assert classify("third_party/tlx/tutorials/test_self_attention_bwd.py") == "test"
    assert classify(HSTU_TEST) == "test"


def test_classify_compiler_and_ignored():
    assert classify("lib/Dialect/TritonGISomething.cpp") == "other"
    assert classify("python/triton/compiler/compiler.py") == "other"
    assert classify(".github/workflows/b200.yml") == "other"
    assert classify(".github/scripts/kernel_test_selection.py") == "other"
    assert classify("__FULL_RUN__") == "other"
    assert classify("") == "other"
    assert classify("docs/index.rst") == "ignored"
    assert classify("website/blog/post.md") == "ignored"
    assert classify("third_party/tlx/tutorials/README.md") == "ignored"
    assert classify(".claude/skills/foo/SKILL.md") == "ignored"


def test_normalize_changed(tmp_path):
    assert normalize_changed("  " + FA + "  ", tmp_path) == FA
    assert normalize_changed("./" + FA, tmp_path) == FA
    assert normalize_changed("", tmp_path) is None
    assert normalize_changed("/etc/passwd", tmp_path) is None
    inside = tmp_path / "a.py"
    inside.write_text("x = 1\n")
    assert normalize_changed(str(inside), tmp_path) == "a.py"


def test_normalize_changed_keeps_dot_directories(tmp_path):
    # A literal "./" prefix is stripped, but leading dots of dot-directories
    # and "../" escapes must survive so classification stays honest.
    assert normalize_changed(".agents/tool.py", tmp_path) == ".agents/tool.py"
    assert normalize_changed(".claude/skills/foo/SKILL.md", tmp_path) == ".claude/skills/foo/SKILL.md"
    assert normalize_changed("./.agents/tool.py", tmp_path) == ".agents/tool.py"
    assert normalize_changed("../escape.py", tmp_path) is None
    assert normalize_changed("a/../../escape.py", tmp_path) is None


def test_parse_catalog_maps_kernel_dir_to_ops(tmp_path):
    (tmp_path / "third_party/tlx/ops").mkdir(parents=True)
    (tmp_path / "third_party/tlx/ops/_catalog.py").write_text(
        'CATALOG = (\n'
        '    OpSpec(op="flash_attn", arch="sm100",\n'
        '           impl="triton.tlx.ops.kernels.flash_attn.sm100:flash_attn"),\n'
        '    OpSpec(op="hstu_attn_dev", arch="sm100",\n'
        '           impl="triton.tlx.ops.kernels.hstu_attn.sm100:hstu_attn"),\n'
        '    OpSpec(op="kda_paged_prefill", arch="gfx950",\n'
        '           impl="triton.tlx.ops.kernels.kda.gfx950_prefill:kda_paged_prefill"),\n'
        '    OpSpec(op="mm_torchtlx", arch="sm100",\n'
        '           impl="triton.language.extra.tlx.inductor.sm100_torch:mm"),\n'
        ')\n')
    catalog, incomplete = parse_catalog(tmp_path)
    assert catalog == {
        "flash_attn": {"flash_attn"},
        "hstu_attn": {"hstu_attn_dev"},
        "kda": {"kda_paged_prefill"},
    }
    # The inductor (non-kernel) impl is deliberately unmapped, not incomplete.
    assert incomplete is False


def test_parse_catalog_unparseable_returns_empty(tmp_path):
    assert parse_catalog(tmp_path) == ({}, True)
    (tmp_path / "third_party/tlx/ops").mkdir(parents=True)
    (tmp_path / "third_party/tlx/ops/_catalog.py").write_text("def broken(:\n")
    assert parse_catalog(tmp_path) == ({}, True)


def test_parse_catalog_partial_flags_incomplete(tmp_path):
    (tmp_path / "third_party/tlx/ops").mkdir(parents=True)
    (tmp_path / "third_party/tlx/ops/_catalog.py").write_text(
        'CATALOG = (\n'
        '    OpSpec(op="flash_attn", arch="sm100",\n'
        '           impl="triton.tlx.ops.kernels.flash_attn.sm100:flash_attn"),\n'
        '    OpSpec(op="dynamic", arch="sm100",\n'
        '           impl=f"triton.tlx.ops.kernels.dynamic.sm100:dynamic"),\n'
        ')\n')
    catalog, incomplete = parse_catalog(tmp_path)
    assert catalog == {"flash_attn": {"flash_attn"}}
    assert incomplete is True


def test_kernel_matchers_tutorial_file_and_package():
    m = kernel_matchers_for([FA], {"hstu_attn": {"hstu_attn_dev"}})
    assert m.prefixes == {FA_MOD}
    assert m.tutorial_stems == {"blackwell_fa_ws_pipelined_persistent"}
    assert not m.all_ops and not m.all_tutorials

    m = kernel_matchers_for(["third_party/tlx/tutorials/hstu_self_attn/stubs.py"], {})
    assert m.prefixes == {"triton.language.extra.tlx.tutorials.hstu_self_attn"}

    m = kernel_matchers_for(["third_party/tlx/tutorials/data/blob.bin"], {})
    assert m.all_tutorials


def test_kernel_matchers_ops_resolves_public_op_names():
    catalog = {"hstu_attn": {"hstu_attn_dev"}, "kda": {"kimi_delta_attention"}}
    m = kernel_matchers_for([HSTU_OP], catalog)
    assert m.prefixes == {"triton.tlx.ops.kernels.hstu_attn"}
    assert m.op_names == {"hstu_attn_dev"}

    m = kernel_matchers_for(["third_party/tlx/ops/__init__.py"], catalog)
    assert m.all_ops

    # Unparseable catalog keeps the fast path but runs every ops test.
    m = kernel_matchers_for([HSTU_OP], {})
    assert m.all_ops


def test_kernel_matchers_partial_catalog_runs_all_ops_tests():
    catalog = {"hstu_attn": {"hstu_attn_dev"}}
    # A partial catalog cannot resolve every op name: run all ops tests.
    m = kernel_matchers_for([HSTU_OP], catalog, True)
    assert m.all_ops
    # Tutorial-only changes are unaffected by ops resolution: keep the fast path.
    m = kernel_matchers_for([FA], catalog, True)
    assert not m.all_ops
    assert m.prefixes == {FA_MOD}


def test_tutorial_change_selects_only_importing_tests():
    m = matchers(prefixes={FA_MOD})
    fa_test = f"from {FA_MOD} import _attn_fwd_ws\n"
    assert file_affected(fa_test, CORRECTNESS_REL, m, None)
    hstu_test = "from triton.tlx.ops import hstu_attn_dev\n"
    assert not file_affected(hstu_test, HSTU_TEST, m, None)
    lang_test = "import triton.language as tl\n"
    assert not file_affected(lang_test, "python/test/unit/language/test_tlx_dot_sm90.py", m, None)


def test_ops_change_matches_shapes_import_and_public_op_use():
    m = matchers(prefixes={"triton.tlx.ops.kernels.hstu_attn"}, op_names={"hstu_attn_dev"})
    pat = op_pattern("hstu_attn_dev")
    shapes = "from triton.tlx.ops.kernels.hstu_attn._shapes import CORRECTNESS_SHAPES\n"
    assert file_affected(shapes, HSTU_TEST, m, pat)
    public = "from triton.tlx.ops import hstu_attn_dev\n"
    assert file_affected(public, HSTU_TEST, m, pat)
    attr_use = "import triton.tlx.ops as ops\n\nout = ops.hstu_attn_dev(q, k)\n"
    assert file_affected(attr_use, HSTU_TEST, m, pat)
    other_op = "from triton.tlx.ops import flash_attn\n"
    assert not file_affected(other_op, "python/test/unit/tlx_ops/test_flash_attn_sm100.py", m, pat)


def test_all_ops_matches_ops_tests_only():
    m = matchers(all_ops=True)
    assert file_affected("from triton.tlx.ops import mm\n", HSTU_TEST, m, None)
    assert file_affected("import torch\n", HSTU_TEST, m, None)
    assert not file_affected("import torch\n", "python/test/unit/runtime/test_cache.py", m, None)


def _write_correctness(repo: Path) -> None:
    path = repo / CORRECTNESS_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent import (\n"
                    "    _attn_fwd_ws as _fa_fwd,\n"
                    ")\n"
                    "from triton.language.extra.tlx.tutorials.hopper_gemm_ws import matmul as _hopper_ws\n"
                    "\n"
                    "class Mxfp8Gemm:\n"
                    "    @staticmethod\n"
                    "    def run_test(shape):\n"
                    "        return _fa_fwd(shape)\n"
                    "\n"
                    "class FlashAttention:\n"
                    "    @staticmethod\n"
                    "    def get_reference(q):\n"
                    "        return q\n"
                    "\n"
                    "def _run_numeric(x):\n"
                    "    return _fa_fwd(x)\n"
                    "\n"
                    "def test_blackwell_fa_ws_pipelined_persistent():\n"
                    "    Mxfp8Gemm.run_test((1, 1, 1))\n"
                    "\n"
                    "def test_blackwell_fa_numeric():\n"
                    "    _run_numeric(1)\n"
                    "\n"
                    "def test_hopper_gemm_ws():\n"
                    "    _hopper_ws(1, 2)\n"
                    "\n"
                    "def test_unrelated_helper():\n"
                    "    FlashAttention.get_reference(1)\n")


def test_correctness_selection_filters_to_referencing_tests(tmp_path):
    _write_correctness(tmp_path)
    m = matchers(prefixes={FA_MOD})
    mode, expr = correctness_selection(tmp_path, m, set())
    assert mode == "filtered"
    assert expr == "test_blackwell_fa_ws_pipelined_persistent or test_blackwell_fa_numeric"


def test_correctness_selection_skips_when_nothing_references(tmp_path):
    _write_correctness(tmp_path)
    m = matchers(prefixes={"triton.language.extra.tlx.tutorials.hstu_self_attn"})
    assert correctness_selection(tmp_path, m, set()) == ("skip", "")


def test_correctness_selection_runs_all_when_file_changed_or_unparseable(tmp_path):
    _write_correctness(tmp_path)
    m = matchers(prefixes={FA_MOD})
    assert correctness_selection(tmp_path, m, {CORRECTNESS_REL}) == ("all", "")
    assert correctness_selection(tmp_path, matchers(all_tutorials=True), set()) == ("all", "")
    (tmp_path / CORRECTNESS_REL).write_text("def broken(:\n")
    assert correctness_selection(tmp_path, m, set()) == ("all", "")


def test_correctness_ops_change_without_ops_import_skips(tmp_path):
    _write_correctness(tmp_path)
    m = matchers(prefixes={"triton.tlx.ops.kernels.hstu_attn"}, op_names={"hstu_attn_dev"})
    assert correctness_selection(tmp_path, m, set()) == ("skip", "")
    assert correctness_selection(tmp_path, matchers(all_ops=True), set()) == ("skip", "")


def test_correctness_ops_change_with_ops_import_runs_all(tmp_path):
    path = tmp_path / CORRECTNESS_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("from triton.tlx.ops import hstu_attn_dev\n\ndef test_x():\n    pass\n")
    m = matchers(op_names={"hstu_attn_dev"})
    assert correctness_selection(tmp_path, m, set()) == ("all", "")


def test_correctness_plain_import_of_kernel_runs_all(tmp_path):
    path = tmp_path / CORRECTNESS_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"import {FA_MOD}\n\ndef test_x():\n    pass\n")
    m = matchers(prefixes={FA_MOD})
    assert correctness_selection(tmp_path, m, set()) == ("all", "")


def test_collect_imports_skips_relative():
    imports = collect_imports("from . import x\nfrom foo import y\nimport a.b\n")
    assert ("foo", ["y"]) in imports
    assert ("a.b", []) in imports
    assert len(imports) == 2


def test_tritonbench_affected(tmp_path):
    tb = tmp_path / "tb"
    (tb / "pkg").mkdir(parents=True)
    (tb / "pkg" / "op.py").write_text("from triton.tlx.ops import hstu_attn_dev\n")
    (tb / ".git" / "hooks").mkdir(parents=True)
    (tb / ".git" / "hooks" / "x.py").write_text("blackwell_fa_ws_pipelined_persistent")

    m = matchers(op_names={"hstu_attn_dev"})
    assert tritonbench_affected(m, [tb]) is True
    m = matchers(prefixes={FA_MOD})
    m.tutorial_stems = {"blackwell_fa_ws_pipelined_persistent"}
    # Only match is under .git: skipped.
    assert tritonbench_affected(m, [tb]) is False
    # No needles (test-only change): skip.
    assert tritonbench_affected(matchers(), [tb]) is False
    # No checkout to scan: fail open.
    assert tritonbench_affected(m, []) is True
    assert tritonbench_affected(matchers(all_ops=True), [tb]) is True


def _write_repo(repo: Path) -> None:
    (repo / "third_party/tlx/ops").mkdir(parents=True)
    (repo / "third_party/tlx/ops/_catalog.py"
     ).write_text('OpSpec(op="hstu_attn_dev", impl="triton.tlx.ops.kernels.hstu_attn.sm100:hstu_attn")\n')
    hstu = repo / HSTU_TEST
    hstu.parent.mkdir(parents=True)
    hstu.write_text("from triton.tlx.ops.kernels.hstu_attn._shapes import CORRECTNESS_SHAPES\n"
                    "from triton.tlx.ops import hstu_attn_dev\n")
    fa_test = repo / "python/test/unit/tlx_ops/test_flash_attn_sm100.py"
    fa_test.write_text("from triton.tlx.ops import flash_attn\n")
    lang = repo / "python/test/unit/language"
    lang.mkdir(parents=True)
    (lang / "test_tlx_dot_sm90.py").write_text("import triton.language as tl\n")
    _write_correctness(repo)


def _run_main(repo: Path, changed: list[str], extra_args=(), env=None):
    changed_file = repo / "changed.txt"
    changed_file.write_text("".join(c + "\n" for c in changed))
    affected = repo / "affected.txt"
    out = repo / "gh_out.txt"
    full_env = dict(os.environ)
    full_env["GITHUB_OUTPUT"] = str(out)
    full_env["TRITONBENCH_ROOT"] = str(repo / "no-such-tb")
    full_env["GITHUB_WORKSPACE"] = str(repo / "no-workspace")
    if env:
        full_env.update(env)
    script = Path(__file__).parent / "kernel_test_selection.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script), "--changed-files",
            str(changed_file), "--affected-out",
            str(affected), "--repo",
            str(repo), *extra_args
        ],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=repo,
    )
    assert proc.returncode == 0, proc.stderr
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)
    return outputs, affected.read_text().splitlines()


def test_main_tutorial_change_selects_only_direct_tests(tmp_path):
    _write_repo(tmp_path)
    tb = tmp_path / "tb"
    (tb / "pkg").mkdir(parents=True)
    (tb / "pkg" / "op.py").write_text("import torch\n")
    outputs, affected = _run_main(tmp_path, [FA], ["--tritonbench", str(tb)])
    assert outputs["kernel_only"] == "true"
    assert outputs["correctness_mode"] == "filtered"
    assert outputs["correctness_k"] == "test_blackwell_fa_ws_pipelined_persistent or test_blackwell_fa_numeric"
    assert outputs["run_tritonbench"] == "false"
    assert affected == [CORRECTNESS_REL]


def test_main_ops_change_selects_ops_test_and_runs_tritonbench_without_checkout(tmp_path):
    _write_repo(tmp_path)
    outputs, affected = _run_main(tmp_path, [HSTU_OP])
    assert outputs["kernel_only"] == "true"
    assert outputs["correctness_mode"] == "skip"
    # No TritonBench checkout to scan: fail open.
    assert outputs["run_tritonbench"] == "true"
    assert affected == [HSTU_TEST]


def test_main_tritonbench_skipped_when_no_reference(tmp_path):
    _write_repo(tmp_path)
    tb = tmp_path / "tb"
    (tb / "pkg").mkdir(parents=True)
    (tb / "pkg" / "op.py").write_text("import torch\n")
    outputs, _ = _run_main(tmp_path, [HSTU_OP], ["--tritonbench", str(tb)])
    assert outputs["run_tritonbench"] == "false"


def test_main_compiler_change_runs_everything(tmp_path):
    _write_repo(tmp_path)
    outputs, affected = _run_main(tmp_path, [FA, "python/triton/compiler/compiler.py"])
    assert outputs["kernel_only"] == "false"
    assert outputs["correctness_mode"] == "all"
    assert outputs["run_tritonbench"] == "true"


def test_main_empty_changed_list_runs_everything(tmp_path):
    _write_repo(tmp_path)
    outputs, _ = _run_main(tmp_path, [])
    assert outputs["kernel_only"] == "false"


def test_main_changed_test_runs_itself(tmp_path):
    _write_repo(tmp_path)
    lang_test = "python/test/unit/language/test_tlx_dot_sm90.py"
    outputs, affected = _run_main(tmp_path, [lang_test])
    assert outputs["kernel_only"] == "true"
    assert lang_test in affected
    assert outputs["correctness_mode"] == "skip"
    assert outputs["run_tritonbench"] == "false"
