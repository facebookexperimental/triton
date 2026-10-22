"""Non-GPU checks for the paper benchmark registry and skip metadata."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PKG = Path(__file__).resolve().parent.parent
BENCH_SOURCE = PKG / "bench" / "bench_bars.py"


def _source_tree() -> ast.Module:
    return ast.parse(BENCH_SOURCE.read_text())


def _exec_nodes(*names: str) -> dict[str, object]:
    tree = _source_tree()
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in names
        )
    ]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(selected, type_ignores=[]), BENCH_SOURCE, "exec"), namespace)
    return namespace


def test_jos_registry_targets_v8_emitter_outputs():
    tree = _source_tree()
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("JOS_")
    ]
    namespace = {
        "EXAMPLES": Path("/examples"),
        "PKG_DIR": Path("/pkg"),
        "_JOS_MANIFEST_ENV": "",
    }
    exec(
        compile(ast.Module(assignments, type_ignores=[]), BENCH_SOURCE, "exec"),
        namespace,
    )
    assert namespace["JOS_FWD_KERNEL"] == Path(
        "/examples/case3_FA_fp16_subtiled/generated_jos_v8.py"
    )
    assert namespace["JOS_BWD_KERNEL"] == Path(
        "/examples/case4_FA_bwd_subtiled/generated_jos_v8.py"
    )
    assert namespace["JOS_BWD_LR_KERNEL"] == Path(
        "/examples/case4_FA_bwd_subtiled/generated_jos_lr4096_v8.py"
    )

    make_registry_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_registry"
    )
    registry_namespace = {
        "EXAMPLES": Path("/examples"),
        **{name: namespace[name] for name in namespace if name.startswith("JOS_")},
        "build_triton": lambda *args: args,
        "build_triton_tiled": object(),
        "build_tlx_fwd": lambda *args: args,
        "build_cudnn": object(),
        "build_cudnn_bwd": object(),
        "build_tlx_bwd": lambda *args: args,
    }
    exec(
        compile(ast.Module([make_registry_node], type_ignores=[]), BENCH_SOURCE, "exec"),
        registry_namespace,
    )
    make_registry = registry_namespace["make_registry"]
    fwd = make_registry(SimpleNamespace(mode="fwd"))
    bwd = make_registry(SimpleNamespace(mode="bwd"))
    cfg = object()
    assert fwd["tlx_jos_fwd"](cfg) == (
        cfg,
        namespace["JOS_FWD_KERNEL"],
        "fa_fwd_kernel_nows_subtiled",
    )
    assert bwd["tlx_jos_bwd"](cfg) == (
        cfg,
        namespace["JOS_BWD_KERNEL"],
        "fa_bwd_dkdv_subtiled",
    )
    assert bwd["tlx_jos_bwd_lr"](cfg) == (
        cfg,
        namespace["JOS_BWD_LR_KERNEL"],
        "fa_bwd_dkdv_subtiled",
    )


def test_missing_jos_kernel_is_an_explicit_skip():
    namespace = _run_one_namespace()
    unavailable = namespace["BarUnavailable"]

    def builder(_cfg):
        raise unavailable("missing file: generated_jos_v8.py")

    rec = namespace["run_one"]("tlx_jos_fwd", builder, object(), "fwd", _args())
    assert rec["ok"] is False
    assert rec["status"] == "SKIP"
    assert rec["skip_phase"] == "availability"
    assert rec["reason"] == "missing file: generated_jos_v8.py"
    assert rec["schedule_classification"] == "non-paper-emitter-compatibility"
    assert rec["selection_constraints"] == {
        "prefix_lane_masks": True,
        "full_group_lane_masks": True,
    }
    assert rec["emission_manifest_sha256"] == "manifest-sha"
    assert rec["emitter_solution_sha256"] == "solution-sha"
    assert rec["schedule_graph_sha256"] == "graph-sha"
    assert rec["kernel_sha256"] == "kernel-sha"


def test_jos_compile_failure_is_an_explicit_skip():
    namespace = _run_one_namespace()

    def builder(_cfg):
        def compile_and_check():
            raise RuntimeError("generated kernel did not compile")

        return (lambda: None), compile_and_check

    rec = namespace["run_one"]("tlx_jos_bwd", builder, object(), "bwd", _args())
    assert rec["ok"] is False
    assert rec["status"] == "SKIP"
    assert rec["skip_phase"] == "compile_or_correctness"
    assert rec["reason"] == "RuntimeError: generated kernel did not compile"


def test_jos_manifest_failure_is_an_explicit_provenance_skip():
    namespace = _run_one_namespace()

    def reject(_name, _manifest_path=None):
        raise namespace["EmissionProvenanceError"]("kernel hash mismatch")

    namespace["_verified_jos_provenance"] = reject
    rec = namespace["run_one"](
        "tlx_jos_fwd", lambda _cfg: None, object(), "fwd", _args()
    )

    assert rec["status"] == "SKIP"
    assert rec["skip_phase"] == "provenance"
    assert rec["reason"] == "kernel hash mismatch"


def test_sticky_cuda_cleanup_preserves_explicit_skip():
    namespace = _run_one_namespace()

    def builder(_cfg):
        def compile_and_check():
            raise RuntimeError("generated kernel did not compile")

        return (lambda: None), compile_and_check

    def sticky_sync():
        raise RuntimeError("sticky CUDA error")

    namespace["torch"].cuda.synchronize = sticky_sync
    rec = namespace["run_one"](
        "tlx_jos_bwd", builder, object(), "bwd", _args()
    )

    assert rec["status"] == "SKIP"
    assert rec["reason"] == "RuntimeError: generated kernel did not compile"
    assert rec["cleanup_errors"] == [
        "synchronize: RuntimeError: sticky CUDA error"
    ]


def test_jos_provenance_requires_success_manifest_and_matching_hashes(tmp_path):
    namespace = _exec_nodes(
        "BarUnavailable",
        "EmissionProvenanceError",
        "_sha256",
        "_emission_sources_sha256",
        "_verified_jos_provenance",
    )
    source = tmp_path / "emitter.py"
    kernel = tmp_path / "generated_jos_v8.py"
    graph = tmp_path / "schedule_graph_jos_v8.json"
    solution = tmp_path / "fwd_subtiled_emitter_v8.json"
    manifest_path = tmp_path / "manifest.json"
    source.write_text("source\n")
    kernel.write_text("kernel\n")
    graph.write_text("graph\n")
    solution.write_text("solution\n")
    namespace.update(
        hashlib=hashlib,
        json=json,
        Path=Path,
        TOOLS_DIR=tmp_path,
        EMISSION_SOURCE_FILES=(source,),
        JOS_EMISSION_MANIFEST=tmp_path / "default-missing.json",
        JOS_ARTIFACTS={
            "tlx_jos_fwd": ("fwd_subtiled_emitter_v8", kernel),
        },
    )
    source_digest = namespace["_emission_sources_sha256"]()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "jos-bench-emission-v1",
                "status": "success",
                "selection_constraints": {
                    "classification": "non-paper-emitter-compatibility",
                    "prefix_lane_masks": True,
                    "full_group_lane_masks": True,
                },
                "emission_sources_sha256": source_digest,
                "cases": {
                    "fwd_subtiled_emitter_v8": {
                        "kernel_output": str(kernel),
                        "kernel_sha256": hashlib.sha256(
                            kernel.read_bytes()
                        ).hexdigest(),
                        "graph_output": str(graph),
                        "graph_sha256": hashlib.sha256(
                            graph.read_bytes()
                        ).hexdigest(),
                        "solution": str(solution),
                        "solution_sha256": hashlib.sha256(
                            solution.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        )
    )

    provenance = namespace["_verified_jos_provenance"](
        "tlx_jos_fwd", manifest_path
    )
    assert provenance["emission_manifest"] == str(manifest_path.resolve())
    assert provenance["emission_sources_sha256"] == source_digest
    assert provenance["kernel_sha256"] == hashlib.sha256(
        kernel.read_bytes()
    ).hexdigest()

    source.write_text("changed source\n")
    with pytest.raises(
        namespace["EmissionProvenanceError"], match="sources changed"
    ):
        namespace["_verified_jos_provenance"]("tlx_jos_fwd", manifest_path)
    source.write_text("source\n")
    kernel.write_text("changed\n")
    with pytest.raises(namespace["EmissionProvenanceError"], match="kernel hash"):
        namespace["_verified_jos_provenance"]("tlx_jos_fwd", manifest_path)


def _args() -> SimpleNamespace:
    return SimpleNamespace(warmup=1, rep=1, jos_manifest=None)


def _run_one_namespace() -> dict[str, object]:
    namespace = _exec_nodes(
        "BarUnavailable",
        "EmissionProvenanceError",
        "_set_skip",
        "_cleanup_after_run",
        "run_one",
    )
    cuda = SimpleNamespace(
        synchronize=lambda: None,
        empty_cache=lambda: None,
    )
    namespace.update(
        EXTERNAL_BARS=set(),
        JOS_BARS={"tlx_jos_fwd", "tlx_jos_bwd", "tlx_jos_bwd_lr"},
        _verified_jos_provenance=lambda name, manifest_path=None: {
            "emission_manifest_sha256": "manifest-sha",
            "emission_sources_sha256": "sources-sha",
            "emitter_solution_sha256": "solution-sha",
            "schedule_graph_sha256": "graph-sha",
            "kernel_sha256": "kernel-sha",
        },
        QUANTILES=[0.5, 0.2, 0.8],
        _nvsmi=lambda: "not-run",
        torch=SimpleNamespace(cuda=cuda),
        triton=SimpleNamespace(
            testing=SimpleNamespace(do_bench=lambda *args, **kwargs: (1.0, 1.0, 1.0))
        ),
        gc=SimpleNamespace(collect=lambda: None),
        total_flops=lambda mode, seqlen: 1.0,
    )
    return namespace
