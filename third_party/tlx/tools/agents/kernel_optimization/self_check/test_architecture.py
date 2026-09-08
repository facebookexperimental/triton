from pathlib import Path

from ..contracts import CandidateChange, ChangeScope, ExperimentKind, KernelTarget
from ..decision_maker import DecisionMaker, KernelOptimizer
from ..decision_maker.profiling import native_profiler_for_backend
from ..decision_maker.targets import expected_cuda_major, resolve_target_paths
from ..optimizer import CandidateProposal
from ..optimizer.knowledge import knowledge_paths


def test_public_roles_have_one_implementation() -> None:
    assert DecisionMaker is KernelOptimizer


def test_candidate_can_describe_composed_and_experimental_changes() -> None:
    proposal = CandidateProposal(
        source="kernel source",
        change_scopes=frozenset({ChangeScope.CONFIG, ChangeScope.KERNEL, ChangeScope.COMPILER}),
        experiment_kind=ExperimentKind.IR_OVERRIDE,
        changes=(
            CandidateChange(ChangeScope.CONFIG, "adjust launch configuration"),
            CandidateChange(ChangeScope.KERNEL, "change data movement"),
            CandidateChange(ChangeScope.COMPILER, "scope the confirmed lowering change"),
        ),
    )
    assert proposal.change_scopes == frozenset(ChangeScope)
    assert proposal.experiment_kind is ExperimentKind.IR_OVERRIDE


def test_target_registry_resolves_vendor_architecture_bundle() -> None:
    harness, cases, target = resolve_target_paths(Path("gemm.py"), None, None, None, "b200")
    assert harness.parts[-5:] == (
        "targets",
        "nvidia",
        "blackwell",
        "gemm",
        "harness.py",
    )
    assert cases.name == "cases.json"
    assert target.name == "target.json"
    assert expected_cuda_major("B200") == 10
    assert expected_cuda_major("H100") == 9


def test_platform_registries_select_knowledge_and_profiler() -> None:
    target = KernelTarget(backend="cuda", architecture="B200")
    paths = knowledge_paths(target)
    assert any("knowledge/common" in str(path) for path in paths)
    assert any("knowledge/nvidia/common" in str(path) for path in paths)
    assert any("knowledge/nvidia/blackwell" in str(path) for path in paths)
    assert native_profiler_for_backend("cuda") == "ncu"
    assert native_profiler_for_backend("hip") is None
