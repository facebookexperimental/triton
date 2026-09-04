from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class TLPlan:
    plan_id: str
    hypothesis: str
    change: str
    evidence: str = ""
    predicted_signal: str = ""
    falsifier: str = ""
    risk: str = ""
    change_scope: str = "kernel-python"
    state: str = "pending"
    worker_id: str = ""
    outcome: str = ""


@dataclass(frozen=True)
class TLRetro:
    worker_id: str
    plan_id: str
    outcome: str
    predicted_signal: str
    observed_signal: str
    diagnosis: str
    dropped_plan_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TLPlanPoolRevision:
    revision: int
    trigger: str
    plans: tuple[TLPlan, ...]
    retro: TLRetro | None = None


class TLProposalPool:
    def __init__(self) -> None:
        self._revision = 0
        self._plans: dict[str, TLPlan] = {}
        self._latest_retro: TLRetro | None = None

    @property
    def latest_retro(self) -> TLRetro | None:
        return self._latest_retro

    def initial(self) -> TLPlanPoolRevision:
        return TLPlanPoolRevision(revision=0, trigger="baseline", plans=())

    def prompt_plans(self) -> tuple[Mapping[str, str], ...]:
        return tuple(
            {
                "plan_id": plan.plan_id,
                "hypothesis": plan.hypothesis,
                "change": plan.change,
                "predicted_signal": plan.predicted_signal,
                "falsifier": plan.falsifier,
                "state": plan.state,
            }
            for plan in self._plans.values()
            if plan.state == "pending"
        )

    def dispatch(self, proposal: Any, worker_id: str) -> TLPlanPoolRevision:
        for plan_id in proposal.drop_plan_ids:
            plan = self._plans.get(plan_id)
            if plan is not None and plan.state == "pending":
                self._plans[plan_id] = replace(plan, state="dropped")
        for index, idea in enumerate(proposal.pending_plans):
            plan_id = idea.get("plan_id") or f"{worker_id}-pending-{index:02d}"
            if plan_id not in self._plans:
                self._plans[plan_id] = TLPlan(
                    plan_id=plan_id,
                    hypothesis=idea.get("hypothesis", ""),
                    change=idea.get("change", ""),
                    evidence=idea.get("evidence", ""),
                    predicted_signal=idea.get("predicted_signal", ""),
                    falsifier=idea.get("falsifier", ""),
                    risk=idea.get("risk", ""),
                    change_scope=idea.get("change_scope", "kernel-python"),
                )
        plan_id = proposal.plan_id or f"plan-{worker_id}"
        selected = self._plans.get(plan_id)
        plan = TLPlan(
            plan_id=plan_id,
            hypothesis=proposal.hypothesis or (selected.hypothesis if selected else ""),
            change=proposal.summary or (selected.change if selected else ""),
            evidence=proposal.evidence or (selected.evidence if selected else ""),
            predicted_signal=proposal.expected_effect
            or (selected.predicted_signal if selected else ""),
            falsifier=proposal.falsifier or (selected.falsifier if selected else ""),
            risk=proposal.risk or (selected.risk if selected else ""),
            change_scope=proposal.change_scope
            or (selected.change_scope if selected else "kernel-python"),
            state="dispatched",
            worker_id=worker_id,
        )
        self._plans[plan_id] = plan
        return self._snapshot(f"dispatch:{worker_id}")

    def refresh(
        self,
        *,
        worker_id: str,
        plan_id: str,
        outcome: str,
        observed_signal: str,
        diagnosis: str,
    ) -> TLPlanPoolRevision:
        selected = self._plans[plan_id]
        self._plans[plan_id] = replace(
            selected,
            state="promoted" if outcome == "global_win" else "completed",
            outcome=outcome,
        )
        dropped: list[str] = []
        if outcome in {"correctness_failure", "implementation_failure"}:
            signature = (selected.hypothesis.strip(), selected.change.strip())
            for pending_id, pending in tuple(self._plans.items()):
                if pending.state != "pending":
                    continue
                if signature == (pending.hypothesis.strip(), pending.change.strip()):
                    self._plans[pending_id] = replace(pending, state="dropped")
                    dropped.append(pending_id)
        self._latest_retro = TLRetro(
            worker_id=worker_id,
            plan_id=plan_id,
            outcome=outcome,
            predicted_signal=selected.predicted_signal,
            observed_signal=observed_signal,
            diagnosis=diagnosis,
            dropped_plan_ids=tuple(dropped),
        )
        return self._snapshot(f"callback:{worker_id}", self._latest_retro)

    def _snapshot(
        self, trigger: str, retro: TLRetro | None = None
    ) -> TLPlanPoolRevision:
        self._revision += 1
        return TLPlanPoolRevision(
            revision=self._revision,
            trigger=trigger,
            plans=tuple(self._plans.values()),
            retro=retro,
        )
