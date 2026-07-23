"""SKC Phase B CPU tests (main venv; no flash_attn import).

Tests touching the FA4 clone need env FA4_CUTE=<clone>/flash_attn/cute.
"""

import json
import os
from pathlib import Path

import pytest

from skc_cute import binder_cute, fa4_pin

PKG = Path(__file__).resolve().parent.parent
_FA4_ENV = os.environ.get("FA4_CUTE", "")  # flash_attn/cute dir of the pinned FA4 clone
FA4_CUTE = Path(_FA4_ENV) if _FA4_ENV else None


def _need_fa4():
    if FA4_CUTE is None or not FA4_CUTE.exists():
        pytest.skip("set FA4_CUTE to the FA4 clone flash_attn/cute dir")


def test_pin_matches_clone():
    _need_fa4()
    if not fa4_pin.PIN_FILE.exists():
        pytest.skip("pin not yet written")
    fa4_pin.verify_pin(FA4_CUTE)


def test_pin_detects_drift(tmp_path):
    _need_fa4()
    if not fa4_pin.PIN_FILE.exists():
        pytest.skip("pin not yet written")
    import shutil
    fake = tmp_path / "cute"
    shutil.copytree(FA4_CUTE, fake,
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    f = fake / "flash_fwd_sm100.py"
    f.write_text(f.read_text() + "\n# drift\n")
    with pytest.raises(RuntimeError, match="drift"):
        fa4_pin.verify_pin(fake)


def test_bind_fwd_invariants(tmp_path):
    b = binder_cute.bind_fwd(out_path=tmp_path / "f.json")
    for name, regs in b["overrides"]["reg_candidates"].items():
        sm, corr = regs["num_regs_softmax"], regs["num_regs_correction"]
        other = 512 - 2 * sm - corr
        assert sm % 8 == 0 and corr % 8 == 0 and other % 8 == 0 and other >= 24, name
    assert b["verifies"]["kv_stage"]["expect"] == 3
    assert b["verifies"]["s_stage"]["expect"] == 2
    assert b["verifies"]["q_stage"]["expect"] == 2
    assert b["overrides"]["split_P_arrive"]["default"] == 96
    # four-state classification is complete
    assert b["dropped"] and b["frozen"]


def test_bind_fwd_solver_quota_differs_from_default(tmp_path):
    # E2 is only an experiment if the solver candidate is not the fallback.
    b = binder_cute.bind_fwd(out_path=tmp_path / "f.json")
    cands = b["overrides"]["reg_candidates"]
    assert cands["solver_liveness"] != cands["upstream_default"]


def test_bind_bwd_invariants(tmp_path):
    b = binder_cute.bind_bwd(out_path=tmp_path / "b.json")
    for name, r in b["overrides"]["reg_candidates"].items():
        total = (r["num_regs_reduce"] + 2 * r["num_regs_compute"]
                 + max(r["num_regs_load"], r["num_regs_mma"]))
        assert total <= 512, name
    assert b["verifies"]["Q_stage"]["expect"] == 2
    assert b["verifies"]["dedicated_mma_issuer"]["expect"] is True
    assert "risk_R1" in b


def test_fa4_source_facts_still_hold():
    """The binder's E1 expectations are keyed to source facts — re-grep them."""
    _need_fa4()
    src = (FA4_CUTE / "flash_fwd_sm100.py").read_text()
    # the untuned 1-CTA nc key must still be absent from _TUNING_CONFIG
    assert "(False, False, 128, False)" not in src.split("_FP8_TUNING_CONFIG")[0]
    assert "self.s_stage = 2" in src
    assert "self.split_P_arrive = n_block_size // 4 * 3" in src
    util = (FA4_CUTE / "utils.py").read_text()
    assert "FA_DISABLE_2CTA" in util
