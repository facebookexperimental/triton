"""SKC bwd shim — subclass FlashAttentionBackwardSm100.

Write point: __init__ tail for register quotas (set flash_bwd_sm100.py:205-234
with the 512 invariant assert).  Q_stage is VERIFY at _setup_attributes
(set :238, consumed by SharedStorage :775+ and pipeline creates in __call__).

FA4-venv only.
"""

from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100


class SKCBackwardSm100(FlashAttentionBackwardSm100):
    _skc: dict = {}
    _skc_audit: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        b = type(self)._skc
        if not b:
            return
        audit = type(self)._skc_audit
        assert not self.use_2cta_instrs, \
            "binding is 1-CTA only; run under FA_DISABLE_2CTA=1"

        regs = b.get("regs")
        if regs:
            pre = (self.num_regs_reduce, self.num_regs_compute,
                   self.num_regs_load, self.num_regs_mma)
            self.num_regs_reduce = regs["num_regs_reduce"]
            self.num_regs_compute = regs["num_regs_compute"]
            self.num_regs_load = regs["num_regs_load"]
            self.num_regs_mma = regs["num_regs_mma"]
            total = (self.num_regs_reduce + 2 * self.num_regs_compute
                     + max(self.num_regs_load, self.num_regs_mma))
            assert total <= 512, f"bwd 512 invariant broken: {total}"
            audit.append({"attr": "num_regs_*", "pre": pre,
                          "post": (self.num_regs_reduce, self.num_regs_compute,
                                   self.num_regs_load, self.num_regs_mma),
                          "mode": "BIND"})

    def _setup_attributes(self):
        super()._setup_attributes()
        b = type(self)._skc
        if not b:
            return
        expect_q = b.get("verify_Q_stage")
        if expect_q is not None:
            assert self.Q_stage == expect_q, \
                f"E1 VERIFY failed: Q_stage={self.Q_stage} != {expect_q}"
            type(self)._skc_audit.append({"attr": "Q_stage",
                                          "value": self.Q_stage,
                                          "mode": "VERIFY"})
        clamp = b.get("Q_stage_clamp")
        if clamp is not None:
            assert clamp <= self.Q_stage
            pre = self.Q_stage
            self.Q_stage = clamp
            type(self)._skc_audit.append({"attr": "Q_stage", "pre": pre,
                                          "post": clamp, "mode": "BIND(clamp)"})
