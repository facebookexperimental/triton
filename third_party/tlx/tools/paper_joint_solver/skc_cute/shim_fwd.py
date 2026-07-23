"""SKC fwd shim — subclass FlashAttentionForwardSm100, bind at safe points.

Write points (chosen upstream of every consumer of the dual-encoded values):
  1. __init__ tail: register quotas, split_P_arrive (set in __init__ around
     flash_fwd_sm100.py:164/324-337, no later re-derivation).
  2. _setup_attributes() after super(): kv_stage down-clamp + E1 verify
     asserts — all consumers (SMEM layouts :520-557, SharedStorage :694-731,
     pipeline create() :931-1022) run after _setup_attributes returns.

VERIFY-only (never written): q_stage / is_persistent — interface-layer ctor
args; rewriting them here would desynchronize launch geometry.

FA4-venv only (imports flash_attn).
"""

from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

_HD = 128  # binder refuses anything else (uneven_kv_smem derived state at 192)


class SKCForwardSm100(FlashAttentionForwardSm100):
    _skc: dict = {}          # binding dict, set by driver.install()
    _skc_audit: list = []    # class-level: one process = one binding

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        b = type(self)._skc
        if not b:
            return
        audit = type(self)._skc_audit
        assert self.head_dim_padded == _HD and self.head_dim_v_padded == _HD, \
            f"binding only valid at hd128, got {self.head_dim_padded}"
        assert not self.use_2cta_instrs, \
            "binding is 1-CTA only; run under FA_DISABLE_2CTA=1"

        regs = b.get("regs")
        if regs:
            pre = (self.num_regs_softmax, self.num_regs_correction,
                   self.num_regs_other)
            self.num_regs_softmax = regs["num_regs_softmax"]
            self.num_regs_correction = regs["num_regs_correction"]
            self.num_regs_other = (512 - 2 * self.num_regs_softmax
                                   - self.num_regs_correction)
            assert self.num_regs_other >= 24 and self.num_regs_other % 8 == 0
            audit.append({"attr": "num_regs_*", "pre": pre,
                          "post": (self.num_regs_softmax,
                                   self.num_regs_correction,
                                   self.num_regs_other), "mode": "BIND"})

        if "split_P_arrive" in b:
            pre = self.split_P_arrive
            v = b["split_P_arrive"]
            assert v % 32 == 0 and 0 <= v < self.n_block_size
            self.split_P_arrive = v
            audit.append({"attr": "split_P_arrive", "pre": pre, "post": v,
                          "mode": "BIND"})

        for attr, expect in (("q_stage", b.get("verify_q_stage")),
                             ("m_block_size", 128), ("n_block_size", 128)):
            if expect is not None:
                got = getattr(self, attr)
                assert got == expect, f"E1 VERIFY failed: {attr}={got} != {expect}"
                audit.append({"attr": attr, "value": got, "mode": "VERIFY"})

    def _setup_attributes(self):
        super()._setup_attributes()
        b = type(self)._skc
        if not b:
            return
        audit = type(self)._skc_audit
        # E1: FA4's own SMEM formula must re-derive the solver's KV depth.
        expect_kv = b.get("verify_kv_stage")
        if expect_kv is not None:
            assert self.kv_stage == expect_kv, \
                f"E1 VERIFY failed: kv_stage={self.kv_stage} != {expect_kv}"
            audit.append({"attr": "kv_stage", "value": self.kv_stage,
                          "mode": "VERIFY"})
        assert self.s_stage == 2, f"E1 VERIFY failed: s_stage={self.s_stage}"
        assert not getattr(self, "uneven_kv_smem", False), \
            "uneven_kv_smem must be off at hd128"

        clamp = b.get("kv_stage_clamp")
        if clamp is not None:
            assert clamp <= self.kv_stage, \
                "kv_stage may only be clamped downward (SMEM formula is the max)"
            pre = self.kv_stage
            self.kv_stage = clamp
            audit.append({"attr": "kv_stage", "pre": pre, "post": clamp,
                          "mode": "BIND(clamp)"})
