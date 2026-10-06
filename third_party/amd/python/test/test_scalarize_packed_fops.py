import re
from pathlib import Path

import pytest
import triton

current_target = triton.runtime.driver.active.get_current_target()
if current_target.arch not in ("gfx950", "gfx942", "gfx90a", "gfx908"):
    pytest.skip(allow_module_level=True)


def get_func_body(llir):
    func_body = re.findall(r"define amdgpu_kernel void .*? \{(.* ret void.*?)}", llir, flags=re.DOTALL)
    assert len(func_body) == 1, "couldn't find kernel body"
    return func_body[0]


def get_func_body_asm(amdgcn):
    amdgcn = re.findall(r"^attn_fwd:(.*); -- End function", amdgcn, flags=re.DOTALL | re.MULTILINE)
    assert len(amdgcn) == 1, "couldn't find kernel body"
    return amdgcn[0]


def _packed_op_feeds_wide_store(bb, match, wide_store):
    line_start = bb.rfind("\n", 0, match.start()) + 1
    line_end = bb.find("\n", match.end())
    line = bb[line_start:line_end if line_end != -1 else None]
    dest = re.search(r"v\[(\d+):(\d+)\]", line)
    if not dest:
        return False
    lo, hi = sorted((int(dest.group(1)), int(dest.group(2))))
    for store_line in bb.splitlines():
        if wide_store.search(store_line) and all(
            re.search(rf"v(?:\[)?{lane}\b", store_line) for lane in range(lo, hi + 1)
        ):
            return True
    return False


# check there are actually instances of colliding/adjacent fops and mfma without scalarization
def test_check_not_scalarize():
    triton.knobs.amd.scalarize_packed_fops = False
    kernel = triton.compile(str(Path(__file__).parent / "attn_fwd.ttir"), target=current_target)
    llir = kernel.asm["llir"]
    func_body = get_func_body(llir)

    # check for specific patterns that we'll be rewriting in the pass
    def checked_packed_fops_ir_bbs():
        bbs = list(re.split(r"^\d+:\s+; preds = %.*?$", func_body, flags=re.MULTILINE))
        assert len(bbs) > 1, "didn't split func body correctly"
        found_colliding_packed_fop = False
        packed_fop = re.compile(r"= f(add|sub|mul) <")
        for bb in bbs:
            if ("mfma" in bb or "wmma" in bb) and packed_fop.search(bb):
                found_colliding_packed_fop = True
        assert found_colliding_packed_fop, "couldn't find adjacent packed fop and mfma"

    # check that the pattern has the pessimistic effect on the assembly
    amdgcn = get_func_body_asm(kernel.asm["amdgcn"])

    def checked_packed_fops_asm_bbs():
        bbs = list(re.split(r"^.L\w+:", amdgcn, flags=re.MULTILINE))
        assert len(bbs) > 1, "didn't split func body correctly"
        found_mfma = False
        found_colliding_packed_fop = False
        packed_fop = re.compile(r"v_pk_\w+")
        for bb in bbs:
            if "mfma" in bb or "wmma" in bb:
                found_mfma = True
            if packed_fop.search(bb) and ("mfma" in bb or "wmma" in bb):
                found_colliding_packed_fop = True

        assert (found_mfma and found_colliding_packed_fop
                ), f"couldn't find mfma or packed fop {found_mfma=} {found_colliding_packed_fop=}"

    checked_packed_fops_ir_bbs()
    checked_packed_fops_asm_bbs()


# check scalarization "fixes"
def test_check_scalarized():
    triton.knobs.amd.scalarize_packed_fops = True
    kernel = triton.compile(str(Path(__file__).parent / "attn_fwd.ttir"), target=current_target)

    # check the specific IR pattern was rewritten
    llir = kernel.asm["llir"]
    func_body = get_func_body(llir)

    def checked_packed_fops_ir_bbs():
        bbs = list(re.split(r"^\d+:\s+; preds = %.*?$", func_body, flags=re.MULTILINE))
        assert len(bbs) > 1, "didn't split func body correctly"
        found_mfma = False
        packed_fop = re.compile(r"= f(add|sub|mul) <")
        for bb in bbs:
            if "mfma" in bb or "wmma" in bb:
                assert not packed_fop.search(bb)
                found_mfma = True
            if packed_fop.search(bb):
                assert not ("mfma" in bb or "wmma" in bb)

        assert found_mfma, "couldn't find packed mfma"

    # check that it had the profitable effect on the assembly
    amdgcn = get_func_body_asm(kernel.asm["amdgcn"])

    def checked_packed_fops_asm_bbs():
        bbs = list(re.split(r"^.L\w+:", amdgcn, flags=re.MULTILINE))
        assert len(bbs) > 1, "couldn't split amdgcn bbs"
        found_mfma = False
        found_packed_fop = False
        packed_fop = re.compile(r"v_pk_(add|sub|mul)\w+")
        # The backend may merge scalar ALU ops feeding a merged wide store into
        # a packed op (fewer instructions + wider memory op). LLVM's own
        # MFMA-adjacent unpacking leaves those alone when they do not overlap
        # MFMA latency, so they are not scalarization failures. Anything else
        # packed in an MFMA block is.
        wide_store = re.compile(
            r"(ds_write2\w*|buffer_store_format_xy\w*|(?:buffer|flat|global)_store_dwordx2\w*)\b"
        )
        for bb in bbs:
            has_mfma = "mfma" in bb or "wmma" in bb
            if has_mfma:
                found_mfma = True
            for m in packed_fop.finditer(bb):
                if has_mfma:
                    assert _packed_op_feeds_wide_store(bb, m, wide_store), (
                        f"packed op in MFMA block not feeding a wide store: {m.group(0)}"
                    )
                else:
                    found_packed_fop = True
            # the remaining v_pk_muls are in the epilogue; the one v_pk_add left
            # in the MFMA loop feeds a merged ds_write2 wide store (see above)

        assert found_mfma and found_packed_fop, f"couldn't find mfma or packed fop: {found_mfma=}, {found_packed_fop=}"

    checked_packed_fops_ir_bbs()
    checked_packed_fops_asm_bbs()
