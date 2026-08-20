from pathlib import Path

from torch._inductor.kernel.mm_common import mm_grid  # noqa: F401
from torch._inductor.select_algorithm import SymbolicGridFn, TritonTemplate
from torch._inductor.utils import load_template

from ..hw.target import is_rocm

# TLX kernel .jinja templates ship alongside this module (packaged as buck
# resources of the triton beta python library), so load them from here rather
# than from the OSS inductor template dir.
_TLX_TEMPLATE_DIR = Path(__file__).parent


def load_tlx_template(name: str) -> str:
    return load_template(name, template_dir=_TLX_TEMPLATE_DIR)


@SymbolicGridFn
def _persistent_mm_grid_split_k(*args, cdiv, min):
    """Grid for persistent TLX GEMM kernels with split-K support.

    Accepts variable positional args to handle both MM (M, N, meta)
    and BMM (B, M, N, meta) call signatures.
    """
    meta = args[-1]
    M = args[-3]
    N = args[-2]
    num_mn_tiles = cdiv(M, meta["BLOCK_M"]) * cdiv(N, meta["BLOCK_N"])
    split_k = meta.get("SPLIT_K", 1)
    return (min(meta["NUM_SMS"], num_mn_tiles * split_k), 1, 1)


@SymbolicGridFn
def _mm_grid_split_k(m, n, meta, *, cdiv):
    """Non-persistent grid for the warp-pipe addmm: one program per (tile, K-split).

    grid = grid_m * grid_n * SPLIT_K (SPLIT_K defaults to 1 = the plain data-parallel grid,
    identical to mm_grid). Each program owns one output tile and one K-slice; partials are
    summed by _reduce_k_kernel. Unlike _persistent_mm_grid_split_k this is NOT capped at
    NUM_SMS -- the warp-pipe kernel is non-persistent (one tile per program).
    """
    return (
        cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]) * meta.get("SPLIT_K", 1),
        1,
        1,
    )


@SymbolicGridFn
def _bmm_grid_warppipe(b, m, n, meta, *, cdiv):
    """1-D grid for the warp-pipe bmm: BATCH * grid_m * grid_n programs, one per (batch, tile).

    The batch index is recovered in-kernel as pid // grid_mn (grid dim 0 is not 65535-limited,
    and BATCH*grid_mn stays well under 2**31), so no grid_y/grid_z batch split is needed.
    """
    return (b * cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


blackwell_gemm_ws_template = TritonTemplate(
    name="tlx_blackwell_gemm_ws",
    grid=_persistent_mm_grid_split_k,
    source=load_tlx_template("blackwell_gemm_ws"),
)

# TLX warp-pipelined addmm template (MI350X/gfx950), col-major B only.
# Hand-pipelined (num_stages=1): async_load prefetch into multi-buffered LDS +
# tlx.warp_pipeline_stage("mfma"/"mem"). Wins on latency-bound thin-N fp16 addmm.
# The col-major-B requirement is enforced by the heuristic's adjust_kernel_inputs
# (see registry.py); selection/gating is via TORCHINDUCTOR_TLX_MODE (tlx_config).
#
# Scope note, and why these carry an arch name rather than a vendor one: the
# gfx950_* templates are named for where they were tuned and validated, exactly
# as blackwell_* are. Registration is broader than the name -- the warp-pipe
# heuristics register for ALL of ROCm (register=IS_ROCM), just as the Blackwell
# GEMM registers for all of CUDA -- so a config still has to fit the LDS budget
# of the *actual* target. Use resources.AMD_WARP_PIPE with that target's
# DeviceLimits rather than assuming gfx950's 160KB when extending to MI300X
# (gfx942, only 64KB) or MI450X (gfx1250).
gfx950_addmm_warppipe_template = TritonTemplate(
    name="tlx_gfx950_addmm_warppipe",
    grid=_mm_grid_split_k,  # SPLIT_K=1 -> identical to mm_grid; >1 -> grid_mn*SPLIT_K
    source=load_tlx_template("gfx950_addmm_warppipe"),
)

# TLX warp-pipelined bmm template (MI350X/gfx950). Same warp-pipe core as the addmm, plus a
# batch axis on the grid + a per-batch int64 base advance. B is the standard torch.bmm [BATCH,K,N]
# row-major layout (loaded as (BLOCK_K, BLOCK_N) tiles, no transpose). Selection via TLX_MODE.
# Dual path (USE_ASYNC constexpr from the heuristic): aligned K (K % 8 for fp16/bf16) uses the fast
# async_load direct-to-LDS warp-pipe; unaligned/odd K -- which async_copy can't legalize on CDNA4 --
# uses a register-path fallback (tl.load->tl.dot, T280910119). Gate: int32-representable per-batch
# offsets.
gfx950_bmm_warppipe_template = TritonTemplate(
    name="tlx_gfx950_bmm_warppipe",
    grid=_bmm_grid_warppipe,
    source=load_tlx_template("gfx950_bmm_warppipe"),
)

# Persistent variant of the gfx950 warp-pipe addmm: same warp-pipe body, but the grid
# is capped at NUM_SMS (_persistent_mm_grid_split_k) and the kernel loops over output
# tiles. Competes as an additional addmm candidate so per-template selection can be
# attributed in the merge_net gemm sweep. Requires NUM_SMS in the template kwargs
# (supplied by Gfx950AddMMPersistentWarpPipeConfigHeuristic in registry.py).
gfx950_addmm_persistent_warppipe_template = TritonTemplate(
    name="tlx_gfx950_addmm_persistent_warppipe",
    grid=_persistent_mm_grid_split_k,
    source=load_tlx_template("gfx950_addmm_persistent_warppipe"),
)


def append_tlx(templates, op_name="mm"):
    # Import registry to trigger heuristic registration via decorators
    from . import registry  # noqa: F401

    if is_rocm():
        return _append_tlx_amd(templates, op_name)
    return _append_tlx_nvidia(templates, op_name)


def _append_tlx_amd(templates, op_name):
    if op_name == "addmm":
        # The warp-pipe addmm competes as an ADDITIONAL candidate alongside the stock
        # mm_template + vendor (aten). tuned_addmm issues several get_template_configs
        # calls (aten, subgraph, unified); inject only into the call that already carries
        # mm_template (the unified one) so the warp-pipe is added exactly once.
        from torch._inductor.kernel.mm import mm_template

        uids = {getattr(t, "uid", None) for t in templates}
        if mm_template.uid in uids:
            # per-tile warp-pipe (existing candidate)
            if gfx950_addmm_warppipe_template.uid not in uids:
                templates.append(gfx950_addmm_warppipe_template)
            # persistent warp-pipe (new): competes as an ADDITIONAL addmm candidate,
            # kept a distinct template so the sweep attributes its selection count
            # separately from the per-tile warp-pipe.
            if gfx950_addmm_persistent_warppipe_template.uid not in uids:
                templates.append(gfx950_addmm_persistent_warppipe_template)
    elif op_name == "bmm":
        # Compete as an additional candidate alongside the stock bmm_template + aten. Inject once,
        # gated on bmm_template already being present (the unified choice call).
        from torch._inductor.kernel.bmm import bmm_template

        uids = {getattr(t, "uid", None) for t in templates}
        if bmm_template.uid in uids and gfx950_bmm_warppipe_template.uid not in uids:
            templates.append(gfx950_bmm_warppipe_template)
    # else: no AMD TLX template for plain mm (or any other op) yet. Proposing
    # the Blackwell one here is the bug reported in P2462423082: it cannot be
    # selected on gfx9xx and only produces log noise.
    return templates


def _append_tlx_nvidia(templates, op_name):
    # The amd_* warp-pipe templates have no heuristic registered on CUDA, so
    # addmm/bmm get no TLX candidate here; only plain mm does.
    if op_name in ("addmm", "bmm"):
        return templates
    templates.append(blackwell_gemm_ws_template)
    return templates
