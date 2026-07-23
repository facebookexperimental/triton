from pathlib import Path

from torch._inductor.kernel.mm_common import mm_grid  # noqa: F401
from torch._inductor.select_algorithm import SymbolicGridFn, TritonTemplate
from torch._inductor.utils import load_template

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


blackwell_gemm_ws_template = TritonTemplate(
    name="tlx_blackwell_gemm_ws",
    grid=_persistent_mm_grid_split_k,
    source=load_tlx_template("blackwell_gemm_ws"),
)

# TLX warp-pipelined addmm template (AMD / MI350X gfx950), col-major B only.
# Hand-pipelined (num_stages=1): async_load prefetch into multi-buffered LDS +
# tlx.warp_pipeline_stage("mfma"/"mem"). Wins on latency-bound thin-N fp16 addmm.
# The col-major-B requirement is enforced by the heuristic's adjust_kernel_inputs
# (see registry.py); selection/gating is via TORCHINDUCTOR_TLX_MODE (tlx_config).
amd_addmm_warppipe_template = TritonTemplate(
    name="tlx_amd_addmm_warppipe",
    grid=_mm_grid_split_k,  # SPLIT_K=1 -> identical to mm_grid; >1 -> grid_mn*SPLIT_K
    source=load_tlx_template("amd_addmm_warppipe"),
)


def append_tlx(templates, op_name="mm"):
    # Import registry to trigger heuristic registration via decorators
    from . import registry  # noqa: F401

    if op_name == "addmm":
        # The warp-pipe addmm competes as an ADDITIONAL candidate alongside the stock
        # mm_template + vendor (aten). tuned_addmm issues several get_template_configs
        # calls (aten, subgraph, unified); inject only into the call that already carries
        # mm_template (the unified one) so the warp-pipe is added exactly once.
        from torch._inductor.kernel.mm import mm_template

        uids = {getattr(t, "uid", None) for t in templates}
        if mm_template.uid in uids and amd_addmm_warppipe_template.uid not in uids:
            templates.append(amd_addmm_warppipe_template)
    else:
        templates.append(blackwell_gemm_ws_template)
    return templates
