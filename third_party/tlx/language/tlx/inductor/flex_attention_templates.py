import torch
from torch._inductor.kernel.flex.common import load_flex_template
from torch._inductor.kernel.flex.flex_attention import flex_attention_grid
from torch._inductor.select_algorithm import TritonTemplate

from .mm_templates import load_tlx_template


def _make_flex_template(name, source):
    # always_freeze_layout is a newer-torch TritonTemplate kwarg; drop it on
    # torch versions that don't accept it so the module still imports.
    try:
        return TritonTemplate(name=name, grid=flex_attention_grid, source=source,
                              always_freeze_layout=True)
    except TypeError:
        return TritonTemplate(name=name, grid=flex_attention_grid, source=source)


tlx_flex_attention_template = _make_flex_template(
    "tlx_blackwell_flex_attention_ws",
    load_tlx_template("tlx_flex_attention") + load_flex_template("utilities"),
)

# AMD CDNA4 (gfx950/MI350) flex-attention: single-task MFMA + LDS async_load body
# (no warp specialization / TMEM / TMA). Shares the flex scaffolding + utilities.
tlx_amd_flex_attention_template = _make_flex_template(
    "tlx_amd_flex_attention",
    load_tlx_template("tlx_amd_flex_attention") + load_flex_template("utilities"),
)


def _is_amd_gfx950() -> bool:
    """True on AMD MI350X (gfx950), where the AMD flex template applies."""
    if torch.version.hip is None:
        return False
    try:
        return "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


def append_tlx_flex_attention_choice(
    choices,
    configs,
    input_nodes,
    subgraphs,
    layout,
    original_kernel_options,
    sparse_q_block_size,
    sparse_kv_block_size,
):
    """Add Blackwell TLX flex-attention template choices to ``choices``.

    Gated by ``config.triton.tlx_mode``:
      - None (disabled): no-op.
      - "allow":   add TLX candidates alongside the standard template.
      - "force":   drop the standard choices and use only TLX.

    Two warp-specialization shapes are offered as autotuning candidates: a
    2-MMA-group variant (one per base config) and a single 1-MMA-group variant
    (fewer barriers, 8 warps).

    ``input_nodes`` is the standard flex-attention forward input list:
    (query, key, value, logsumexp, max_scores, kv_num_blocks, kv_indices,
    full_kv_num_blocks, full_kv_indices).
    """
    from torch._inductor import config

    if config.triton.tlx_mode is None:
        return

    if config.triton.tlx_mode == "force":
        choices.clear()

    query, logsumexp, max_scores = input_nodes[0], input_nodes[3], input_nodes[4]
    mutated_inputs = [logsumexp, max_scores]
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # ---- AMD (gfx950/MI350): single-task MFMA/LDS template ----
    if _is_amd_gfx950():
        _append_amd_flex_choices(
            choices, configs, input_nodes, subgraphs, layout,
            original_kernel_options, sparse_q_block_size, sparse_kv_block_size,
            query, mutated_inputs,
        )
        return

    def tlx_options():
        opts = original_kernel_options.copy()
        for k in list(opts.keys()):
            if k.startswith("fwd_"):
                opts[k[4:]] = opts.pop(k)
            elif k.startswith("bwd_"):
                opts.pop(k)
        opts["USE_TMA"] = True
        opts["NUM_SMS"] = num_sms
        opts.setdefault("SPARSE_Q_BLOCK_SIZE", sparse_q_block_size)
        opts.setdefault("SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size)
        return opts

    def append(opts):
        tlx_flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
            subgraphs=subgraphs,
            mutated_inputs=mutated_inputs,
            call_sizes=query.get_size(),
            **opts,
        )

    # 4-task warp specialization, 2 MMA groups: one candidate per base config.
    for conf in configs:
        opts = tlx_options()
        opts["num_warps"] = conf.num_warps
        opts["num_stages"] = conf.num_stages
        opts["BLOCK_M"] = conf.block_m
        opts["BLOCK_N"] = conf.block_n
        append(opts)

    # 4-task warp specialization, 1 MMA group (fewer barriers): 8 warps.
    last_conf = configs[-1]
    opts = tlx_options()
    opts["num_warps"] = 8
    opts["num_stages"] = 1
    opts["NUM_MMA_GROUPS"] = 1
    opts["BLOCK_M"] = last_conf.block_m
    opts["BLOCK_N"] = last_conf.block_n
    append(opts)


def _append_amd_flex_choices(
    choices,
    configs,
    input_nodes,
    subgraphs,
    layout,
    original_kernel_options,
    sparse_q_block_size,
    sparse_kv_block_size,
    query,
    mutated_inputs,
):
    """Append AMD (gfx950) flex-attention candidates: single-task MFMA/LDS body.

    Unlike the Blackwell path this uses no warp specialization / TMEM / TMA, so
    USE_TMA / NUM_MMA_GROUPS are not set. num_warps follows the MFMA tile
    (BLOCK_M / mfma_m rows per wave), capped at 8.
    """
    def amd_options():
        opts = original_kernel_options.copy()
        for k in list(opts.keys()):
            if k.startswith("fwd_"):
                opts[k[4:]] = opts.pop(k)
            elif k.startswith("bwd_"):
                opts.pop(k)
        opts["USE_TMA"] = False
        opts.setdefault("SPARSE_Q_BLOCK_SIZE", sparse_q_block_size)
        opts.setdefault("SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size)
        return opts

    mfma_m = 32
    for conf in configs:
        opts = amd_options()
        opts["BLOCK_M"] = conf.block_m
        opts["BLOCK_N"] = conf.block_n
        opts["num_warps"] = min(8, max(1, conf.block_m // mfma_m))
        # TLX is hand-pipelined; disable Triton software pipelining.
        opts["num_stages"] = 1
        tlx_amd_flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
            subgraphs=subgraphs,
            mutated_inputs=mutated_inputs,
            call_sizes=query.get_size(),
            **opts,
        )
