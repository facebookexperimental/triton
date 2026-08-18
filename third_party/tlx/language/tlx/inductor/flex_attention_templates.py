from torch._inductor.kernel.flex.common import load_flex_template
from torch._inductor.kernel.flex.flex_attention import flex_attention_grid
from torch._inductor.select_algorithm import TritonTemplate

from .mm_templates import load_tlx_template
from ..hw.target import current_target


def _make_flex_template(name, source):
    # always_freeze_layout is a newer-torch TritonTemplate kwarg; drop it on
    # torch versions that don't accept it so the module still imports.
    try:
        return TritonTemplate(name=name, grid=flex_attention_grid, source=source,
                              always_freeze_layout=True)
    except TypeError:
        return TritonTemplate(name=name, grid=flex_attention_grid, source=source)


blackwell_flex_attention_template = _make_flex_template(
    "tlx_blackwell_flex_attention_ws",
    load_tlx_template("blackwell_flex_attention") + load_flex_template("utilities"),
)

# MI350X/gfx950 flex-attention: single-task MFMA + LDS async_load body (no warp
# specialization / TMEM / TMA). Shares the flex scaffolding + utilities. Named
# for the arch it was tuned on, as blackwell_* above is; see _AMD_FLEX_ARCHES
# for where it is actually offered.
gfx950_flex_attention_template = _make_flex_template(
    "tlx_gfx950_flex_attention",
    load_tlx_template("gfx950_flex_attention") + load_flex_template("utilities"),
)


#: Arches offered the gfx950 flex-attention template.
#:
#: Unlike the warp-pipe GEMM templates, this one is gated to exactly the arch
#: it was tuned on. The body is portable across CDNA in principle, but gfx942
#: (MI300X) and gfx1250 (MI450X) each need their own pass -- MFMA tile shape,
#: LDS budget and num_warps all differ -- so add them here once that lands.
_AMD_FLEX_ARCHES = frozenset({"gfx950"})


def _use_amd_flex_template() -> bool:
    """True where the AMD flex-attention template applies."""
    return current_target().key in _AMD_FLEX_ARCHES


def append_tlx_flex(
    choices,
    configs,
    input_nodes,
    subgraphs,
    layout,
    original_kernel_options,
    sparse_q_block_size,
    sparse_kv_block_size,
):
    """Add TLX flex-attention template choices to ``choices``.

    Dispatches on the target, mirroring ``mm_templates.append_tlx``: arches in
    ``_AMD_FLEX_ARCHES`` go to :func:`_append_tlx_flex_amd`, everything else to
    :func:`_append_tlx_flex_nvidia`.

    Gated by ``config.triton.tlx_mode``:
      - None (disabled): no-op.
      - "allow":   add TLX candidates alongside the standard template.
      - "force":   drop the standard choices and use only TLX.

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

    appender = (
        _append_tlx_flex_amd if _use_amd_flex_template() else _append_tlx_flex_nvidia
    )
    appender(
        choices, configs, input_nodes, subgraphs, layout,
        original_kernel_options, sparse_q_block_size, sparse_kv_block_size,
        query, mutated_inputs,
    )


def _append_tlx_flex_nvidia(
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
    """Append Blackwell flex-attention candidates: 4-task WS, 2 MMA groups.

    One candidate per base config. The kernel body hard-codes NUM_MMA_GROUPS=2
    and splits BLOCK_M across the two groups, so only base configs whose
    per-group tile meets the tcgen05 MMA minimum (M >= 64, i.e. BLOCK_M >= 128)
    yield a candidate.
    """
    num_sms = current_target().num_sms

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
        blackwell_flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
            subgraphs=subgraphs,
            mutated_inputs=mutated_inputs,
            call_sizes=query.get_size(),
            **opts,
        )

    # BLOCK_M_SPLIT = BLOCK_M // 2 per MMA group; skip base configs whose
    # per-group tile falls below the tcgen05 minimum.
    NUM_MMA_GROUPS = 2
    MIN_MMA_M = 64
    for conf in configs:
        if conf.block_m // NUM_MMA_GROUPS < MIN_MMA_M:
            continue
        opts = tlx_options()
        opts["num_warps"] = conf.num_warps
        opts["num_stages"] = conf.num_stages
        opts["BLOCK_M"] = conf.block_m
        opts["BLOCK_N"] = conf.block_n
        append(opts)


def _append_tlx_flex_amd(
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
        gfx950_flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
            subgraphs=subgraphs,
            mutated_inputs=mutated_inputs,
            call_sizes=query.get_size(),
            **opts,
        )
