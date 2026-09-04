import torch

from torch._inductor.kernel.flex.common import load_flex_template
from torch._inductor.kernel.flex.flex_attention import flex_attention_grid
from torch._inductor.select_algorithm import SymbolicGridFn, TritonTemplate

from .mm_templates import load_tlx_template
from ..hw.target import current_target


def _make_flex_template(name, source, grid=flex_attention_grid):
    # always_freeze_layout is a newer-torch TritonTemplate kwarg; drop it on
    # torch versions that don't accept it so the module still imports.
    try:
        return TritonTemplate(name=name, grid=grid, source=source,
                              always_freeze_layout=True)
    except TypeError:
        return TritonTemplate(name=name, grid=grid, source=source)


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


@SymbolicGridFn
def gfx950_flex_attention_bwd_grid(
    batch_size,
    q_heads,
    num_queries,
    d_model,
    kv_heads,
    num_key_value,
    meta,
    *,
    cdiv,
):
    del d_model
    return (
        cdiv(num_queries, meta["BLOCK_M2"]) * (q_heads // kv_heads)
        + cdiv(num_key_value, meta["BLOCK_N1"]),
        batch_size,
        kv_heads,
    )


gfx950_flex_attention_backward_template = _make_flex_template(
    "tlx_gfx950_flex_attention_bwd",
    load_tlx_template("gfx950_flex_attention_bwd")
    + load_flex_template("utilities"),
    grid=gfx950_flex_attention_bwd_grid,
)


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
    mutated_inputs=None,
):
    """Add TLX flex-attention template choices to ``choices``.

    Dispatches on the target, mirroring ``mm_templates.append_tlx``: arches in
    ``_AMD_FLEX_ARCHES`` go to :func:`_append_tlx_flex_amd`, everything else to
    :func:`_append_tlx_flex_nvidia`.

    Gated by ``config.triton.tlx_mode``:
      - None (disabled): no-op.
      - "allow":   add TLX candidates alongside the standard template.
      - "force":   drop the standard choices and use only TLX.

    The nine-node forward and sixteen-node backward payloads share this hook;
    backward payloads are dispatched before forward-only indexing. The standard
    forward input list is:
    (query, key, value, logsumexp, max_scores, kv_num_blocks, kv_indices,
    full_kv_num_blocks, full_kv_indices).
    """
    from torch._inductor import config

    if config.triton.tlx_mode is None:
        return choices

    if _is_backward_payload(input_nodes, subgraphs):
        if _is_eligible_amd_flex_bwd(input_nodes):
            _append_tlx_flex_amd_backward(
                choices,
                configs,
                input_nodes,
                subgraphs,
                layout,
                original_kernel_options,
                sparse_q_block_size,
                sparse_kv_block_size,
                mutated_inputs,
            )
        return choices

    if len(input_nodes) != 9:
        return choices

    force_tlx = config.triton.tlx_mode == "force"
    tlx_choices = [] if force_tlx else choices

    query, logsumexp, max_scores = input_nodes[0], input_nodes[3], input_nodes[4]
    mutated_inputs = [logsumexp, max_scores]

    appender = (
        _append_tlx_flex_amd if _use_amd_flex_template() else _append_tlx_flex_nvidia
    )
    appender(
        tlx_choices, configs, input_nodes, subgraphs, layout,
        original_kernel_options, sparse_q_block_size, sparse_kv_block_size,
        query, mutated_inputs,
    )
    if force_tlx and tlx_choices:
        choices.clear()
        choices.extend(tlx_choices)
    return choices


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


def _is_backward_payload(input_nodes, subgraphs):
    """Recognize the stock FlexAttention backward template payload."""
    return len(input_nodes) == 16 and len(subgraphs) == 4


def _is_eligible_amd_flex_bwd(input_nodes):
    """Whether the conservative first gfx950 backward contract applies."""
    if len(input_nodes) != 16:
        return False
    query, key, value = input_nodes[:3]
    if not _use_amd_flex_template():
        return False
    try:
        dtype = query.get_dtype()
        head_dims = tuple(node.get_size()[-1] for node in (query, key, value))
        return (
            dtype == torch.bfloat16
            and key.get_dtype() == dtype
            and value.get_dtype() == dtype
            and head_dims[0] in (64, 128)
            and head_dims[0] == head_dims[1] == head_dims[2]
        )
    except (AttributeError, IndexError, TypeError):
        return False


def _append_tlx_flex_amd_backward(
    choices,
    configs,
    input_nodes,
    subgraphs,
    layout,
    original_kernel_options,
    sparse_q_block_size,
    sparse_kv_block_size,
    mutated_inputs,
):
    """Append mask-aware gfx950 FlexAttention backward candidates."""
    template = gfx950_flex_attention_backward_template
    # Captured score_mod gradients can mutate buffers beyond DQ and DV, so the
    # caller must provide the complete mutation contract.
    if template is None or not configs or mutated_inputs is None:
        return

    from torch._inductor import config

    query, key = input_nodes[0], input_nodes[1]

    opts = original_kernel_options.copy()
    for name in list(opts):
        if name.startswith("bwd_"):
            opts[name[4:]] = opts.pop(name)
        elif name.startswith("fwd_"):
            opts.pop(name)
    opts.pop("TLX_FUSED_KV_DQ", None)
    opts.pop("FUSED_KV_DQ", None)
    opts.pop("DQ_IS_ZEROED", None)

    opts.setdefault("BLOCK_M1", 32)
    opts.setdefault("BLOCK_N1", 128)
    opts.setdefault("BLOCK_M2", 128)
    opts.setdefault("BLOCK_N2", 32)
    opts.setdefault("num_warps", 4)
    opts.setdefault("SPARSE_Q_BLOCK_SIZE", sparse_q_block_size)
    opts.setdefault("SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size)
    try:
        block_m1 = int(opts["BLOCK_M1"])
        block_n1 = int(opts["BLOCK_N1"])
        block_m2 = int(opts["BLOCK_M2"])
        block_n2 = int(opts["BLOCK_N2"])
        num_warps = int(opts["num_warps"])
        effective_sparse_q = int(opts["SPARSE_Q_BLOCK_SIZE"])
        effective_sparse_kv = int(opts["SPARSE_KV_BLOCK_SIZE"])
    except (KeyError, TypeError, ValueError):
        return
    if not (
        (block_m1, block_n1, block_m2, block_n2) == (32, 128, 128, 32)
        and num_warps == 4
        and effective_sparse_q > 0
        and effective_sparse_kv > 0
        and effective_sparse_kv % block_n1 == 0
        and effective_sparse_q % block_m1 == 0
        and effective_sparse_kv % block_n2 == 0
        and effective_sparse_q % block_m2 == 0
    ):
        return

    opts["USE_TMA"] = False
    opts["num_warps"] = num_warps
    # The TLX body hand-pipelines its LDS loads and therefore uses one Triton
    # stage independently of the stock Inductor candidate.
    opts["num_stages"] = 1
    opts["matrix_instr_nonkdim"] = 16
    opts["kpack"] = 1
    opts["waves_per_eu"] = 0

    tlx_choices = []
    template.maybe_append_choice(
        choices=tlx_choices,
        input_nodes=input_nodes,
        layout=layout,
        subgraphs=subgraphs,
        mutated_inputs=mutated_inputs,
        call_sizes=query.get_size() + key.get_size()[1:3],
        **opts,
    )

    if tlx_choices:
        if config.triton.tlx_mode == "force":
            choices.clear()
        choices.extend(tlx_choices)


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
