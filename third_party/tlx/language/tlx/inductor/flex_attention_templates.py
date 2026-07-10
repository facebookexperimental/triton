import torch
from torch._inductor.kernel.flex.common import load_flex_template
from torch._inductor.kernel.flex.flex_attention import flex_attention_grid
from torch._inductor.select_algorithm import TritonTemplate

from .mm_templates import load_tlx_template


tlx_flex_attention_template = TritonTemplate(
    name="tlx_blackwell_flex_attention_ws",
    grid=flex_attention_grid,
    source=load_tlx_template("tlx_flex_attention")
    + load_flex_template("utilities"),
    always_freeze_layout=True,
)


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
