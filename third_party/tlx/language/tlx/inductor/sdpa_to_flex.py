"""Route eligible SDPA flash-attention calls to Inductor's flex-attention lowering.

With ``tlx_config.sdpa_to_flex`` on, ``aten._scaled_dot_product_flash_attention``
and its backward lower through the flex-attention fwd/bwd lowerings (identity
score_mod, no block mask), so flex templates -- stock today, TLX later -- can
serve SDPA callers. Ineligible calls keep the original fallback.

Routed calls autotune their flex template even with max_autotune off, since flex's
default tiles are slower than FA2 at these shapes; the rest of the graph keeps its
config. The fwd returns flex's logsumexp (log2 units) in place of FA2's (ln units); only
the rewritten backward consumes it, so both use the same eligibility rule, and the
flag is part of the FX/AOT graph cache key so cached and fresh graphs never mix.

When the backward's dK and dV only meet in one add (through identical view/cast chains)
and the sm90 TLX backward template applies, the template writes dK + dV once as dK and
dV becomes zeros, so the add reads a single tensor.
"""

import hashlib
import math
import operator
from pathlib import Path

import torch
from torch._inductor import ir
from torch._inductor.ir import ExternKernel
from torch._inductor.lowering import lowerings
from torch._inductor.virtualized import V

from . import tlx_config
from .flex_attention_templates import sm90_flex_bwd_eligible

aten = torch.ops.aten
prims = torch.ops.prims
_FWD = aten._scaled_dot_product_flash_attention.default
_BWD = aten._scaled_dot_product_flash_attention_backward.default
_LARGE_SPARSE_BLOCK_SIZE = 1 << 30
# Shapes where flex beats FA2 on H100 (short queries, D=80/96); others stay on FA2.
HEAD_DIMS = (80, 96)
MAX_Q_LEN = 32

_ORIGINAL: dict = {}


def _static_int(x):
    try:
        return int(x)
    except TypeError:
        return None


def eligible(query, key, value, dropout_p, is_causal) -> bool:
    """Shared by fwd and bwd so the logsumexp producer and consumer always agree.

    Uses only properties both graphs see identically. K/V node identity is not one:
    saved-activation hooks (e.g. fp8 quantization) can split an aliased K=V into
    separate backward nodes.
    """
    if not tlx_config.sdpa_to_flex:
        return False
    if dropout_p != 0.0 or is_causal:
        return False
    device = query.get_device()
    if device is None or device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        return False
    if query.get_dtype() not in (torch.bfloat16, torch.float16):
        return False
    q_len = _static_int(query.get_size()[-2])
    return _static_int(query.get_size()[-1]) in HEAD_DIMS and q_len is not None and q_len <= MAX_Q_LEN


def _graph_module(placeholders, make_output):
    graph = torch.fx.Graph()
    nodes = [graph.placeholder(name) for name in placeholders]
    graph.output(make_output(graph, nodes))
    return torch.fx.GraphModule({}, graph)


def _subgraphs(device):
    """Identity score_mod, its joint graph, and the no-op mask graph."""
    fw = _graph_module(("score", "b", "h", "m", "n"), lambda g, n: n[0])
    # Gradients w.r.t. (score, b, h, m, n); only the score gradient is used.
    joint = _graph_module(
        ("score", "b", "h", "m", "n", "grad_score_mod"),
        lambda g, n: [n[5], None, None, None, None],
    )
    mask = _graph_module(
        ("b", "h", "m", "n"),
        lambda g, n: g.call_function(aten.full.default, ([], True), {"dtype": torch.bool, "device": device}),
    )
    return tuple(ir.Subgraph(name=f"sdpa_to_flex_{k}", graph_module=gm)
                 for k, gm in (("score_mod", fw), ("joint", joint), ("mask_mod", mask)))


def _graph_cache():
    # One set of constant tensors per graph: per-call copies cost a launch each.
    cache = getattr(V.graph, "_tlx_sdpa_to_flex_cache", None)
    if cache is None:
        cache = {}
        V.graph._tlx_sdpa_to_flex_cache = cache
    return cache


def _full(shape, value, dtype, device):
    key = (tuple(shape), value, dtype, str(device))
    cache = _graph_cache()
    if key not in cache:
        cache[key] = lowerings[aten.full.default](list(shape), value, dtype=dtype, device=device)
    return cache[key]


def _empty_block_mask(device, mask_graph):
    """Same contents as torch.nn.attention.flex_attention._create_empty_block_mask."""
    num_blocks = _full([1, 1, 1], 1, torch.int32, device)
    indices = _full([1, 1, 1, 1], 0, torch.int32, device)
    return (
        1,
        1,
        num_blocks,
        indices,
        None,
        None,
        num_blocks,
        indices,
        None,
        None,
        None,
        None,
        None,
        None,
        _LARGE_SPARSE_BLOCK_SIZE,
        _LARGE_SPARSE_BLOCK_SIZE,
        mask_graph,
    )


def _dedup_kv(key, value, key_index):
    """Pass one buffer when K and V are the same FX node.

    The SDPA layout constraint realizes K and V separately, so an aliased K=V would
    be written twice and read twice. Routing itself never depends on this.
    """
    args = V.graph.current_node.args
    return key if args[key_index] is args[key_index + 1] else value


def _kernel_options():
    # Defaults torch.nn.attention.flex_attention._apply_kernel_options sets before the HOP;
    # logsumexp is always written because the SDPA backward consumes it.
    return {
        "BACKEND": "AUTO",
        "PRESCALE_QK": False,
        "ROWS_GUARANTEED_SAFE": False,
        "BLOCKS_ARE_CONTIGUOUS": False,
        "WRITE_DQ": True,
        "OUTPUT_LOGSUMEXP": True,
        "OUTPUT_MAX": False,
    }


# Ops that commute with adding dK and dV when applied identically to both.
_KV_SUM_PASSTHROUGH = (
    prims.convert_element_type.default,
    aten.squeeze.dim,
    aten.squeeze.default,
    aten.unsqueeze.default,
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
    aten.permute.default,
)


def _single_user(node):
    users = list(node.users)
    return users[0] if len(users) == 1 else None


def _kv_grad_sum(node) -> bool:
    """True when dK and dV (outputs 1 and 2) are only used by one add of identical chains.

    The chains need not be bitwise no-ops: with an fp32 upcast, the graph rounds dK and dV
    to the grad dtype and adds in fp32, while the template adds them in its fp32
    accumulator and rounds once. The two differ by at most one grad-dtype rounding.
    """
    getitems = {u.args[1]: u for u in node.users if u.target is operator.getitem}
    if 1 not in getitems or 2 not in getitems:
        return False
    a, b = getitems[1], getitems[2]
    while True:
        ua, ub = _single_user(a), _single_user(b)
        if ua is None or ub is None:
            return False
        if ua is ub:
            return ua.target is aten.add.Tensor and not ua.kwargs and set(ua.args) == {a, b}
        if (
            ua.target not in _KV_SUM_PASSTHROUGH
            or ua.target is not ub.target
            or ua.args[0] is not a
            or ub.args[0] is not b
            or ua.args[1:] != ub.args[1:]
            or ua.kwargs != ub.kwargs
        ):
            return False
        a, b = ua, ub


def _frozen(x):
    x = ExternKernel.realize_input(x)
    ir.as_storage_and_layout(x, freeze=True)
    return x


def _like_meta(x, meta):
    """Match the exact strides the traced graph recorded for this output."""
    return ExternKernel.require_exact_strides(x, meta.stride())


def _placeholder_output(meta, device):
    # cum_seq_*, philox_* and debug_mask: unused by the rerouted backward.
    if not isinstance(meta, torch.Tensor):
        return meta
    return _full(meta.shape, 0, meta.dtype, meta.device or device)


def _sdpa_flash_fwd(query, key, value, dropout_p=0.0, is_causal=False, return_debug_mask=False, *, scale=None):
    metas = V.graph.current_node.meta.get("val")
    lse_shape = list(query.get_size()[:-1])
    if (
        return_debug_mask
        or not eligible(query, key, value, dropout_p, is_causal)
        or metas is None
        or list(metas[1].shape) != [_static_int(s) for s in lse_shape]
    ):
        return _ORIGINAL[_FWD](query, key, value, dropout_p, is_causal, return_debug_mask, scale=scale)
    device = query.get_device()
    value = _dedup_kv(key, value, 1)
    score_mod, _, mask_mod = _subgraphs(device)
    scale = scale if scale is not None else 1.0 / math.sqrt(_static_int(query.get_size()[-1]))
    # The decoding path returns (out, lse); the main template also returns max_scores.
    with torch._inductor.config.patch(max_autotune=True):
        out, lse = lowerings[torch.ops.higher_order.flex_attention](
            query, key, value, score_mod, _empty_block_mask(device, mask_mod), scale, _kernel_options(), [], []
        )[:2]
    # FA2 outputs: out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_mask.
    rest = [_placeholder_output(m, device) for m in metas[2:]]
    return (_like_meta(out, metas[0]), _like_meta(lse, metas[1]), *rest)


def _sdpa_flash_bwd(
    grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
    dropout_p, is_causal, philox_seed, philox_offset, *, scale=None,
):
    if not eligible(query, key, value, dropout_p, is_causal):
        return _ORIGINAL[_BWD](
            grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
            dropout_p, is_causal, philox_seed, philox_offset, scale=scale,
        )
    metas = V.graph.current_node.meta.get("val")
    if metas is None or len(metas) < 3:
        # The forward already returned flex's log2 logsumexp, so FA2 cannot take over here.
        raise RuntimeError(
            f"sdpa_to_flex: SDPA backward node {V.graph.current_node.name} lacks output metadata; "
            "set TORCHINDUCTOR_TLX_SDPA_TO_FLEX=0 to keep FA2"
        )
    device = query.get_device()
    value = _dedup_kv(key, value, 2)
    score_mod, joint, mask_mod = _subgraphs(device)
    scale = scale if scale is not None else 1.0 / math.sqrt(_static_int(query.get_size()[-1]))
    options = _kernel_options()
    from .choices import TLXInductorChoices

    kv_sum = False
    # The choices handler is created on the first compile; one made while TLX was off
    # never reaches the TLX template, and stock choices would leave dK + dV half done.
    if isinstance(V.choices, TLXInductorChoices) and _kv_grad_sum(V.graph.current_node):
        # Template eligibility reads strides, so fix the layouts first.
        shared = value is key
        query, key, grad_out = _frozen(query), _frozen(key), _frozen(grad_out)
        value = key if shared else _frozen(value)
        kv_sum = sm90_flex_bwd_eligible(
            query, key, value, grad_out, options, _LARGE_SPARSE_BLOCK_SIZE, _LARGE_SPARSE_BLOCK_SIZE
        )
    options["TLX_KV_SUM"] = kv_sum
    with torch._inductor.config.patch(max_autotune=True):
        dq, dk, dv, _ = lowerings[torch.ops.higher_order.flex_attention_backward](
            query, key, value, out, logsumexp, grad_out, None, score_mod, joint,
            _empty_block_mask(device, mask_mod), scale, options, [], [],
        )
    if kv_sum:
        # dk holds dK + dV; an unrealized zero keeps the consuming add to one read.
        dv = lowerings[aten.full.default](list(metas[2].shape), 0, dtype=metas[2].dtype, device=device)
        return _like_meta(dq, metas[0]), _like_meta(dk, metas[1]), dv
    return tuple(_like_meta(g, m) for g, m in zip((dq, dk, dv), metas))


def _add_flag_to_cache_key() -> None:
    # tlx_config is not part of the Inductor config hashed into FX/AOTAutograd cache
    # keys (AOTAutogradCacheDetails subclasses FxGraphHashDetails).
    from torch._inductor.codecache import FxGraphHashDetails

    original_init = FxGraphHashDetails.__init__

    # These sources change generated code but are not in the torch/triton keys.
    here = Path(__file__).parent
    source_hash = hashlib.sha256(
        b"".join(
            (here / name).read_bytes()
            for name in ("sdpa_to_flex.py", "flex_attention_templates.py", "sm90_flex_attention_bwd.py.jinja")
        )
    ).hexdigest()

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.tlx_sdpa_to_flex = (tlx_config.sdpa_to_flex, source_hash)

    FxGraphHashDetails.__init__ = init


def register_sdpa_to_flex() -> None:
    """Wrap the SDPA flash fallbacks once; routing is decided per call by eligible()."""
    if _ORIGINAL:
        return
    for op, fn in ((_FWD, _sdpa_flash_fwd), (_BWD, _sdpa_flash_bwd)):
        _ORIGINAL[op] = lowerings[op]
        lowerings[op] = fn
    _add_flag_to_cache_key()
