"""Lower eligible (K, 1) convolutions to GEMMs and fused pointwise/reduction ops.

With torchTLX on and ``tlx_config.conv1d_to_mm`` set, a convolution over an [N, C, L, 1]
input with an [O, C/groups, K, 1] weight, stride 1, dilation 1 and no width padding
lowers as follows:
- groups == 1: the weight grad is K GEMMs of the flattened grad against the flattened,
  padded input shifted by k rows; for K == 1 the input grad is a GEMM too.
- depthwise (groups == C == O): forward, input grad and weight grad are K shifted
  multiply-adds, which Inductor fuses into one pointwise kernel or a reduction.
The dense forward and the dense K > 1 input grad keep the original lowering; cuDNN runs
them near their bound.
"""

import hashlib
from pathlib import Path

import torch
from torch._inductor.lowering import lowerings, to_dtype

from . import tlx_config

aten = torch.ops.aten
_CONV = aten.convolution.default
_CONV_BWD = aten.convolution_backward.default
_ORIGINAL: dict = {}


def _ints(xs):
    try:
        return [int(x) for x in xs]
    except TypeError:
        return None


def _params(x, weight, stride, padding, dilation, transposed, output_padding, groups):
    """Return (kind, N, C, L, O, K, P) for an eligible conv, else None."""
    if torch._inductor.config.triton.tlx_mode is None or not tlx_config.conv1d_to_mm:
        return None
    if transposed or any(_ints(output_padding) or [1]) or _ints(stride) != [1, 1] or _ints(dilation) != [1, 1]:
        return None
    xs, ws, pad = _ints(x.get_size()), _ints(weight.get_size()), _ints(padding)
    if xs is None or ws is None or pad is None or len(xs) != 4 or len(ws) != 4:
        return None
    device = x.get_device()
    if device is None or device.type != "cuda" or x.get_dtype() not in (torch.bfloat16, torch.float16):
        return None
    (n, c, l, w), (o, cg, k, kw), (p, pw) = xs, ws, pad
    if w != 1 or kw != 1 or pw != 0 or not 0 <= p <= k - 1:
        return None
    # cuDNN wgrad falls to slow sm80 kernels only off 64-channel multiples: at L=100 on
    # H100, GEMMs win 2.6x at C=96 and lose 2.3x at C=128 (sm90_04_conv1d_to_mm).
    if groups == 1 and cg == c and (c % 64 or o % 64):
        return "dense", n, c, l, o, k, p
    if groups == c == o and cg == 1:
        return "depthwise", n, c, l, o, k, p
    return None


def _f32(x):
    return to_dtype(x, torch.float32)


def _cast(x, dtype):
    return to_dtype(x, dtype)


def _pad_l(x, left, right):
    """Pad dim 2 of an [N, C, L, 1] tensor."""
    return x if left == right == 0 else lowerings[aten.constant_pad_nd.default](x, [0, 0, left, right], 0.0)


def _shift(x, start, length):
    return lowerings[aten.slice.Tensor](x, 2, start, start + length)


def _tap(weight, k, c):
    """weight[:, :, k] of a depthwise weight as a [1, C, 1, 1] fp32 tensor."""
    return _f32(lowerings[aten.view.default](_shift(weight, k, 1), [1, c, 1, 1]))


def _shifted_sum(x, weight, c, k, length, flip):
    """sum_k weight[c, k] * x[..., s_k : s_k + length], accumulated in fp32."""
    acc = None
    for i in range(k):
        start = k - 1 - i if flip else i
        term = lowerings[aten.mul.Tensor](_f32(_shift(x, start, length)), _tap(weight, i, c))
        acc = term if acc is None else lowerings[aten.add.Tensor](acc, term)
    return acc


def _depthwise_fwd(x, weight, bias, n, c, l, k, p):
    l_out = l + 2 * p - k + 1
    acc = _shifted_sum(_pad_l(x, p, p), weight, c, k, l_out, flip=False)
    if bias is not None:
        acc = lowerings[aten.add.Tensor](acc, _f32(lowerings[aten.view.default](bias, [1, c, 1, 1])))
    return _cast(acc, x.get_dtype())


def _channel_sum(x):
    return lowerings[aten.sum.dim_IntList](x, [0, 2, 3])


def _depthwise_bwd(grad, x, weight, mask, n, c, l, k, p):
    l_out = l + 2 * p - k + 1
    dx = dw = None
    if mask[0]:
        dx = _cast(_shifted_sum(_pad_l(grad, k - 1 - p, k - 1 - p), weight, c, k, l, flip=True), x.get_dtype())
    if mask[1]:
        xp, g = _pad_l(x, p, p), _f32(grad)
        taps = [
            lowerings[aten.view.default](_channel_sum(lowerings[aten.mul.Tensor](g, _f32(_shift(xp, i, l_out)))),
                                         [c, 1, 1, 1]) for i in range(k)
        ]
        dw = _cast(lowerings[aten.cat.default](taps, 2), weight.get_dtype())
    return dx, dw


def _nlc(x, n, c, length):
    """[N, C, L, 1] -> [N, L, C] view."""
    return lowerings[aten.permute.default](lowerings[aten.view.default](x, [n, c, length]), [0, 2, 1])


def _flat(x, rows, cols):
    return lowerings[aten.view.default](x, [rows, cols])


def _dense_wgrad(grad, x, n, c, l, o, k, p):
    """dW[:, :, k] = G^T @ X[k : k + R] on [N * Lp, *] flattenings; G is zero-padded to Lp."""
    lp = l + 2 * p
    g = _flat(lowerings[aten.clone.default](_nlc(_pad_l(grad, 0, k - 1), n, o, lp)), n * lp, o)
    xp = _flat(lowerings[aten.clone.default](_nlc(_pad_l(x, p, p), n, c, lp)), n * lp, c)
    rows = n * lp - (k - 1)
    g_t = lowerings[aten.permute.default](lowerings[aten.slice.Tensor](g, 0, 0, rows), [1, 0])
    taps = [
        lowerings[aten.view.default](lowerings[aten.mm.default](g_t, lowerings[aten.slice.Tensor](xp, 0, i, i + rows)),
                                     [o, c, 1, 1]) for i in range(k)
    ]
    return taps[0] if k == 1 else lowerings[aten.cat.default](taps, 2)


def _pointwise_dgrad(grad, weight, n, c, l, o):
    """dX[n] = W^T @ G[n] for a 1x1 conv, as a batched GEMM over N."""
    w_t = lowerings[aten.permute.default](lowerings[aten.view.default](weight, [o, c]), [1, 0])
    w_t = lowerings[aten.expand.default](lowerings[aten.view.default](w_t, [1, c, o]), [n, c, o])
    g = lowerings[aten.view.default](grad, [n, o, l])
    return lowerings[aten.view.default](lowerings[aten.bmm.default](w_t, g), [n, c, l, 1])


def _conv(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    params = _params(x, weight, stride, padding, dilation, transposed, output_padding, groups)
    if params is None or params[0] != "depthwise":
        return _ORIGINAL[_CONV](x, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
    _, n, c, l, _, k, p = params
    return _depthwise_fwd(x, weight, bias, n, c, l, k, p)


def _conv_bwd(grad, x, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask):
    args = (bias_sizes, stride, padding, dilation, transposed, output_padding, groups)
    params = _params(x, weight, stride, padding, dilation, transposed, output_padding, groups)
    if params is None:
        return _ORIGINAL[_CONV_BWD](grad, x, weight, *args, output_mask)
    kind, n, c, l, o, k, p = params
    mask = list(output_mask)
    db = None
    if mask[2]:
        db = _cast(_channel_sum(_f32(grad)), grad.get_dtype())
    if kind == "depthwise":
        dx, dw = _depthwise_bwd(grad, x, weight, mask, n, c, l, k, p)
        return dx, dw, db
    dw = _dense_wgrad(grad, x, n, c, l, o, k, p) if mask[1] else None
    dx = None
    if mask[0] and k == 1:
        dx = _pointwise_dgrad(grad, weight, n, c, l, o)
    elif mask[0]:
        dx = _ORIGINAL[_CONV_BWD](grad, x, weight, *args, [True, False, False])[0]
    return dx, dw, db


def _add_flag_to_cache_key() -> None:
    # tlx_config is not part of the Inductor config hashed into FX/AOTAutograd cache keys.
    from torch._inductor.codecache import FxGraphHashDetails

    original_init = FxGraphHashDetails.__init__
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.tlx_conv1d_to_mm = (tlx_config.conv1d_to_mm, source_hash)

    FxGraphHashDetails.__init__ = init


def register_conv1d_to_mm() -> None:
    """Wrap the conv lowerings once; routing is decided per call by _params()."""
    if _ORIGINAL:
        return
    for op, fn in ((_CONV, _conv), (_CONV_BWD, _conv_bwd)):
        _ORIGINAL[op] = lowerings[op]
        lowerings[op] = fn
    _add_flag_to_cache_key()
