import triton
import triton.language as tl


@triton.jit
def _split_n_2D(x, SPLIT_FACTOR: tl.constexpr):
    if SPLIT_FACTOR == 1:
        return (x, )
    else:
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return _split_n_2D(x0, SPLIT_FACTOR // 2) + _split_n_2D(x1, SPLIT_FACTOR // 2)


@triton.jit
def _join_n_2D(xs):
    if len(xs) == 1:
        return xs[0]
    else:
        x0 = _join_n_2D(xs[:len(xs) // 2])
        x1 = _join_n_2D(xs[len(xs) // 2:])
        x = tl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] * 2])
        return x
