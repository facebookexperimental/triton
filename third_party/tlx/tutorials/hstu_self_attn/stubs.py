# Standalone OSS stubs for the fbcode leaf deps used by the HSTU cross-attention
# triton kernels. Lets the kernels run under OSS triton (MetaMain2) without buck.
import triton
import triton.language as tl


def switch_to_contiguous_if_needed(x):
    if x is None:
        return x
    return x.contiguous()


def next_power_of_2(n: int) -> int:
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def prev_power_of_2(n: int) -> int:
    n = int(n)
    return 1 if n < 1 else 1 << (n.bit_length() - 1)


def autotune_max_seq_len(n) -> int:
    return next_power_of_2(int(n))


def triton_autotune(configs, key, **kwargs):
    allowed = {"prune_configs_by", "reset_to_zero", "restore_value", "warmup", "rep", "use_cuda_graph"}
    return triton.autotune(configs=configs, key=key, **{k: v for k, v in kwargs.items() if k in allowed})


def get_full_autotune(*a, **k):
    return False


def is_sm90(*a, **k):
    return False


def is_sm90_plus(*a, **k):
    return False


import functools


class _CustomOpStub:
    """Mimics a torch custom op enough for import: callable + register_fake/register_kernel."""

    def __init__(self, fn):
        self._fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *a, **k):
        return self._fn(*a, **k)

    def register_fake(self, f=None):
        return f if f is not None else (lambda g: g)

    def register_kernel(self, *a, **k):
        return lambda g: g

    def register_autograd(self, *a, **k):
        return lambda g: g


def maybe_register_custom_op(*a, **k):
    if a and callable(a[0]):
        return _CustomOpStub(a[0])

    def deco(fn):
        return _CustomOpStub(fn)

    return deco


# tritoncc spec stubs (only used by fwd/spec codegen paths, not the bwd reduce_dq path)
class NamedSpecType:  # noqa
    pass


class VersionedSpec:  # noqa
    def __init__(self, *a, **k):
        pass


def tritoncc_specs(*a, **k):
    # used as a decorator: @tritoncc_specs(...) -> identity decorator
    def deco(fn):
        return fn
    return deco


# Real acc_dq (self-attention bwd uses it for the dq accumulation), ported from
# generative_recommenders/ops/triton/triton_attention_utils.py.
@triton.jit
def acc_dq(
    dq_ptrs_trans,
    start_m,
    stride_dqm,
    k,
    dqk_trans,
    alpha,
    mask_m,
    MAX_SEQ_LEN,
    LOCK,
    BLOCK_M: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    if ATOMIC_ADD:
        lock_id = start_m // BLOCK_M
        stride_lock = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
        lock = LOCK + tl.program_id(0) * stride_lock + lock_id
        tl.debug_barrier()  # add a barrier to force sync
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass
    dq_trans = tl.load(
        dq_ptrs_trans + start_m * stride_dqm,
        mask=mask_m[None, :],
        other=0.0,
        eviction_policy="evict_last",
    )
    dq_trans += tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
    dq_trans = dq_trans.to(k.dtype)
    tl.store(
        dq_ptrs_trans + start_m * stride_dqm,
        dq_trans,
        mask=mask_m[None, :],
        eviction_policy="evict_last",
    )
    if ATOMIC_ADD:
        tl.atomic_xchg(lock, 0)  # pyre-ignore [61]
