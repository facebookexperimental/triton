"""Bench Triton tutorial 06-fused-attention.py FA-fwd against the
transpose-propagate pass enabled / disabled. Reports TFLOPS at
B=1, H=32, S=4096, D=128, non-causal.

Reads the FA-fwd tutorial from MetaMain/triton (where the pass
is wired in via TRITON_TRANSPOSE_PROPAGATE_ENABLE).
"""
import importlib.util, os, sys, torch, triton

# Stub modules the tutorial imports at top-level but we don't need.
for name in ("pytest", "matplotlib", "matplotlib.pyplot"):
    if name not in sys.modules:
        sys.modules[name] = type(sys)(name)


class _NoOpMark:
    def __getattr__(self, _):
        return lambda *a, **kw: (lambda fn: fn)


sys.modules['pytest'].mark = _NoOpMark()

TUT = os.environ.get(
    "FA_TUTORIAL_PATH",
    "/home/mren/MetaMain/triton/python/tutorials/06-fused-attention.py",
)

sys.modules.setdefault('matplotlib', type(sys)('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', type(sys)('matplotlib.pyplot'))

import triton.testing as _tt
_orig = _tt.perf_report


class _NoOpBench:
    def __init__(self, fn):
        self.fn = fn

    def run(self, *a, **kw):
        pass


_tt.perf_report = lambda *a, **kw: (lambda fn: _NoOpBench(fn))
spec = importlib.util.spec_from_file_location("tut06", TUT)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
finally:
    _tt.perf_report = _orig

attention = mod.attention

# FlyDSL shape
B, H, N_CTX, HEAD_DIM = 1, 32, 4096, 128
DTYPE_NAMES = {"fp16": torch.float16, "bf16": torch.bfloat16}
DEVICE = torch.device("cuda:0")

env_state = {
    "ENABLE": os.environ.get("TRITON_TRANSPOSE_PROPAGATE_ENABLE", "0"),
    "AUTOANNOT": os.environ.get(
        "TRITON_TRANSPOSE_PROPAGATE_AUTOANNOTATE_FIRST_DOT", "0"),
}

print(f"\nTriton tutorial 06-fused-attention.py FA-fwd (MetaMain)")
print(f"Shape: B={B} H={H} S={N_CTX} D={HEAD_DIM}  causal=False")
print(f"GPU: {torch.cuda.get_device_name(0)}   Triton {triton.__version__}")
print(f"TRANSPOSE_PROPAGATE_ENABLE={env_state['ENABLE']}  "
      f"AUTOANNOTATE_FIRST_DOT={env_state['AUTOANNOT']}\n")
print(f"{'dtype':>6}  {'ms':>10}  {'µs':>10}  {'TFLOPS':>10}")

for name, dtype in DTYPE_NAMES.items():
    q = torch.randn((B, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE,
                    requires_grad=False)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    # Warmup
    for _ in range(3):
        out = attention(q, k, v, False, sm_scale)
    torch.cuda.synchronize()

    # Time
    rep = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        out = attention(q, k, v, False, sm_scale)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / rep
    us = ms * 1000
    flops = 4.0 * B * H * N_CTX * N_CTX * HEAD_DIM
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"{name:>6}  {ms:>10.4f}  {us:>10.2f}  {tflops:>10.2f}")
