# torchTLX perf benchmarks (git-repo, standalone; no tritonbench).
#
# Do not import this subpackage from the tlx __init__ chain -- it pulls in torch
# and torch._inductor, which a triton-only consumer must not require.
