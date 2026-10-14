# TLX hardware model.
#
# What GPU are we generating for, and does a given kernel config fit on it?
#
#   resources.py  the hardware facts. One arch class per part (Sm90, Sm100,
#                 Sm107, Gfx942, Gfx950, Gfx1250) collected in ARCH_SPECS, plus
#                 one resource model per template answering "does this tile
#                 fit?" against on-chip bytes and TMEM columns. Add a part
#                 here, not at the call site.
#   target.py     detection and resolution only: current_target() picks the
#                 arch class for the live device and caches it.
#
# This lives outside tlx/inductor/ on purpose, and depends only on torch and
# the standard library -- never on torch._inductor. Two reasons, in order of
# how much they bite:
#
#  1. Import cycle. torch's shim (torch/_inductor/template_heuristics/tlx.py)
#     does `import triton.language.extra.tlx.inductor.registry`, so the chain
#     runs torch._inductor -> tlx.inductor.registry -> tlx.hw. Importing
#     torch._inductor from here would re-enter it mid-initialization.
#  2. Shareability. The standalone tutorial kernels can use this model without
#     dragging in Inductor. Nothing outside tlx/inductor/ uses it yet -- see
#     the note below -- so this one is about keeping the door open.
#
# A test pins reason 1: test_hw_package_does_not_pull_in_inductor imports
# tlx.hw in a subprocess and asserts no torch._inductor module is loaded.
#
# This is NOT a torch-free guarantee: target.py imports torch. It does not need
# to be, since tlx/__init__.py imports neither hw nor inductor, so a
# triton-only consumer never pulls torch through either path.
#
# Not yet shared: third_party/tlx/tutorials/blackwell_gemm_ws.py still carries
# two private copies of the Blackwell SMEM/TMEM estimate (in
# get_heuristic_config and preprocess_configs). They disagree with this model
# -- they cap SMEM at 232*1024 rather than the real 232448, and size TMEM in
# bytes rather than columns, which admits configs whose accumulators cannot
# fit. Folding them in tightens the tutorial's autotune prune, so it is its own
# change with its own perf validation.
