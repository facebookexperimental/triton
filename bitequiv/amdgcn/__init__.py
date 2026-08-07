"""AMDGCN reduction/GEMM-equivalence engine (the AMD peer of :mod:`bitequiv.ptx`).

Parses the **real** gfx942 machine ISA text (``ck.asm["amdgcn"]`` — register-allocated and
scheduled by LLVM's AMDGPU backend, unlike NVIDIA's virtual PTX) and reconstructs the
floating-point reduction tree / MMA accumulation signature, reusing the ISA-neutral tree
model and canonicalization from :mod:`bitequiv.ptx.treeir`.

Pipeline mirrors the PTX engine: ``parser`` -> ``linker`` -> ``affine`` -> ``leaves`` ->
``builder``.
"""
