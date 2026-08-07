"""The NVIDIA/PTX checker's architecture-NEUTRAL layer.

What this package is: the part of the PTX forward/backward equivalence checker that carries no
PTX parser, no instruction decoding, and no target-specific addressing — the reduction-tree node
model (:mod:`~bitequiv.core.treeir`), the canonicalizer that post-orders / Merkle-hashes / collapses
those trees (:mod:`~bitequiv.core.canonicalize`), and the symbolic integer algebra an address
evaluator is built on (:mod:`~bitequiv.core.affine_algebra`).

What it is NOT: a neutral third-party library. It was seeded from the NV side and NV has priority —
it is exactly what the NV checker depends on, factored out so a checker for another ISA can reuse
and extend it instead of hard-copying it. It does not aim for 100% ISA purity, and it will not grow
hooks or parameters for a second ISA on speculation. Concretely, a few PTX names still show through:
``_PURE_ELT`` lists PTX mnemonics (``cvt``, ``ex2``, ``rcp``, ...) and ``treeir._norm`` strips PTX's
implicit ``.rn`` rounding modifier. Another ISA adapts to this layer; the layer does not pre-adapt to it.

Layering rule: nothing under ``bitequiv/core/`` may import from ``bitequiv/ptx/`` (or any other
per-ISA package). The dependency runs one way only.
"""
