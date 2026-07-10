"""PTX reduction-tree reconstruction engine for the standalone PTX layout-equivalence
checker (M1-PTX).

This package turns a parsed PTX ``.entry`` into a reconstructed floating-point
reduction tree whose leaves are labeled by the tensor element each load reads (the
PTX echo of the TTGIR thread<->data map), so two PTX modules can be compared
*structurally* for bitwise equivalence — independently of the TTGIR checker.

Layers (built on the ``pyptx`` parser, which gives no def-use / no symbolic eval):

* :mod:`bitequiv.ptx.linker`  — def-use / last-writer over the flat instruction list.
* :mod:`bitequiv.ptx.affine`  — conservative symbolic evaluator for integer address
  registers (affine over ``tid``/``ctaid``/params, or a structural OPAQUE token).
* :mod:`bitequiv.ptx.treeir`  — the reduction-tree node types + canonical serialization.
* :mod:`bitequiv.ptx.builder` — backward tree builder from the result store.
* :mod:`bitequiv.ptx.leaves`  — leaf labeling (the layout analog).

Conservative-fallback floor: anything not provably modeled becomes an OPAQUE token derived
from the verbatim structural text, which matches only a byte-identical token. This makes the
relation **sound relative to the reconstructed result-DAG** — it over-splits (costing tuning
freedom) rather than over-merging on differences it can see. It is not an absolute proof of
bit-equality; see :mod:`bitequiv.ptx_reduction` for the precise guarantee and its residuals
(PTX->SASS, walk-completeness, opaque-token collisions).
"""
