# gfx950 LLIR scheduler reference

This directory preserves the source and design notes for the out-of-tree LLVM
scheduler shipped with ROCm's gfx950 Gluon tutorials.  It is reference material
for future generic Wave scheduling work; it is not built, loaded, or otherwise
integrated with Triton or Wave.

Provenance:

- Repository: `ROCm/gfx950-gluon-tutorials`
- Commit: `f7706fba765193ff1058cd9e97ebe800ba32773d`
- Source date: 2026-07-29
- Tutorial LLVM pin: `850a2b1b975c061ae0fc982ba68064d305485cb2`
- `LlirSchedPlugin.cpp` SHA-256:
  `3c9c92a43fef7198a87063fcf5f11abb8f1cf1fb2f286d17fa248e7ec1af0b13`
- `llir_scheduler.html` SHA-256:
  `b5c22c8f7671bda6742562281fb45c2377f832eea50951957117132c1391a80c`

The prebuilt `libLlirSched.so` is deliberately omitted.  LLVM plugins are ABI
locked, and the tutorial binary only works with its exact LLVM pin.  Rebuild
from `LlirSchedPlugin.cpp` when an executable experiment is needed; the original
build and environment instructions are retained in `UPSTREAM_README.md`.

The broadly reusable parts are the dependency-safe region formation, the
distinction between MFMA-memory throughput scheduling and MFMA-VALU
co-execution, and the use of scheduling barriers to preserve a selected order.
Any future port should express those as generic scheduler/model behavior rather
than recognizing an attention kernel.
