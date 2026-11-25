# AMD memory-width analyzer

`amd_memory_width.py` is a standalone static analyzer for final AMDGCN memory
instructions. It accepts compiler-emitted `.amdgcn` text or an `.hsaco` code
object disassembled with AMD-capable `llvm-objdump`.

The tool does not depend on TritonParse. TritonParse can consume its JSON in a
later integration.

## Usage

```bash
python -m triton.tools.amd_memory_width kernel.amdgcn
python -m triton.tools.amd_memory_width kernel.amdgcn --json
python -m triton.tools.amd_memory_width kernel.hsaco --json
python -m triton.tools.amd_memory_width kernel.hsaco \
  --objdump /opt/rocm/llvm/bin/llvm-objdump --json
python -m triton.tools.amd_memory_width /tmp/tlx-ir-dump --top 30
python -m triton.tools.amd_memory_width /tmp/tlx-ir-dump \
  --index /tmp/tlx-ir-dump/index.json --json
```

For `.hsaco` input, tool discovery checks:

1. `--objdump`;
2. `TRITON_AMD_LLVM_OBJDUMP`;
3. `/opt/rocm/llvm/bin/llvm-objdump`; and
4. `llvm-objdump` on `PATH`.

## Report

The report contains:

- opcode, instruction-family, and bytes-per-lane histograms;
- narrow and wide instruction counts;
- unknown-width opcodes;
- adjacent scalar `buffer_load_dword`/`buffer_store_dword` groups that have
  consecutive VGPRs and immediate offsets;
- narrow direct-to-LDS loads; and
- regular buffer loads whose explicit contiguity is below the contiguous
  elements owned per lane by their preserved result layout;
- byte-scale loads where four adjacent logical bytes are distributed across
  lane bases and can be investigated as one packed 32-bit word per row;
- regular buffer stores whose blocked or linear result layout owns more
  contiguous output elements than the final ISA stores per instruction; and
- source locations when compiler-emitted `.loc` comments are present.

An adjacent scalar group is still not proven legal to merge. Every candidate
retains the unproven base-alignment, bounds, and mask-equivalence requirements.
The compiler must establish those facts before changing the instruction width.

A narrow direct-to-LDS instruction is a weaker signal. The report requires
proof of source contiguity, destination contiguity, alignment, bounds, and useful
bytes. In particular, widening one load is not profitable when the lane owns no
additional useful elements.

For a regular `amdg.buffer_load`, the dump-directory analysis compares the
explicit contiguity with the result layout's register bases and the target's
16-byte transaction width. A smaller explicit value is reported as a cap, not
as an automatically legal widening: consecutive addresses, alignment, bounds,
and mask equivalence still require proof. Nested TTGIR callsites are resolved
back to their Python source location. Findings in a kernel that independently
has hot-loop issue headroom rank before prologue or tail-only findings.
Caps whose final ISA is already fully wide are recorded as compiler-recovered
and excluded from the unresolved ranking.
ISA corroboration is source-location-specific when compiler locations match.
When they do not, the report labels the corroboration scope as `kernel` rather
than implying that an unrelated narrow instruction came from the capped site.

Packed-byte analysis is a lane-ownership experiment rather than ordinary
vectorization. It recognizes a logical `[row, 4]` byte quartet whose last
dimension is supplied by a complete `1, 2` lane basis and has no register
ownership. The suggested implementation loads one aligned `uint32` per row and
reinterprets it as two `uint16` and four `uint8` values. This changes which lane
owns the bytes, so word alignment, address adjacency, redistribution cost, and
performance remain explicit gates.

Store analysis reads blocked `sizePerThread` and linear-layout register bases
to derive consecutive output elements owned by each lane. Broadcast and
non-contiguous register bases do not increase that width. It correlates the
candidate with stores at the same source location and reports masks, bounds,
alignment, and layout-conversion cost as unproven requirements. Stores that
remain in MFMA layout are classified as requiring an output-layout conversion
rather than as free vectorization. This distinction matters in practice:
reusing the existing permlane epilogue on
`(64000,256,256)` and moving `(25600,256,256)` to a wide-store inter-wave
family both regressed, so neither kernel experiment was promoted.

For kernels with a load-bearing MFMA loop, the analyzer also computes a
reference-free layout-headroom lower bound:

```text
actual VMEM load issues - ceil(useful bytes per lane / 16)
```

This is not a legal local combine. A positive result means that a different
lane ownership or operand-copy layout could express the same byte volume with
fewer 16-byte transactions. Register pressure, LDS use, alignment, masks, and
occupancy still require compilation and measurement.

When the input is a Triton dump directory, the analyzer reads every `.amdgcn`
file below its `dump` subdirectory. If `index.json` is present, it also groups
the results by operation family and reports the benchmark cases and kernel
names associated with each compilation.

## Full gfx950 MM validation

The complete `gfx950_all` MM and MXFP8 MM focus suites were compiled with a
private Triton cache. All 216 benchmark cases compiled successfully into 242
kernel specializations:

| Family | Cases | Compiled kernels | Fully 16 B/lane | Width histogram, bytes/lane | Adjacent merges | Direct-to-LDS investigations | IR lane-saturated | Unresolved |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `mm` | 138 | 152 | 68 | 2 B: 2,441; 4 B: 246; 8 B: 801; 16 B: 8,320 | 0 | 24 | 0 | 24 |
| `mm_mxfp8` | 78 | 90 | 12 | 1 B: 208; 2 B: 1,888; 4 B: 877; 8 B: 1,102; 16 B: 7,466 | 0 | 244 | 244 | 0 |
| Total | 216 | 242 | 80 | 1 B: 208; 2 B: 4,329; 4 B: 1,123; 8 B: 1,903; 16 B: 15,786 | 0 | 268 | 244 | 24 |

"Fully 16 B/lane" means that every recognized global-memory instruction in
the specialization is at least 16 bytes per lane. A kernel is not defective
merely because it is absent from this column: output element size, boundary
masks, layout ownership, and scale data can all require narrower operations.

There are no post-register-allocation adjacent scalar groups that satisfy the
analyzer's conservative structural checks. Consequently, this corpus provides
no candidate for a mechanical assembly-level `dword` to `dwordx4` combine.

The directory analysis correlates final AMDGCN with TTGIR ownership. It derives
contiguous elements per lane from blocked `sizePerThread` and layout order or
from one-dimensional linear register bases, then applies the element width and
explicit contiguity cap. A final narrow load is lane-saturated only when the
matching IR source site proves that the lane owns no additional useful bytes.
The 244 MXFP8 investigations have such evidence. The 24 MM investigations do
not have a matching TTGIR site and remain unresolved rather than inheriting
kernel-wide IR evidence.

Only one MM specialization has direct-to-LDS investigations:
`float16` `(M, N, K) = (256, 282, 282)`. Its 24 `buffer_load_dword lds`
instructions come from a `contiguity = 2` FP16 copy. The retained dump does not
map those instructions to the same TTGIR source site, so the analyzer does not
use another operation in the kernel to declare them lane-saturated.

The 244 MXFP8 investigations occur in 15 specializations. They are 32-bit
scale copies, including the four scale-copy streams in the inter-wave kernel
and the scale-word copies in the MI32 kernel. Their layouts assign one packed
word, or four E8M0 bytes, to each copying lane. They have the same conclusion:
widening requires a kernel layout and ownership change, not a local compiler
combine.

The same corpus contains 24 source-corroborated packed-byte candidates across
12 MXFP8 generic kernels: one A-scale and one B-scale quartet in each kernel.
There are no such candidates in the dense FP16/BF16 MM corpus.

## Reference-free hot-loop evaluation

The hot-loop analysis was evaluated on all 138 gfx950 MM shapes and compared
afterward with independently selected exhaustive hipBLASLt winners. The tool
reported nine layout-headroom candidates. All five cases where a faster vendor
kernel used a strictly wider steady-state load mix were in that set. The other
four remain layout hypotheses rather than compiler bugs; their vendor winners
did not establish that wider loads improve end-to-end latency.

The generic explicit-contiguity analysis was then run on the original narrow
D132 MM corpus. It reported 48 unique source/layout caps across 36 of 152
compiled kernels. Final-ISA corroboration classified 24 caps as already
widened by codegen and retained 24 unresolved caps across 13 kernels. Only
eight kernels had both an unresolved cap and independent hot-loop issue
headroom, which makes that intersection the first investigation queue. It found
all five known vendor-wider cases without reading vendor ISA:

The retained dumps do not map those cap locations to individual final-ISA
loads, so all 24 classifications carry `isa_corroboration_scope = "kernel"`.
The tool does not represent unrelated narrow instructions as source-local
evidence.

| Original TLX shape `(M,N,K)` | Explicit → owned FP16/lane | Hot-loop excess issues | Source |
| --- | ---: | ---: | --- |
| `(3072,2048,797)` | 4 → 8 | 4 | `gfx950.py:3007` |
| `(3072,4096,797)` | 4 → 8 | 16 | `gfx950.py:3007` |
| `(3072,2048,3965)` | 4 → 8 | 4 | `gfx950.py:3007` |
| `(4096,242432,1894)` | 4 → 8 | 8 | `gfx950.py:3007` |
| `(3072,5888,44536)` | 4 → 8 | 8 | `gfx950.py:3007` |

Each finding says that the preserved TTGIR layout assigns eight FP16 elements
to a lane while the source requests four. The analyzer therefore recommends a
4 → 8 audit and records the remaining legality requirements. The faster
reference kernels are used only after detection to evaluate precision.

The three remaining high-priority hypotheses outside the five known cases were
then widened at the reported PGR2 source site and measured against their
original implementation on the same GPU:

| Shape `(M,N,K)` | Original TLX | Widened TLX | Throughput change | Ratio vs default hipBLASLt |
| --- | ---: | ---: | ---: | ---: |
| `(3072,242176,1892)` | 759 TF/s | 795 TF/s | +4.7% | 0.960x → 1.008x |
| `(128,224,20926)` | 78 TF/s | 80 TF/s | +2.6% | 1.406x → 1.436x |
| `(3072,38848,607)` | 537 TF/s | 549 TF/s | +2.2% | 1.041x → 1.066x |

All three passed exact-shape correctness. Reanalyzing the widened K1892 shader
reports sixteen 16-byte steady-state loads, zero excess issues, and no
steady-state explicit-contiguity cap. Its remaining capped loads are tail-only
and are not correlated with hot-loop headroom.

The ninth hot-loop candidate, `(256,2048,359)`, had no explicit-contiguity
cap: its register kernel issued 24 scalar FP16 loads for 48 bytes per lane.
Reusing the existing transposed compact `MIWT1x2` PGR2 pipeline changed the
logical load ownership without adding a kernel. It improved paired throughput
by 14.7% (approximately 38 to 44 TF/s) and passed exact-shape correctness.
This case demonstrates the second diagnostic class: when no local cap exists,
the issue lower bound points to a layout or pipeline-family experiment rather
than a one-parameter widening.

After removing compiler-recovered caps, five lower-priority narrow-ISA cases
remained. Measuring all five found one large main-load win and one small,
repeatable tail win:

| Shape `(M,N,K)` | Scope | Result |
| --- | --- | ---: |
| `(3072,2048,6640)` | main PGR2 loads | 674 → 787 TF/s (+16.8%) |
| `(2032,2048,7072)`, padded B stride 7104 | masked K tail | +0.5–0.7% |
| `(3072,128,44536)` | main PGR2 loads | neutral |
| `(128,9216,7392)` | main PGR2 loads | neutral |
| BF16 `(1024,4096,6145)` | one-element K tail | neutral |

The neutral candidates were rejected. This final pass shows why both IR and ISA
evidence are needed: explicit source caps can be optimized away, and remaining
narrow tail instructions may be too infrequent to matter.

One non-cap short-K register kernel remained after that filtering.
`(256,282,282)` used 36 scalar dword loads in a fully unrolled body. Expanding
the existing register plan from `32x16x128` to `32x32x128` improved paired
throughput by 2.4--4.9% across two runs and passed exact-shape correctness. The
only other unclassified narrow-load kernel was a Split-K reducer, not a GEMM
main loop.

## Store-width evaluation

The original 152-kernel MM corpus contains 282 TTGIR store sites. Source-local
final-ISA corroboration identifies four linear-layout sites with narrower
emitted stores. Another 105 MFMA-layout sites require an explicit layout
conversion before widening can be considered. Blocked-layout stores retain
their directly derivable last-dimension ownership rather than being counted as
MFMA conversions. Candidates are ranked only when contiguous ownership and a
narrow store at the same source location agree. Layout-conversion candidates
remain visible but do not enter the low-risk widening queue.

The first promoted experiment was FP16 `(M, N, K) = (3072, 2048, 797)`:

| Kernel | VMEM issues per 160 B/lane | Steady-state widths | Paired result |
| --- | ---: | --- | ---: |
| Previous TLX | 14 | 8 x 8 B, 6 x 16 B | baseline |
| Exhaustive hipBLASLt winner | 10 | 10 x 16 B | independent width evidence |
| Packed-direct TLX candidate | 10 | 10 x 16 B | 1.054--1.062x faster than previous TLX |

The packed-direct loop covers 320 bytes with 20 loads, so normalization gives
the same ten issues per 160 bytes as the vendor kernel. The post-change public
TLX dispatch measured 1.073x over default hipBLASLt. Reanalyzing that kernel
removes it from the headroom list. This validates that the diagnostic can
discover the issue without using the vendor ISA; the vendor kernel serves as
independent evidence that the predicted layout is achievable. Cross-harness
latency normalization is intentionally excluded from this conclusion.

The FP16 `(3072, 38848, 607)` candidate is a negative control. The analyzer
reports ten theoretical excess issues, but the vendor steady-state loop uses
the same 8/16-byte mixture and six packed-direct schedule variants measured
within 0.6% of the incumbent. The candidate was rejected. Hot-loop headroom is
therefore a hypothesis generator: promotion still requires a legal layout and
a measured win.

For MXFP8, hipBLASLt rejects `algo_method=all` for block-scaled FP8, so its
heuristic path is not labeled exhaustive or SOTA. The analyzer reports 44
hot-loop layout hypotheses across the 78-shape MXFP8 corpus. Many arise from
packed scale words: widening them requires a different natural, blocked,
K-major, or tile-preshuffled scale contract rather than a local instruction
combine. They remain unvalidated hypotheses until a same-contract candidate is
compiled, checked, and measured.

## Focused validation

One production specialization from each requested family was compiled with a
private Triton cache and analyzed:

| Family | Final memory instructions | Widths, bytes/lane | Adjacent merges | Narrow direct-to-LDS |
| --- | ---: | --- | ---: | ---: |
| `mm` | 56 | 56 x 16 B | 0 | 0 |
| `mm_mxfp8` | 65 | 17 x 4 B, 48 x 16 B | 0 | 17 |

The MXFP8 direct-to-LDS findings map primarily to the four scale-copy sites in
`mm_mxfp8/gfx950.py`. Their `_COPY_SCALE` layout deliberately gives each lane
four adjacent E8M0 bytes. The emitted 32-bit load therefore consumes all useful
per-lane elements. Turning it into a 128-bit load would over-fetch unless the
kernel first redistributes scale ownership. This is a kernel-layout experiment,
not a safe local compiler combine.

## Compiler diagnostic

The AMD `BufferLoadToLocalOpConversion` lowering already computes the facts
needed for missed-width diagnostics:

```text
elements per thread
pointer/offset vector width
mask vector width
explicit contiguity hint
target-supported direct-to-LDS width
destination shared-layout contiguity
```

With `MLIR_ENABLE_DIAGNOSTICS=warnings,remarks`, the lowering compares the
selected vector size with:

```text
min(contiguous_elements_per_thread, widest_supported_target_transaction)
```

When the selected transaction is narrower, the remark reports its width, the
widest useful target-supported width, contiguous elements per thread,
pointer/offset proof, mask proof, explicit contiguity, and whether the limiting
fact is the mask, pointer/alignment information, or destination LDS layout.

The diagnostic intentionally stays silent for the sampled MXFP8 scale copies:
their selected 32-bit transaction equals their four useful E8M0 elements per
lane. An automatic transformation is justified only when useful elements exist,
all legality facts support the wider width, the destination layout remains
coalesced, and a benchmark confirms the register/occupancy trade-off.
