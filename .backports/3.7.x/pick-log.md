# Pick log — release/3.7.x (branch `backport/3.7.x-backports`)

Two-pass pick. Pass 1: cherry-pick the pick-list in bulk, keep clean applies, skip conflicts, drop empties,
then build once. Pass 2: resolve the skipped (conflicted) commits one at a time, resolving to match Meta main.

## Pass 1 — bulk clean picks

| # | source | sha | commit title | notes (pick or skip) |
|---|--------|-----|--------------|----------------------|
| 1 | openai | 5483c9b95 | [Release] Revert enabling use_async_copy by default (#9445) | skip (conflict → pass 2) |
| 2 | openai | 7e48d5dfc | [Nvidia] Revert ptxas SM89 workaround #7067 (#9756) | skip (conflict → pass 2) |
| 3 | openai | b4e20bbe5 | release/3.7.x — triton-ext plugin extension updates (#9965) | skip (conflict → pass 2) |
| 4 | openai | b7fa781f9 | Split RemoveLayoutConversions cleanup; tolerate SCF non-convergence (#10174) | skip (conflict → pass 2) |
| 5 | meta-main | 000100c2c | [AutoWS][FA] Fix non-deterministic SIGSEGV in insertAsyncComm (#1895) | pick |
| 6 | meta-main | ed73fee96 | [AutoWS][FA] early-TMA staging + fwd-persistent deadlock fix (#1864) | pick |
| 7 | meta-main | 8cb621e52 | [AutoWS] Adjust failing lit tests for fbtriton (#1945) | pick |
| 8 | meta-main | 857686596 | [AutoWS] TaskIdPropagation: handle unstructured control flow (#1951) | pick |
| 9 | meta-main | 41fbdb444 | [CI] Align llvm-build.yml with upstream #10358 (#1958) | pick |
| 10 | meta-main | 297f1202f | Fix make dev-install-llvm on macOS without lld (#1966) | pick |
| 11 | meta-main | 16dcf4e78 | [ci] disable blackwell_attentions_mxfp8 on h100 and mi350 (#1965) | pick |
| 12 | meta-main | 8259b5ad9 | [TritonNvidiaGPU] Fix 2-CTA TMEM verify crash (#1905) | pick |
| 13 | meta-main | e75dd31fe | [AutoWS] Fix WSDataPartition scf.for slicing (#2002) | pick |
| 14 | meta-main | e8a7e28f8 | Fix tritonbench blackwell_attentions CI (#2004) | pick |
| 15 | meta-main | 766ba4303 | [TLX] Lower storage-alias after layout propagation for 2-CTA TMEM (#1988) | pick |
| 16 | meta-main | 5a183059e | [AutoWS] Skip terminators in reorderEpilogOps (#2006) | pick |
| 17 | meta-main | 0900afe83 | Build C dispatcher before it is read on JITFunction.run (#1973) | pick |
| 18 | meta-main | aaa4ecb6d | [bitequiv] exclude corpus kernels from OSS pre-commit yapf (#2034) | pick |
| 19 | meta-main | 4f2446760 | Fix tritonbench blackwell_attentions_mxfp8 CI (#2041) | pick |
| 20 | meta-main | f12244641 | [CI] Bump TLX tutorial correctness test timeout 5→15 min (#2029) | skip (conflict → pass 2) |
| 21 | meta-main | 6097c079e | Re-sync test_line_info FileCheck line numbers (#2043) | pick |
| 22 | meta-main | 9ac5568a6 | [TLX][AMD] Document direct-to-LDS requirements for buffer_load_to_local (#2035) | pick |
| 23 | meta-main | 56b4c7aaa | Gate NVIDIA-only beta unit tests correctly on AMD (#2050) | pick |
| 24 | meta-main | da3dd080a | Derive NVIDIA launch signature from object signature (#2045) | pick |
| 25 | meta-main | 23f02c160 | [TLX FA BWD] Fix intra-task warp race on aliased TMEM in 2-CTA (#1992) | pick |
| 26 | meta-main | d52e8e93e | [triton] Move launch.h to the nvidia backend dir (#1816) | pick (reclassified skip→pick: launch.h crash fix) |
| 27 | release/3.7.0 | 45a6f9d50 | fix gluon | pick |
| 28 | release/3.7.0 | 091d450cf | [AutoWS] Adjust failing lit tests for fbtriton (#1945) | drop (empty; same patch as 8cb621e52) |

## Pass 2 — resolve the skipped commits (resolve to match Meta main)

| # | source | sha | commit title | pick/skip | resolution |
|---|--------|-----|--------------|-----------|------------|
| 1 | openai | b7fa781f9 | Split RemoveLayoutConversions cleanup (#10174) | pick | Resolved to match Meta main, then **swapped to the upstream original `ca21b1b95` (#10132)** for honest provenance. **Applied.** |
| 2 | openai | 5483c9b95 | Revert use_async_copy default (#9445) | skip | Initially applied, then **REJECTED** (dropped) — superseded fallback; the real fix (#9431) is already present. Same class as #9756/#9965. See note below. |
| 3 | openai | 7e48d5dfc | Revert ptxas SM89 workaround (#9756) | skip | **Rejected.** Meta main still keeps the workaround, and the fork evolved this store-lowering with barrier support; applying would diverge from main + risk clobbering the barrier-store path. SM89 is non-primary. |
| 4 | meta-main | f12244641 | Bump TLX tutorial test timeout (#2029) | pick | Resolved (**reduced**): kept h100.yml timeout 5→15 + .pre-commit-config bitequiv exclude; dropped absent-path (b200.yml + 2 new test .mlir) and cosmetic .cpp lint-autofix. **Applied.** |
| 5 | openai | b4e20bbe5 | triton-ext plugin extension updates (#9965) | skip | **Skipped — fork adaptation needed + low value** (*not* because it's "absent from main"; Meta main lags OpenAI, so that alone isn't disqualifying). Meta deliberately keeps pybind11 out of the non-python `TritonTools` lib: `PluginUtils.h` forward-declares `TritonOpBuilder` rather than including `python/src/ir.h` (the #9748 adaptation, bundled in `81cb42793` / #1985). A triton-ext plugin extension off OpenAI's release branch must preserve that isolation or it breaks the beta buck build — and #9965 isn't in our repo to diff cleanly. Given low-med value for this release, skip. If wanted later: re-apply the pybind11-isolation adaptation and validate the buck build (ideally on main first, then bundle). |

Net: applied `b7fa781f9`(→`ca21b1b95`), `f12244641`(reduced); rejected `7e48d5dfc`, `b4e20bbe5`, and **`5483c9b95` (#9445)** (rejected post-hoc — superseded fallback; see note below).
Final branch order (after reorder): fix gluon → 20 clean picks → #1816 → #10132 (ca21b1b95) → TLX timeout.

## Rejected post-hoc — #9445 use_async_copy revert (`5483c9b95` / applied as `9b6640b1f`)

**Should have been rejected in pass 2; caught and dropped after the fact.** This is a
triage-classification miss, not a code bug — #9445 belongs in the same "reject: OpenAI
release-branch revert of something Meta main deliberately keeps" bucket as #9756 and #9965.

**Effect of the pick.** The revert set `is_async_copy_enabled(arch)` to plain
`return knobs.amd.use_async_copy`, defaulting async-copy to **off** on gfx950/gfx1250.
That regresses MI350 (gfx950): the AMD Flash-Attention pipelined kernel needs the
async-copy / direct-to-LDS path, and with it off the `ttg.async_copy_global_to_local`
ops don't lower, leaving a dangling `builtin.unrealized_conversion_cast` →
`RuntimeError: failed to translate module to LLVM IR`. The conflict resolution masked
this: because main already had #9087's other four files, the revert collapsed to a single
compiler.py line, so it looked like a tiny clean pick.

- **Symptom:** CI `mi350-tlx-test` — `test_amd_fa_pipelined[amd_fa_pipelined-True]` (and
  the other AMD FA/addmm cases) fail to compile.
- **Bisect (confirmed):** on the wheel built from this branch, the test **fails** by default
  and **passes** with `TRITON_HIP_USE_ASYNC_COPY=1` (i.e. async-copy forced back on).
- **Superseded fallback — the decisive point.** Per its own commit message, #9445 was a
  *provisional fallback* OpenAI added "in the hopes of fixing the many rocm failures,"
  explicitly preferring the real fix **#9431** ("[Backend] Bump to llvm/llvm-project@979132a")
  and keeping the revert only "in case LLVM upgrades cause other issues." **#9431 is already
  present in both `release/3.7.x` and this branch** (`8e47abe9e`/`8e562c8c1` reachable; our
  LLVM pin `62b7cf96…` is on the post-#9431 lineage), so the failure #9445 guarded against is
  already resolved. Keeping #9445 would disable async-copy *and* discard the fix that made
  disabling unnecessary.
- **Alignment.** Meta main keeps async-copy **on** for gfx950/gfx1250 (cherry-picked upstream
  #9087 as #1335 `8e47abe9e`); upstream `main` direction is also on. Dropping #9445 matches
  Meta main, upstream's direction, and OpenAI's own stated preference (the fix, not the revert).
- **Action:** removed `9b6640b1f` from the branch via `git rebase --onto 1dd6e2d07 9b6640b1f`.
  `is_async_copy_enabled` is back to the arch-default (async-copy on for gfx950/gfx1250),
  matching Meta main. Empirically the AMD FA kernels compile and pass on gfx950 with it on.
  The SM89 revert (#9756) and triton-ext bundle (#9965) remain rejected.
- **Process gap:** run the gfx950/MI350 correctness gate
  (`test_correctness.py -k "amd or ikbo"`, i.e. the `mi350-tlx-test` CI) before finalizing a
  backport so an out-of-context flag flip like this is caught pre-merge.

## Raw candidate universe (triage input)

All 118 commits considered, verbatim from `candidates.tsv`, each with its **pick / drop reason**
(23 pick, 95 drop). "Already in `release/3.7.x`" is surfaced as a drop reason — that's the
pre-backport base, so it never matches our own fresh picks. Detection: **meta-main by SHA ancestry** (real
commits in our repo — none were already present; they're the delta), **openai by PR-number match** against
`release/3.7.x` history (their SHAs aren't in our repo). **16 openai candidates were already in release** and
needed no pick — the "Meta main lags OpenAI, but the bundles already carry most of it" effect. 15 are caught
automatically by PR-number; the 16th (#9621) wraps two commits — #9450 and #9499 — that are bundled under
their own numbers (see its row), so it's redundant too. No rows are left unresolved. The definitive
picks/rejects are the Pass 1 / Pass 2 tables above; this section is the full input + verdict.

### openai (release/3.7.x source)
| sha | subject | reason |
|-----|---------|--------|
| 5483c9b95 | [Release] Revert enabling use_async_copy by default (revert https://github.com/triton-lang/triton/pull/9087) (#9445) | **drop** — rejected: superseded fallback (#9445; see note) |
| 731540ba9 | [Backend] Bump to llvm/llvm-project@979132a (#9431) (#9477) | **drop** — already in `release/3.7.x` (bundled, #9431 in `094868b73`) |
| fce6b8da7 | [AMD] Disable True16 for assembler on gfx11 (#9447) (#9476) | **drop** — already in `release/3.7.x` (bundled, #9447 in `91bc8b3b3`) |
| 5d72932fc | [AMD][BACKEND] Cherry pick pr 9487 to rel 3.7 (#9502) | **drop** — already in `release/3.7.x` (bundled, #9487 in `f9fd9377c`) |
| 3dfa9e303 | [Revert][BACKEND] Create llvm.store when we do not need predication  (#9601) | **drop** — revert not needed here |
| 2c3dac9db | [CUDA][rel/3.7] Cherrypick two commits that are necessary for PyTorch CUDA Binaries especially for GB300/Spark/THOR (#9621) | **drop** — already in `release/3.7.x` (its 2 commits: #9450 PTX-min-version bundled in `799ec1265`, #9499 CUtensorMap align in `3e5305389`) |
| 07bcc9a68 | Revert "[Revert][BACKEND] Create llvm.store when we do not need predication " (#9672) | **drop** — revert not needed here |
| 9be634c2e | Revert "[AMD][BACKEND] Cherry pick pr 9487 to rel 3.7" (#9673) | **drop** — already in `release/3.7.x` (bundled, #9487 in `f9fd9377c`) |
| 151e3e2f3 | Reland "[AMD][BACKEND] Cherry pick pr 9487 to rel 3.7" (#9675) | **drop** — already in `release/3.7.x` (bundled, #9487 in `f9fd9377c`) |
| 3bed8f599 | Fix: emit a deprecation warning for make_block_ptr (#9667) (#9738) | **drop** — already in `release/3.7.x` (bundled, #9667 in `183108f09`) |
| 7e48d5dfc | [Nvidia] Revert the PR https://github.com/triton-lang/triton/pull/7067 for working around a ptxas bug on SM89 (#9756) | **drop** — rejected: diverges from main (#9756) |
| 9c288bc5e | Advance version 3.6.0->3.7.0 (#9885) (#9888) | **drop** — already in `release/3.7.x` (bundled, #9885 in `6440b589d`) |
| 282c8251e | Revert "[Backend] Bump to llvm/llvm-project@979132a (#9431) (#9477)" (#9942) | **drop** — already in `release/3.7.x` (bundled, #9431 in `094868b73`) |
| 64b87cf81 | [release/3.7.x][AMD][BACKEND] Fix RangeAnalysis tripCount calculation (#9383) (#9944) | **drop** — already in `release/3.7.x` (bundled, #9383 in `68b71c789`) |
| 2f5ab8c45 | [release/3.7.x][AMD] Fix OOM in pipelining with padded layout async copy on GFX950 (#9442) (#9945) | **drop** — already in `release/3.7.x` (bundled, #9442 in `f9fd9377c`) |
| 74d18477b | [release/3.7.x][AMD] Fix BlockPingpong for non-MFMA dot (#9618) (#9948) | **drop** — already in `release/3.7.x` (bundled, #9618 in `ff9fcabf9`) |
| 3a64615a5 | [release/3.7.x] Enable TRITON_EXT_ENABLED for Wheels Build (#9935) (#9959) | **drop** — release plumbing (we own it) |
| 98a34c588 | [release/3.7.x] [AMD] CanonicalizePointers: Handle different base pointers and offsets (#9541) (#9950) | **drop** — already in `release/3.7.x` (bundled, #9541 in `33de58083`) |
| b4e20bbe5 | release/3.7.x (#9965) | **drop** — skip: fork adaptation + low value (#9965) |
| 6520a28dc | [release/3.7.x][AMD][BACKEND] Fix mixed types MFMA fp8 instruction selection (#9567) (#9946) | **drop** — already in `release/3.7.x` (bundled, #9567 in `ff9fcabf9`) |
| 88b227e23 | [release/3.7.x][AMD][BACKEND] Fix mixed FP8 types promotion for WMMA (#9581) (#9947) | **drop** — already in `release/3.7.x` (bundled, #9581 in `ff9fcabf9`) |
| b7fa781f9 | [release/3.7.x] Cherry pick "Split RemoveLayoutConversions cleanup so scf.if non-convergence is non fatal" (#10174) | **pick** → applied as #10132 (`ca21b1b95`) in pass 2 |
| 8a1faba46 | [release/3.7.x][CI] Reduce wheel size and pin DOCKER_API_VERSION (#10244) | **drop** — release plumbing (we own it) |
| ecbbf0e03 | [CD][Release Only] Increase timeout for Release wheels (#10250) | **drop** — release plumbing (we own it) |
| 5f3f125e8 | [release 3.7] Release Triton to pypi (#10251) | **drop** — release plumbing (we own it) |
| 9c610c781 | [release/3.7.x] Include examples/ in source distribution (#10350) | **drop** — release plumbing (we own it) |
| 6970b11e4 | [Release-only] Bump version to 3.7.1 release, turn off publishing to pypi, and cherry-pick wheels.yml cibuildwheel fix (#10547) (#10546) | **drop** — release plumbing (we own it) |
| 5d6048aa0 | [Release 3.7.x] Cherry-pick "[FenceAsync] Add async read dependencies to the pass (#9610)" (#10537) | **drop** — already in `release/3.7.x` (bundled, #9610 in `442f08f36`) |
| f797708c0 | [Release/3.7] Pin LLVM to triton-lang/llvm-project@1f126a6 and re-enable PyPI publishing (#10631) | **drop** — release plumbing (we own it) |

### meta-main
| sha | subject | reason |
|-----|---------|--------|
| 0dbb1428e | [AutoWS] Support dynamic persistent GEMM in AutoWS (#1879) | **drop** — not selected (meta-main; fixes-only rule) |
| 000100c2c | [AutoWS][FA] Fix non-deterministic SIGSEGV in insertAsyncComm channel-loop detection (#1895) | **pick** |
| 65e0ed083 | sched2tlx: case4 FA backward end-to-end + intra-WG barrier completeness (#1925) | **drop** — not selected (meta-main; fixes-only rule) |
| ddba7515f | [triton][beta] [AMD] Decompose+modulo scheduling driver (gated, uses AMDLatencyModel) (#1884) | **drop** — not selected (meta-main; fixes-only rule) |
| d3be00c80 | [triton][beta] [AMD][TTGIR-SCHED] APPLY: M/N tt.dot decomposition + sched.barrier (+4-5%) (#1937) | **drop** — not selected (meta-main; fixes-only rule) |
| a49ceba46 | [BE] [AutoWS] Collapse dump-intermediate-steps boilerplate (#1938) | **drop** — not selected (meta-main; fixes-only rule) |
| ed73fee96 | [AutoWS][FA] early-TMA staging slot/reuse (T277224987) + fwd-persistent deadlock fix (#1864) | **pick** |
| e1bb0ee29 | [torchTLX] move codebase to triton-beta / fbtriton (#1926) | **drop** — not selected (meta-main; fixes-only rule) |
| 8cb621e52 | [AutoWS] Adjust failing lit tests for fbtriton (#1945) | **pick** |
| d1adc6a52 | [sched2tlx] case1 GEMM: build hand-written TMA descriptors on-device (#1943) | **drop** — not selected (meta-main; fixes-only rule) |
| c32d385a6 | Add README for torchTLX (#1949) | **drop** — not selected (meta-main; fixes-only rule) |
| 26ef36914 | [BE] [AutoWS] Cleanup doTMAStoreWaitReorder API (#1936) | **drop** — not selected (meta-main; fixes-only rule) |
| be5e1fba8 | [AutoWS] [CLC] Add support for CLC APIs in Triton (#1881) | **drop** — not selected (meta-main; fixes-only rule) |
| b21d29c46 | [TLX] Add tlx.dump_layout compile-time layout diagnostic (#1955) (#1955) | **drop** — not selected (meta-main; fixes-only rule) |
| 628e19018 | [ci] add b200 runner (#1934) | **drop** — not selected (meta-main; fixes-only rule) |
| acb8c6d97 | OSS cross-attention fwd+bwd kernels, benchmark & repro harness (#1950) | **drop** — not selected (meta-main; fixes-only rule) |
| 857686596 | [AutoWS] TaskIdPropagation: handle unstructured control flow (cf.cond_br) (#1951) | **pick** |
| 640155ba8 | [AutoWS][HSTU] reduce_dq: release outer-loaded k/v EMPTY only on last inner iter (#1952) | **drop** — not selected (meta-main; fixes-only rule) |
| 4a23c1963 | [TLX][HSTU] Port attn_bwd_ws_2kv (2-KV-block data-partitioned reduce_dq) (#1953) | **drop** — not selected (meta-main; fixes-only rule) |
| 910ce4e40 | [TLX][HSTU] Add BwdVariant.TLX_2KV + shared-KV bench integration (#1954) | **drop** — not selected (meta-main; fixes-only rule) |
| 41fbdb444 | [triton][beta] [CI] Align llvm-build.yml with upstream #10358 (fix AlmaLinux numpy + clang 21.1.8 ICE) (#1958) | **pick** |
| e4d12400e | [triton][beta] [AMD] Modulo scheduling Steps 4.7+4.8: warp-pipeline partition + s_setprio (#1960) | **drop** — not selected (meta-main; fixes-only rule) |
| 297f1202f | [triton][beta] [Cherry-pick] 'Fix make dev-install-llvm on macOS without lld (#10636)' (#1966) | **pick** |
| 16dcf4e78 | [ci] disable blackwell_attentions_mxfp8 on h100 and mi350 (#1965) | **pick** |
| e8260a8c8 | [triton][beta] [Cherry-pick][BUNDLE] Cherry-pick 56 upstream(#9890 #9779 #9881 #9911 #9908 #9899 #9903 #9913 #9918 #9921 #9909 #9910 #9920 #9880 #9925 #9895 #9926 #9905 #9892 #9768 #9791 #9930 #9943 #9935 #9493 #9966 #9960 #9953 #9983 #9981 #9971... (#1956) | **drop** — not selected (meta-main; fixes-only rule) |
| d52e8e93e | [triton] Move launch.h to the nvidia backend dir (#1816) | **pick** (#1816 launch.h crash fix) |
| f4c21765f | [AutoWS] [CLC] Add support for CLC GEMM in Triton AutoWS (#1882) | **drop** — not selected (meta-main; fixes-only rule) |
| 8259b5ad9 | [TritonNvidiaGPU] Fix 2-CTA TMEM verify crash (#1905) | **pick** |
| 0b9086299 | Add fundamental Gluon frontend CI coverage on B200 and MI350 (#1972) | **drop** — not selected (meta-main; fixes-only rule) |
| b47f67efb | [torchTLX] Rename TLX enablement knob value `default` -> `None` (#1967) | **drop** — not selected (meta-main; fixes-only rule) |
| 9f609cfcb | [triton][beta] [Cherry-pick][BUNDLE] Cherry-pick 12 upstream(#9992 #10046 #10039 #9975 #10051 #9919 #10048 #10047 #10038 #10065 #10057 #10059) (#1981) | **drop** — not selected (meta-main; fixes-only rule) |
| 9215502ad | Add auto-bisection + external-dep triage to nightly issue-filing (#1983) | **drop** — not selected (meta-main; fixes-only rule) |
| 4ea10f6e5 | sched2tlx: mark schedule_graph.json/ddg.json dumps as @generated (#1987) | **drop** — not selected (meta-main; fixes-only rule) |
| c36fc2a74 | [TLX][AMD] addmm+GLU: fuse bias+GLU epilogue with register-resident Y prefetch (#1963) | **drop** — not selected (meta-main; fixes-only rule) |
| 79fce0178 | [torchTLX] Add unit tests for templates and Inductor fusion (#1968) | **drop** — not selected (meta-main; fixes-only rule) |
| 889490da2 | [TritonGPU] NPOT elementwise lowering via modular LinearLayout (#1892) | **drop** — not selected (meta-main; fixes-only rule) |
| 8f91ef011 | [sched2tlx] move regression harness; add case5/case7 bench specs (#1939) | **drop** — not selected (meta-main; fixes-only rule) |
| 81cb42793 | [triton][beta] [Cherry-pick][BUNDLE] Beta backport bundle: OptimizeDescriptorEncoding + mxfp/matmul + GSan + DSL plugins (#9709 and 13 more) (#1985) | **drop** — not selected (meta-main; fixes-only rule) |
| fdd126bae | [Modulo Scheduling] Honest latency model, partition-aware cost model, and emitter capabilities to match (#1912) (#1912) | **drop** — not selected (meta-main; fixes-only rule) |
| ae960c0d6 | [BE] [AutoWS] Refactor WarpSpecialization intra-pass declarations to a header (#1940) | **drop** — not selected (meta-main; fixes-only rule) |
| 961212ade | [triton][beta] [Cherry-pick][BUNDLE] Cherry-pick 37 upstream(#10066 #10060 #10083 #9936 #10062 #10055 #10082 #10074 #10076 #10091 #10077 #10030 #10093 #10095 #10097 #10090 #10004 #10101 #10098 #9865 #10110 #10111 #10114 #10112 #9850 #10109 #10119 #10117 # (#2000) | **drop** — not selected (meta-main; fixes-only rule) |
| ed8a245e7 | [AutoWS] Enable standard scheduling definition (#1885) | **drop** — not selected (meta-main; fixes-only rule) |
| e75dd31fe | [AutoWS] Fix WSDataPartition scf.for slicing to keep results/yield 1:1 (#2002) | **pick** |
| e8a7e28f8 | Fix tritonbench blackwell_attentions CI (seq>=256; skip broken upstream FA2) (#2004) | **pick** |
| 092678472 | [triton][bitequiv] diverse realistic-Inductor kernel corpus (#1976) | **drop** — not selected (meta-main; fixes-only rule) |
| 0782e714d | [triton][PR] [TLX] Add CuTe-style tlx.swizzled_layout(B, M, S) for swizzled shared (#1994) | **drop** — not selected (meta-main; fixes-only rule) |
| 8be187601 | [TLX] blackwell_gemm_ws: async tcgen05_commit epilogue signal to close the inter-tile MMA bubble (#2008) | **drop** — not selected (meta-main; fixes-only rule) |
| a695fe56f | [AutoWS][HSTU cross] Manual 2-KV reduce_dq kernel + compute fold (gated, dormant) (#2014) | **drop** — not selected (meta-main; fixes-only rule) |
| 733773c78 | [AutoWS][HSTU] Coalesce chained MMAv5 accumulators in OptimizeAccumulatorInit (#2015) | **drop** — not selected (meta-main; fixes-only rule) |
| 993b34bcd | [ci] disable torchtlx on h100 (#2010) | **drop** — not selected (meta-main; fixes-only rule) |
| 949aa3efa | [CI] Checkout linked TritonBench PR when meta-triton PR references it (#2011) | **drop** — not selected (meta-main; fixes-only rule) |
| 766ba4303 | [TLX] Lower storage-alias after layout propagation for 2-CTA TMEM (#1988) | **pick** |
| 8b13d5a06 | [triton] denoise.sh: add GB300 (1400W) and default unknown NVIDIA GPUs to rated power (#2023) | **drop** — not selected (meta-main; fixes-only rule) |
| 9273a3524 | [AutoWS] handleOperandD: support chained accumulator (shared opndD TMEM tile) (#2018) | **drop** — not selected (meta-main; fixes-only rule) |
| 5d250cdda | [AutoWS][PSM] Group accumulator-chained MMAs into one data partition (#2019) | **drop** — not selected (meta-main; fixes-only rule) |
| 63b28b73f | [AutoWS][CP] Collapse chained operand-D accumulator to one forward channel (#2020) | **drop** — not selected (meta-main; fixes-only rule) |
| bfd86bb22 | [AutoWS][PSM] Co-locate reduce_dq store token-wait with its reduce (#2021) | **drop** — not selected (meta-main; fixes-only rule) |
| 187888830 | [AutoWS][HSTU cross] test: manual 2-KV (autows_2kv) + tlx_2kv correctness coverage (#2022) | **drop** — not selected (meta-main; fixes-only rule) |
| 82ce873ef | [ModuloSchedule] Add TRITON_MODULO_SELECT_VARIANT to lower a chosen partition variant (#2028) | **drop** — not selected (meta-main; fixes-only rule) |
| d40127f3f | [Modulo Scheduling] case8: multi-phase GEMM — first corpus case exercising cross-phase SMEM buffer reuse (#1998) | **drop** — not selected (meta-main; fixes-only rule) |
| 5a183059e | [AutoWS] Skip terminators in reorderEpilogOps (#2006) | **pick** |
| 8dcdcde51 | [ptx-anneal] Relocate + rename triton compile_iq -> triton.magnon; delete third_party/compile_iq (#1979) | **drop** — not selected (meta-main; fixes-only rule) |
| 0900afe83 | [triton][beta] Build C dispatcher before it is read on the JITFunction.run path (#1973) | **pick** |
| 8203dc3bc | [AutoWS] [BE] Collapse defaultNumStages alias (#2027) | **drop** — not selected (meta-main; fixes-only rule) |
| e95d544db | [torchTLX] AMD FlexAttention Templates (gfx950/MI350) (#2003) | **drop** — not selected (meta-main; fixes-only rule) |
| aaa4ecb6d | [triton][bitequiv] exclude corpus kernels from OSS pre-commit yapf (#2034) | **pick** |
| 568e791a6 | [triton][bitequiv] Suite 2: make realistic Inductor GROUP 1 runnable (merge complex_fusion_eval) (#2012) | **drop** — not selected (meta-main; fixes-only rule) |
| 4f2446760 | Fix tritonbench blackwell_attentions_mxfp8 CI (seq>=256) (#2041) | **pick** |
| 5bc6c8a01 | [triton][beta] [Cherry-pick] '[Blackwell] [Gluon] Support Blackwell 256-bit global load/store LLVM lowering (#10861)' (#2025) | **drop** — not selected (meta-main; fixes-only rule) |
| f12244641 | [CI] Bump TLX tutorial correctness test timeout 5→15 min (b200, h100) (#2029) | **pick** (reduced) in pass 2 |
| 6698a6a84 | [AutoWS][ModuloSched] Standalone list scheduler: reorder + top-K/beam + per-loop pick | **drop** — not selected (meta-main; fixes-only rule) |
| 6097c079e | [triton] Re-sync test_line_info FileCheck line numbers (#2043) | **pick** |
| 9ac5568a6 | [TLX][AMD] Document direct-to-LDS requirements for buffer_load_to_local (#2035) | **pick** |
| 59283bfd7 | [TLX][AMD] Add paged-attention decode kernel (#2040) | **drop** — not selected (meta-main; fixes-only rule) |
| f46e74bab | [triton][bitequiv] Suite 1: consolidated hand-written eval kernels (M1 + M3 GEMM + M2 layouts) (#2013) | **drop** — not selected (meta-main; fixes-only rule) |
| 56b4c7aaa | [triton] Gate NVIDIA-only beta unit tests correctly on AMD (#2050) | **pick** |
| ab7579836 | [triton][beta] [Cherry-pick][BUNDLE] Cherry-pick 42 upstream(#10124 #10144 #10151 #10149 #10064 #10147 #10162 #10156 #10150 #10126 #10120 #10099 #9641 #10132 #10175 #10081 #10179 #10128 #10100 #10178 #10125 #10186 #10180 #10193 #9093 #10191 #10185... (#2031) | **drop** — not selected (meta-main; fixes-only rule) |
| da3dd080a | [triton] Derive NVIDIA launch signature from object signature, not str(ty) (#2045) | **pick** |
| 23f02c160 | [TLX FA BWD] Fix intra-task warp race on aliased TMEM in the 2-CTA path (#1992) | **pick** |
| 5b19662d1 | [TLX] Respect user-pinned layouts via #tlx.user_layout wrapper (#2017) | **drop** — not selected (meta-main; fixes-only rule) |
| cf4db2746 | [triton][auto-TMA][1/N] launcher recipe ABI (host-built CUtensorMap) (#1975) | **drop** — not selected (meta-main; fixes-only rule) |
| 76d0bf520 | [TLX] Add warp-specialized FP8 scaled_mm kernel for Blackwell (#1964) | **drop** — not selected (meta-main; fixes-only rule) |
| 53bbc9412 | [sched2tlx] case9: FP8 blockwise scaled_mm + first-class fp8 emitter support (#2053) | **drop** — not selected (meta-main; fixes-only rule) |
| 4e2cc48bb | [Triton][auto-TMA][2/N] PromoteLoadToTMA pass -- host-recipe LOAD promotion (#1977) | **drop** — not selected (meta-main; fixes-only rule) |
| 321c0f667 | [TLX][AMD] Fix warp-pipe addmm RAW race in async_load pipeline (unblocks all shapes) (#2058) | **drop** — not selected (meta-main; fixes-only rule) |
| 23689abed | [sched2tlx] Add perf_harness compare subcommand + harness usage skill (#2057) | **drop** — not selected (meta-main; fixes-only rule) |
| 3915b0ee8 | [triton][beta] [Cherry-pick] '[Blackwell] Add pass to combine TMEM load followed by row reduction for sm103+ (#10551)' (#2142) | **drop** — not selected (meta-main; fixes-only rule) |
| 697baf4ca | [Triton][auto-TMA][3/N] host-recipe STORE promotion (#1978) | **drop** — not selected (meta-main; fixes-only rule) |
| 2054eb494 | [Triton/TLX] Lower overhead Warp Spec LLVM lowering (#2054) | **drop** — not selected (meta-main; fixes-only rule) |
