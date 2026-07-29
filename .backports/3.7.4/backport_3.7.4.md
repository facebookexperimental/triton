# fbtriton v3.7.4 — backport

A point-release **fix** for the gfx950 async-copy legalization crash (**#2285**) that v3.7.2
and v3.7.3 both missed, plus its regression test. Everything else `main` gained since v3.7.2
is a new feature, already shipped in v3.7.3, or fixes code that isn't on the release branch —
so nothing else was taken.

## Scan range

Swept from the previous release's frontier (v3.7.2's heads) to current head — everything older
was already triaged in the v3.7.2 pick-log.

| Source | frontier → head | commits |
|---|---|---|
| OpenAI `release/3.7.x` | `f797708c0` → `f797708c0` | **0** (unchanged since v3.7.2) |
| Meta `main` | `2054eb494` (07-16, v3.7.2 head) → `f4ce95721` (07-29) | **151** |

Of the 151 meta-main commits (classifier: 42 fix · 21 amd · 6 plumbing · 4 bundle · 78 other),
triage kept **1** (#2285); the rest are features, already in v3.7.3, or absent-path.

## Picked

| PR / commit | What it is | Why picked |
|---|---|---|
| **#2285** (`94c634027`) | `CoalesceAsyncCopy` synchronous fallback (`tt.load` + `local_store`) when a direct-to-LDS async copy collapses below a legal per-thread vector width on CDNA4 (masked partial-K, non-16-aligned stride, or a non-contiguous gather). | The reason for this release: without it, such a copy aborts in `make_llir` with a dangling `unrealized_conversion_cast` (“failed to translate module to LLVM IR”). |
| **PR #2525** (`6f13482`) | bf16 non-contiguous-gather `async_load` regression test (`P2440272260`). | Pins the third trigger + the bf16 path of #2285’s fix; fails on stock 3.7.3, passes with the fix. |

## Not picked

| Group | Commits | Why not |
|---|---|---|
| Already on release (v3.7.3) | #2336, #2392, #2290, #2337, #2157 | AMD-perf work already shipped in v3.7.3. |
| New features | #2282 (FA-bwd), #2156, #2158, #1886, #1893, #1879 / #1882 (CLC), #2146, #2248 | A point release is fixes-only; no new features. |
| Absent-path — fixes code not on release | #2346, #2058 (inductor `amd_addmm_warppipe` template); #2237 (needs NPOT feature #1892); #2270 (needs tensordesc bundle #2234) | The fix’s target code was never backported — there’s nothing on release to fix, and taking it would drag in the feature/bundle it depends on. |
| AutoWS / Blackwell fixes | #2311, #2298, #2307, #2516, #2301 | Fix AutoWS/Blackwell paths that aren’t on the release line. |
| Minor / out of scope | #2344 (async_dot asserts), #2355 (Fa4 bwd spill), #2056 (split-K launch path) | Small correctness/perf tweaks; not needed for this release. |

## Validated

gfx950 wheel builds; TLX correctness suite green (**69 passed / 4 skipped**, no regression vs
v3.7.3); #2285 + gather + row-stride unit tests pass against the wheel.

<sub>Branch also carries the `3.7.4+fb` version bump and backport-tooling updates.</sub>
