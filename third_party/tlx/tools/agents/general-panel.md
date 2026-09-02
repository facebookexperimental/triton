# General panel — policies every panel inherits

Not a panel of agents. These are the cross-cutting rules the perf-bench, R&D and
build panels are all bound by, kept in one place so they are reviewed once.

## 1. Agents propose, the harness decides

The `kernel_optimization` CLI owns build / verify / benchmark / decide, and that
loop is deterministic on purpose: *"The candidate generator can propose source,
but it cannot declare a candidate correct or faster."* No panel re-measures, or
re-judges a verdict the CLI issued.

Role boundaries follow from it, and are uniform:

| Role | In | Out | May run commands | May conclude |
|---|---|---|---|---|
| profiler | request | raw profiler output | yes | no |
| worker | finding | artifacts + CLI verdict | yes | no |
| TL | profiles, tickets | findings, report | **no** | no |
| manager | reports | decision | no | **yes** |

Nothing that reasons may execute; nothing that executes may conclude.

## 2. Findings and insights are different things

- A **finding** is a proposed change that might improve perf and can be verified
  quickly. Proposed by the TL agent, validated by a worker through the CLI.
  Cheap, high-volume, per-hypothesis. Findings — including the ones that did not
  work — are recorded in the run artifacts automatically. No approval.
- An **insight** is general: one level of abstraction above a finding, true
  beyond the hypothesis that produced it. Recorded by the knowledge agent into
  the bundle's `knowledge.md` / `optimization_guidance.md`. **Every update needs
  human expert review.**
- Findings do not graduate to insights by being true once. A finding becomes an
  insight when it generalizes, and a human says so.

A finding carries a **predicted signal** — the profile quantity the TL expects to
move (a stall class, an occupancy limiter, a traffic figure). The CLI decides
promotion on wall-clock alone; the manager separately compares the post-run
profile against that prediction, giving three outcomes:

| Outcome | CLI verdict | Predicted signal | Feeds the knowledge agent |
|---|---|---|---|
| confirmed | promoted | moved | yes |
| unexplained win | promoted | did not move | **no** |
| rejected | not promoted | either | as a dead end |

An unexplained win still ships — the harness decided, and the manager does not
override it (§1). But it must not assert a mechanism, because the mechanism it
would assert is exactly the one the profile failed to support. "Faster" and
"faster for the predicted reason" are different claims, and only the second is
insight material.

## 3. Curating knowledge: mechanism over measurement

Guidance files under `harnesses/<arch>/` prefer **methodology, logic, structure
and mechanism**. Detailed numbers do not belong in them.

- A figure copied into a prompt goes stale with nothing to notice, and invites a
  candidate to pattern-match on the value instead of the mechanism that produced
  it.
- Cite where the evidence lives — run artifacts, a benchmark suite, a docstring.
  Claims stay evidence-backed; the evidence is referenced, not reproduced.
- `third_party/tlx/language/tlx/hw/resources.py` is **ground truth and never
  edited by an agent**. It holds hardware quantities and executable heuristics
  tune against it. Cite the arch class attribute and record the consequence.
- The knowledge agent's domain is the causal model — how the hardware behaves,
  and what we found — not the quantity table. It may derive from `resources.py`
  freely; it may not fork or edit it.
- Keep established-on-this-arch separate from ported-from-another-arch. A result
  from a neighbouring part is a hypothesis, and the prompt weighs the two
  differently.

## 4. Anti-slop is a gate, not an agent

Enforce mechanically on every panel output rather than asking an agent to be
careful: each claim resolves to a real location, and each cited path exists. The
failure this prevents is real and recent — a knowledge file shipped with five
figures attributed to a docstring that did not contain them, through review.

## 5. Agent definitions are human-owned

Agent and panel definitions are checked in and reviewed as code. **No agent edits
an agent definition**, including its own. Manager and TL definitions in
particular change only by human edit.

## 6. Measurement is serialized

One benchmark at a time, on a pinned GPU, under `denoise.sh`. Not a resourcing
choice: concurrent load perturbs clocks, power and thermals even on another GPU
of the same node, and the promotion gate is a timing comparison. Generation,
compile and correctness may run in parallel (own git worktree, own
`TRITON_CACHE_DIR`); the benchmark step may not.

## 7. The loop closes on the shape set, not on the ticket

A ticket names a shape group, but `tlx.ops` gates on the whole of `_shapes.py`,
shared by the L1 correctness and L2 perf suites. A winner tuned on two shapes can
regress a third, or move which tile wins elsewhere. So a promotion re-enters the
perf-bench panel: the worker re-runs the full shape set, and the manager confirms
no regression before the diff is published. A local win that is a global loss must
not reach review.
