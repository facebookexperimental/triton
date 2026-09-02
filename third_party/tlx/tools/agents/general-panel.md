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
| worker | plan | artifacts + CLI verdict | yes | no |
| TL | profiles, tickets | plans, report | **no** | no |
| manager | reports | decision | no | **yes** |

Nothing that reasons may execute; nothing that executes may conclude.

## 2. Curating knowledge: mechanism over measurement

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

## 3. Anti-slop is a gate, not an agent

Enforce mechanically on every panel output rather than asking an agent to be
careful: each claim resolves to a real location, and each cited path exists. The
failure this prevents is real and recent — a knowledge file shipped with five
figures attributed to a docstring that did not contain them, through review.

## 4. Agent definitions are human-owned

Agent and panel definitions are checked in and reviewed as code. **No agent edits
an agent definition**, including its own. Manager and TL definitions in
particular change only by human edit.

## 5. Measurement is serialized

One benchmark at a time, on a pinned GPU, under `denoise.sh`. Not a resourcing
choice: concurrent load perturbs clocks, power and thermals even on another GPU
of the same node, and the promotion gate is a timing comparison. Generation,
compile and correctness may run in parallel (own git worktree, own
`TRITON_CACHE_DIR`); the benchmark step may not.
