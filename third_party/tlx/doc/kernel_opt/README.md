# Curated kernel-optimization knowledge

One file per GPU architecture. These files are the human-curated prior that the
kernel-optimization agent
(`third_party/tlx/tools/agents/kernel_optimization/`) is given before it
proposes anything, so that a round starts from what we already know instead of
from blind exploration.

| File | Target |
|---|---|
| `gfx942.md` | CDNA3 / MI300X |

## Contract

- **Human-write-only.** The agent reads these files; it never edits them. Agent
  findings land in `experiments/` under the run's output dir and are promoted
  into a file here only by a human who has read the evidence.
- **Every entry carries its evidence and its provenance.** A claim with no
  number behind it does not belong here.
- **Measured-on-this-arch and ported-from-another-arch are separate sections.**
  A CDNA4 result is a hypothesis for CDNA3, not a fact about it, and mixing the
  two is how an agent gets confidently wrong. Moving an entry up from the
  ported section requires a measurement on the target part.
- **Concise beats complete.** This text is injected into a prompt. An entry that
  does not change what a candidate would do is costing tokens for nothing.

## How it is consumed

A harness under `harnesses/<arch>/` reads the file for its own arch and the
provider injects it into the candidate prompt preamble. Nothing else parses
these files, so prose is fine; keep the section headings stable.
