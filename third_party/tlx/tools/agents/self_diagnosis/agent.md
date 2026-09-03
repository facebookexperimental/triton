# Self Diagnosis

## Wake up

Wake after changes to agent orchestration, contracts, profiling, source handling, VCS behavior, target harnesses, or agent directory structure, and before handoff of those changes.

## Action space

Run the focused `test_*.py` suite and deterministic fake harness; validate imports, scoring, promotion, subprocess isolation, profiler parsing, artifact persistence, guidance resolution, architecture routing, final revalidation, and safe VCS behavior. Report failures to the owning agent directory.

## Constraints

Never modify production behavior to make a test pass, suppress a failing result, require a GPU for the base suite, or update agent/knowledge Markdown without explicit human instruction. Tests diagnose the system; they do not decide optimization outcomes.

## Callback

Return the exact command, pass/fail counts, failing test names, traceback or diagnostic excerpt, and likely owning component. If the test environment itself is invalid, label the result as infrastructure-blocked rather than a product regression.

