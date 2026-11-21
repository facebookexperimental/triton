#!/bin/bash
# RETIRED batch. The automatic TLX emitter arm is no longer part of the
# literal pipeline; this file stays only so the verb fails loudly instead of
# silently doing something else.
#
# The paper's own workflow is an expert manual implementation ("used by an
# expert as reference for a manual implementation"), so the literal branch
# ends at the paper artifacts: run_main_cases.sh emits the pipelined IR
# (paper-joint-pipelined-ir-v3) and the handoff manifest
# (paper-joint-handoff-v2), and a human compiles from those.
set -euo pipefail

cat >&2 <<'MESSAGE'
RETIRED: the automatic TLX emitter arm is retired from the literal pipeline
(see DEVIATIONS.md). The guarded-era version of this batch is preserved at the
pre-literalization baseline D113349666; nothing in the lit1 series reproduces
it. The literal implementation path is: run_main_cases.sh -> paper IR +
handoff manifest -> manual CUDA authoring.
MESSAGE
exit 1
