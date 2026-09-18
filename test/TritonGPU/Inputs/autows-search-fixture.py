import json
import os
from pathlib import Path

schedule_topk = int(os.environ["TRITON_MODULO_TOPK"])
schedule_pick = min(int(os.environ["TRITON_MODULO_PICK"]), schedule_topk - 1)
memory_space_topk = int(os.environ["TRITON_WS_MEMORY_SPACE_TOPK"])
memory_space_pick = min(int(os.environ["TRITON_WS_MEMORY_SPACE_PICK"]), memory_space_topk - 1)
smem_topk = int(os.environ["TRITON_WS_SMEM_PLAN_TOPK"])
smem_pick = min(int(os.environ["TRITON_WS_SMEM_PLAN_PICK"]), smem_topk - 1)
tmem_topk = int(os.environ["TRITON_WS_TMEM_PLAN_TOPK"])
tmem_pick = min(int(os.environ["TRITON_WS_TMEM_PLAN_PICK"]), tmem_topk - 1)
manifest = Path(os.environ["TRITON_WS_SEARCH_MANIFEST"])

with manifest.open("a") as output:
    for rank in range(schedule_topk):
        print(
            json.dumps({
                "kind": "schedule",
                "rank": rank,
                "selected": rank == schedule_pick,
                "ii": 4 + rank,
                "signature": [rank],
            }),
            file=output,
        )
    # A second loop with no promotable ambiguity may emit a selected fallback
    # rank zero. The driver must ignore that no-op record when this loop emits a
    # real memory-space frontier.
    print(
        json.dumps({
            "kind": "memory-space",
            "rank": 0,
            "selected": True,
            "candidate_count": 0,
            "lhs_tmem": [],
        }),
        file=output,
    )
    for rank in range(memory_space_topk):
        print(
            json.dumps({
                "kind": "memory-space",
                "rank": rank,
                "selected": rank == memory_space_pick,
                "candidate_count": 1,
                "lhs_tmem": [] if rank == 0 else [0],
            }),
            file=output,
        )
    for pool, topk, pick in (("smem-fixed", smem_topk, smem_pick), ("tmem", tmem_topk, tmem_pick)):
        for rank in range(topk):
            print(
                json.dumps({
                    "kind": "memory",
                    "schedule_pick": schedule_pick,
                    "pool": pool,
                    "rank": rank,
                    "selected": rank == pick,
                    "blocks": [{"id": 0, "copy": rank + 1}],
                }),
                file=output,
            )
    print(
        json.dumps({
            "kind": "validation",
            "status": "safe",
            "supported_channels": 3,
            "unsupported_channels": 0,
        }),
        file=output,
    )

print(f"latency_ms={1000 * schedule_pick + 100 * memory_space_pick + 10 * smem_pick + tmem_pick + 0.5}")

if schedule_pick == int(os.environ.get("AUTOWS_SEARCH_FIXTURE_REJECT_SCHEDULE", "-1")):
    print("candidate rejected by validator")
    raise SystemExit(3)
