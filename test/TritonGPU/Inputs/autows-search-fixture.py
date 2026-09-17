import json
import os
from pathlib import Path

schedule_topk = int(os.environ["TRITON_MODULO_TOPK"])
schedule_pick = min(int(os.environ["TRITON_MODULO_PICK"]), schedule_topk - 1)
memory_topk = int(os.environ["TRITON_WS_MEM_PLAN_TOPK"])
memory_pick = min(int(os.environ["TRITON_WS_MEM_PLAN_PICK"]), memory_topk - 1)
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
    for rank in range(memory_topk):
        print(
            json.dumps({
                "kind": "memory",
                "schedule_pick": schedule_pick,
                "pool": "smem-fixed",
                "rank": rank,
                "selected": rank == memory_pick,
                "blocks": [{"id": 0, "copy": rank + 1}],
            }),
            file=output,
        )

print(f"latency_ms={10 * schedule_pick + memory_pick + 0.5}")
