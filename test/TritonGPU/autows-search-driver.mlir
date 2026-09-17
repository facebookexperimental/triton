// RUN: python3 %S/../../python/triton/tools/autows_search.py \
// RUN:   --schedule-topk=2 --memory-topk=3 --results=%t \
// RUN:   --metric-regex='latency_ms=([0-9.]+)' -- \
// RUN:   python3 %S/Inputs/autows-search-fixture.py
// RUN: FileCheck %s --input-file=%t

// The driver discovers two schedule ranks, discovers three memory ranks for
// each selected schedule, and evaluates the complete bounded product.
// CHECK: "memory_rank": 0, "metric": 0.5, "returncode": 0, "schedule": [{"ii": 4, "kind": "schedule", "rank": 0, "selected": true, "signature": [0]}], "schedule_rank": 0, "status": "passed"
// CHECK-NEXT: "memory_rank": 1, "metric": 1.5, "returncode": 0, "schedule": [{"ii": 4, "kind": "schedule", "rank": 0, "selected": true, "signature": [0]}], "schedule_rank": 0, "status": "passed"
// CHECK-NEXT: "memory_rank": 2, "metric": 2.5, "returncode": 0, "schedule": [{"ii": 4, "kind": "schedule", "rank": 0, "selected": true, "signature": [0]}], "schedule_rank": 0, "status": "passed"
// CHECK-NEXT: "memory_rank": 0, "metric": 10.5, "returncode": 0, "schedule": [{"ii": 5, "kind": "schedule", "rank": 1, "selected": true, "signature": [1]}], "schedule_rank": 1, "status": "passed"
// CHECK-NEXT: "memory_rank": 1, "metric": 11.5, "returncode": 0, "schedule": [{"ii": 5, "kind": "schedule", "rank": 1, "selected": true, "signature": [1]}], "schedule_rank": 1, "status": "passed"
// CHECK-NEXT: "memory_rank": 2, "metric": 12.5, "returncode": 0, "schedule": [{"ii": 5, "kind": "schedule", "rank": 1, "selected": true, "signature": [1]}], "schedule_rank": 1, "status": "passed"
