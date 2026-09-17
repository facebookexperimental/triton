// RUN: python3 %S/../../python/triton/tools/autows_search.py \
// RUN:   --schedule-topk=2 --memory-space-topk=2 --memory-topk=2 --results=%t \
// RUN:   --metric-regex='latency_ms=([0-9.]+)' -- \
// RUN:   python3 %S/Inputs/autows-search-fixture.py
// RUN: FileCheck %s --input-file=%t

// The driver discovers two ranks in each dimension and evaluates all eight
// schedule x memory-space x physical-memory candidates.
// CHECK: "memory_rank": 0, "memory_space": [{"candidate_count": 1, "kind": "memory-space", "lhs_tmem": [], "rank": 0, "selected": true}], "memory_space_rank": 0, "metric": 0.5, {{.*}}"schedule_rank": 0, "status": "passed"
// CHECK-NEXT: "memory_rank": 1, "memory_space": [{"candidate_count": 1, "kind": "memory-space", "lhs_tmem": [], "rank": 0, "selected": true}], "memory_space_rank": 0, "metric": 1.5, {{.*}}"schedule_rank": 0, "status": "passed"
// CHECK-NEXT: "memory_rank": 0, "memory_space": [{"candidate_count": 1, "kind": "memory-space", "lhs_tmem": [0], "rank": 1, "selected": true}], "memory_space_rank": 1, "metric": 10.5, {{.*}}"schedule_rank": 0, "status": "passed"
// CHECK-NEXT: "memory_rank": 1, "memory_space": [{"candidate_count": 1, "kind": "memory-space", "lhs_tmem": [0], "rank": 1, "selected": true}], "memory_space_rank": 1, "metric": 11.5, {{.*}}"schedule_rank": 0, "status": "passed"
// CHECK-NEXT: "memory_rank": 0, "memory_space": [{"candidate_count": 1, "kind": "memory-space", "lhs_tmem": [], "rank": 0, "selected": true}], "memory_space_rank": 0, "metric": 100.5, {{.*}}"schedule_rank": 1, "status": "passed"
// CHECK-NEXT: "memory_rank": 1, "memory_space": [{"candidate_count": 1, "kind": "memory-space", "lhs_tmem": [], "rank": 0, "selected": true}], "memory_space_rank": 0, "metric": 101.5, {{.*}}"schedule_rank": 1, "status": "passed"
// CHECK-NEXT: "memory_rank": 0, "memory_space": [{"candidate_count": 1, "kind": "memory-space", "lhs_tmem": [0], "rank": 1, "selected": true}], "memory_space_rank": 1, "metric": 110.5, {{.*}}"schedule_rank": 1, "status": "passed"
// CHECK-NEXT: "memory_rank": 1, "memory_space": [{"candidate_count": 1, "kind": "memory-space", "lhs_tmem": [0], "rank": 1, "selected": true}], "memory_space_rank": 1, "metric": 111.5, {{.*}}"schedule_rank": 1, "status": "passed"
