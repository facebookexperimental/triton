// RUN: python3 %S/../../python/triton/tools/autows_search.py \
// RUN:   --schedule-topk=2 --memory-space-topk=2 --smem-topk=2 --tmem-topk=2 --results=%t \
// RUN:   --metric-regex='latency_ms=([0-9.]+)' -- \
// RUN:   python3 %S/Inputs/autows-search-fixture.py
// RUN: FileCheck %s --input-file=%t

// The driver discovers two ranks in each dimension and evaluates all sixteen
// schedule x memory-space x SMEM x TMEM candidates.
// CHECK: "memory_space_rank": 0, "metric": 0.5, {{.*}}"schedule_rank": 0, "smem_rank": 0, "status": "passed", "tmem_rank": 0
// CHECK-NEXT: "memory_space_rank": 0, "metric": 1.5, {{.*}}"schedule_rank": 0, "smem_rank": 0, "status": "passed", "tmem_rank": 1
// CHECK-NEXT: "memory_space_rank": 0, "metric": 10.5, {{.*}}"schedule_rank": 0, "smem_rank": 1, "status": "passed", "tmem_rank": 0
// CHECK-NEXT: "memory_space_rank": 0, "metric": 11.5, {{.*}}"schedule_rank": 0, "smem_rank": 1, "status": "passed", "tmem_rank": 1
// CHECK: "memory_space_rank": 1, "metric": 1111.5, {{.*}}"schedule_rank": 1, "smem_rank": 1, "status": "passed", "tmem_rank": 1
