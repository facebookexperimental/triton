# Barrier Analysis
## Motivation
Benefit for both autoWS and TLX for:
- correctness verification
- performance optimization, including barrier insertion and analysis
- developer productivity, including debugging and profiling

## Data Structure
Walk the IR, and construct thw following main data structure for memory variable analysis with associated barriers:
```
class MemoryObject {
    var_name: string = {var_name_in_producer}_TO_{var_name_in_consumer}
    producer_task_id: int
    consumer_task_id: int
    type: string = tmem or smem
    alloc_name: string = {var_name_in_<ttg.local_alloc>}
    offset: int
    producer_bar: BarrierObject
    consumer_bar: BarrierObject
}
class BarrierObject {
    var_name: string = {var_name_in_producer}_TO_{var_name_in_consumer}
    wait_task_id: int // task id where bar.wait happens
    arrive_task_id: int // task id where bar.arrive happens
    type: string = smem or named
    alloc_name: string = {var_name_in_<ttg.local_alloc>}
    offset: int
    memory_object: MemoryObject
}
```

## Algorithm
### Step 1
Traverse all arguments in ttg.warp_specialize, construct two separate lists of operations. One list contains all memory ops created by ttg.local_alloc, and the other list contains all barrier ops created by ttg.local_alloc and initialized by ttng.init_barrier. Find if the variable gets used by `ttng.init_barrier` to discriminate barriers from memory ops. arguments for `ttg.warp_specialize` and its partitions have one to one exact mapping. If it is necessary, make it a map to store `ttg.warp_specialize` args to partition args.
### Step 2
Traverse each partition in `ttg.warp_specialize`. The default partition use arguments from `ttg.warp_specialize`. Other partitions use arguments from their own parititon block. Traverse all `ttg.memdesc_index` operation that index into the arguments we collected in step 1 to get each barrierObject and MemoryObject. There should be at least two partitions that involve `ttg.memdesc_index` for the same barrier or memory alloc we collected in step 1 -- one is the producer partition, the other is the consumer partition. For memory object, fill out `memoryKind` from the value encoding; fill out `allocOp` as the operand in ``ttg.memdesc_index``, which is also the op we collected in step 1; fill out `name` as the name of the allocOp; fill out `offset` as the `ttg.memdesc_index` index; fill out `usages` like this -- the partition that writes to the memory chunk is the producer, and the parition that reads from the memory chunk is the consumer; `MemoryObjectUsage.op` should be the `ttg.memdesc_index` op; task id is the partition id involving this operation. Leave `barrierPair` as uninitialized for this step. For BarrierObject, fill out `kind` from operation name (named barrier uses `ttg.*_named`) and value type; fill out `allocOp` as the operand in `ttg.memdesc_index` (also the op we collected in step 1) if it is shared_memory barrier, and `ttg.*_named` if named barrier; fill out `name` as the name of the allocOp; fill out `offset` as the `ttg.memdesc_index` index if shared_memory barrier; fill out `usages` like this -- the partition that uses this barrier as `bar.wait` is either ProducerAcquire or ConsumerWait, and we put ProducerAcquire temporarily for this step. the partition that uses this barrier as `bar.arrive` is either ProducerCommit or ConsumerRelease, and we put ProducerCommit temporarily for this step. The `BarrierObjectUsage.op` is the `ttg.memdesc_index` op. The `BarrierObjectUsage.taskId` is the the partition id involving this operation.

## Existing wheels
NVWSOps.td
```
┌───────────────────────┬─────────────────────────────────────────────┬──────────────────────────────────────────────┐
  │          Op           │                    Role                     │                  Lowered To                  │
  ├───────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ nvws.create_token     │ Allocates a pair of barrier arrays (full +  │ ttg.local_alloc × 2 + ttng.init_barrier × 2N │
  │                       │ empty)                                      │                                              │
  ├───────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ nvws.producer_acquire │ Producer waits for buffer to be empty       │ ttng.wait_barrier on empty barrier (phase    │
  │                       │                                             │ XOR'd)                                       │
  ├───────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ nvws.producer_commit  │ Producer signals buffer is full             │ ttng.arrive_barrier on full barrier          │
  ├───────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ nvws.consumer_wait    │ Consumer waits for buffer to be full        │ ttng.wait_barrier on full barrier            │
  ├───────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ nvws.consumer_release │ Consumer signals buffer is consumed         │ ttng.arrive_barrier on empty barrier         │
  └───────────────────────┴─────────────────────────────────────────────┴──────────────────────────────────────────────┘
```
