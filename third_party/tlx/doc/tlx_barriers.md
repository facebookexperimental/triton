#  Barrier Support in TLX

## Introduction

### Barriers

Barriers are primitives that allow synchronization between the warps of
a kernel. There are full synchronous barriers like \_\_syncthreads(),
which requires a warp to wait at the barrier until all other warps have
also reached the barrier. This blocking behavior makes them less
efficient for implementing patterns like Warp Specialized Producer
Consumer, where *Producer* warps can fill *buffer0*, notify *Consumers*
waiting for *buffer0*, then go on to fill *buffer1* without waiting for
Consumers to finish consuming buffer0.

### Asynchronous Barriers

Asynchronous Barriers allow semaphore-like *Arrive()* and *Wait()* based
coordination between warps. Asynchronous Barriers also provide the
ability to perform synchronization only between a subset of the warps
(*participating warps*). A warp that does an Arrive() on a barrier does
not have to wait for other participating warps. The non-participating
warps can execute independent of the participating warps. The
participating warps use hardware barrier instructions, over unique
pre-allocated hardware barrier objects or shared-memory(*shmem*)
allocated barrier objects, to achieve fine-grained synchronization. On
certain NVIDIA platforms, asynchronous barriers can also be used to
track the completion of asynchronous transactions, like TMA loads.

**Note:** In the remainder of this doc we will refer to Asynchronous
Barriers as just Barriers.

**Note:** AMD h/w does not support Asynchronous Barriers but most of the
TLX barrier operations are implemented in s/w using shared-memory
variables

<p align="center">
  <img src="/third_party/tlx/media/image2.PNG"
  style="width:6.5in;height:4.45833in" />

  Figure 1. Producer Consumer example with Synchronous vs Asynchronous
  barriers. Producer wave0/wave1 load the first/second half of bufferA and
  bufferB. Consumer waves use the full buffers. For each wave, the first
  instruction is assumed to start at the same time, across both scenarios.
</p>

#####

### Barrier Operations

Barrier operations can be classified into three categories a)
*Alloc/Init* b) *Arrive* c) *Wait*

*Alloc/Init*

- Allocate barrier objects in shmem.

- Initialize barriers with the count of threads that are expected to
  perform an Arrive operation on the barrier.

- **Note:** This allocation and initialization steps are not required
  for hardware pre-allocated barriers.

- Barriers can also be used to track completion of asynchronous memory
  transactions (like TMA) and can be initialized with an *expected
  transaction count*, like bytes transferred in a TMA.

*Arrive*

- A warp performs an Arrive on a barrier to indicate completion of some
  work.

- Arrive is non-blocking and the warp can proceed as soon as it performs
  an Arrive.

- Once the expected number of threads perform an Arrive on the barrier,
  warps waiting on the barrier become unblocked.

- In cases where a barrier is used to track one or more transactions,
  the Arrive happens implicitly when the transaction is completed. Some
  examples include:

  - A TMA op will arrive a barrier when it has transferred an expected
    amount of bytes

  - A Blackwell tcgen05 commit op will have a barrier to track all prior
    async tcgen05 ops initiated by the calling thread. When those ops
    are done, the barrier will be arrived.

*Wait*

- A warp performing a Wait is blocked at a barrier until the specified
  number of Arrive’ing warps reach the barrier or until the expected
  transaction count is reached.

- A warp that performs a Wait on a barrier executes independent of other
  warps waiting at the barrier and such warps can enter and exit the
  Wait at different times

## TLX Barriers

TLX provides two categories of barriers a) Named Barriers and b) Memory
barriers

### Named Barriers

- **Note:** Named barriers are only supported on NVIDIA

- Named barriers are h/w pre-allocated barrier objects that are
  referenced by a number (name of the barrier). The supported range for
  this number is 0-15 per CTA.

- Named barriers do not have to be allocated or initialized.

- Wait and Arrive are called with the count of expected threads to
  arrive at the barrier.

- All threads in the warp participate in the arrive operation, so the
  thread count should be *number of warps \* threads per warp*

- Suitable for achieving execution patterns like PingPong where a mutual
  exclusive execution order is desired

#### APIs

- ***tlx.named_barrier_wait(bar_id, num_threads)***
  Wait until num_threads threads have reached the phase of the *bar_id*
  named barrier. num_threads has to be a multiple of warp size i.e.
  multiples of 32.

- ***tlx.named_barrier_arrive(bar_id, num_threads)***
  Signal arrival at *bar_id* named barrier with an arrival count of
  *num_threads*. num_threads has to be a multiple of warp size i.e.
  multiples of 32.


| TLX | MLIR | PTX |
|----|----|----|
| tlx.named_barrier_wait | ttng::wait_barrier_named | [<u>bar.sync</u>](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar) |
| tlx.named_barrier_arrive | ttng::arrive_barrier_named | [<u>bar.arrive</u>](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar) |

#### Example (PingPong Schedule)

PingPong scheduling creates mutually exclusive ‘Ping’ and ‘Pong’
execution patterns between warps in order to reduce contention on shared
hardware resources. To achieve this pattern, code is clustered into Ping
and Pong clusters, barriers are placed around these clusters, and then a
subset of the waves are held back from execution behind a conditional
barrier. The following code snippet, taken from the
[<u>ws-pipelined-pingpong-flash-attention-fwd
kernel</u>](https://www.internalfb.com/code/fbsource/third-party/triton/beta/triton/third_party/tlx/tutorials/test_flash-attention-WS-pipelined-pingpong-hopper.py?lines=38)
illustrates this idea.

```python
if cid == 0:
  #Consumer 0 waits for Consumer 1 to reach synchronization point at barrier 9.
  tlx.named_barrier_wait(9, 256)
else:
  #Consumer 1 signals its arrival at barrier 9.
  tlx.named_barrier_arrive(9, 256)
  #Then waits at barrier 10 until Consumer 0 finishes issuing its async_dot.
  tlx.named_barrier_wait(10, 256)
  qk = tlx.async_dot(q_tile, k_tile)
if cid == 0:
  #After issuing async_dot, Consumer0 signals barrier 10 to unblock Consumer 1.

  tlx.named_barrier_arrive(10, 256)
  # wait for the MMA using to complete
  qk = tlx.async_dot_wait(0, qk)
```


The PingPong schedule is achieved using *named barriers* 9 and 10.

This pattern prevents *cid=0* and *cid=1* from executing the
*tlx.async_dot* at the same time and contending on the Tensor Core
units. In this kernel, there are 2 consumer warp-groups, with 4 warps
each, with 32 threads per warp, so the arrive/wait count is set to
2\*4\*32 = 256.

### Memory Barriers

- The kernel has to allocate the *shmem* barrier object and initialize
  it with an integer *expected* *count* value.

- The barrier object implicitly tracks the *phase* of the barrier*.*
  Phase is a 0-initialized boolean value that is toggled every time the
  following conditions are met:

  - The expected count of threads have arrived at the barrier and

  - The expected transaction count is reached

- A phase flip is an indication to the Wait’ing warps that the
  Arrive’ing warps have completed the work that the Wait’ing warps are
  blocked on.

- CUDA [<u>documentation on
  phase</u>](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)

- Wait’ing warps have to maintain the barrier’s phase in a local
  variable and pass it to the Wait call. The Wait will block until the
  passed-in phase is not equal to the barrier’s phase i.e. until a
  barrier phase flip has occurred.

- Pseudocode for *arrive*

```python
arrive(Barrier barrier, int arrive_count = 1):
  barrier.count -= arrive_count # atomic decrement
  if barrier.count == 0:
    barrier.phase ^= 1
    barrier.count = barrier.expected_count
```

- Pseudocode for *wait*

```python
wait(Barrier barrier, bool local_phase):
  while local_phase == barrier.phase:
    pass

```

- Pseudocode for *Producer Consumer* with Memory Barriers

```python
# Producer Consumer
# Barrier init will set barrier phase to 0
barrierFull = Barrier(expected_count = num_producer_threads)
barrierEmpty = Barrier(expected_count = num_consumer_threads)
# The following local phase initialization will ensure that the first
# bufferEmpty.wait() in the producer will be a noop. This will
# ensure that the producer is ahead of the consumer by one phase
buffer_empty_phase = 1
buffer_full_phase = 0
while !done:
  if is_producer_thread():
    # first producer wait will be a noop
    bufferEmpty.wait(buffer_empty_phase)
    buffer_empty_phase ^= 1
    do_load(mem_buffer)
    bufferFull.arrive()
```

#### Barrier APIs

- ***tlx.alloc_barrier(num_barriers, arrive_count=1)**  *
  Allocates a buffer in shared memory for *num_barrier* barrier objects
  and initializes them with *arrive_count*. *arrive_count* should be
  initialized based on the context in which this barrier’s barrier is
  executed.

| Context of arrive | arrive_count | Notes |
|:---|:---|:---|
| Implicit arrive of an *tlx.barrier_expect_bytes* | 1 | Only one thread modifies the barrier arrival count after completion of a transaction |
| *tlx.barrier_arrive* on NV within a *tlx.async_task* region | Number of warp groups | Only one thread per MMA group modifies the barrier arrival count on arrive |
| *tlx.barrier_arrive* on NV outside a *tlx.async_task* region | 1 | Only tid == 0 modifies the barrier arrival count on arrive |
| *tlx.barrier_arrive* on AMD | num_warps that execute *tlx.barrier_arrive* | One thread per wave(warp) increments the barrier count |

- ***tlx.barrier_expect_bytes(bar, bytes)***
  Specifies that *bytes* amount of data is expected to be copied before
  a barrier\_*wait* on *bar* can be unblocked. An implicit arrive will
  happen on *bar* when the corresponding transaction completes reading
  *bytes* amount of data.

- ***tlx.barrier_wait(bar, phase)***
  Wait until the *bar*’s phase has moved ahead of the *phase* argument .

- ***tlx.barrier_arrive(bar, arrive_count=1)***
  Performs an arrive operation on *bar*, by decrementing *arrive_count*
  from the *bar*’s arrival count*.* The phase of *bar* is flipped if
  bar’s arrival count becomes 0. **Note:** It is recommended to use the
  barrier_arrive() with arrive_count=1. The *arrive_count* of
  *tlx.alloc_barrier* can be set to achieve the desired phase change
  behavior.

  | TLX [<u>barriers</u>](https://github.com/facebookexperimental/triton/blob/tlx/third_party/tlx/language/tlx/barrier.py) | MLIR | PTX |
  |----|----|----|
  | tlx.alloc_barriers | ttng::InitBarrierOp | [<u>mbarrier.init</u>](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init) |
  | tlx.barrier_expect_bytes | ttng::BarrierExpectOp | [<u>mbarrier.expect_tx</u>](http://mbarrier.expect_tx) |
  | tlx.barrier_wait | ttng::WaitBarrierOp | [<u>mbarrier.try_wait</u>](http://mbarrier.try_wait) |
  | tlx.barrier_arrive | ttng::ArriveBarrierOp | [<u>mbarrier.arrive</u>](http://mbarrier.arrive) |

### Examples

#### WS-GEMM [<u>https://github.com/facebookexperimental/triton/blob/tlx/third_party/tlx/tutorials/gemm-WS-hopper.py</u>](https://github.com/facebookexperimental/triton/blob/tlx/third_party/tlx/tutorials/gemm-WS-hopper.py)

<p align="center">
  <img src="/third_party/tlx/media/image3.PNG"
  style="width:2.92696in;height:2.21978in" /><img src="/third_party/tlx/media/image4.PNG"
  style="width:3.20541in;height:2.44647in" />
</p>

In the above diagram of GEMM, we have warp specialization with 1
producer warp group (aka async task in TLX) for TMA load and 2 consumer
warp groups for MMA. The target tile BMxBN (in green) is computed by
BMxK (in blue) and KxBN (in yellow) and requires 8 MMA of smaller tiles
with sizes of (BM/2) x BK and BK x BN. (Dividing BM by 2 because we have
2 consumer groups)

TLX mbarriers are used by WS-GEMM for asynchronized communication
between warp groups. In the simplest case, when TMA WG issues a TMA load
bonding barFull, the MMA WG waits for barFull before doing MMA. Once the
TMA load finishes, barFull will arrive and MMA op begins. Once MMA is
completed, a barEmpty will be marked 'arrived' so that the other waiting
WG can proceed.

Mbarrier contains 'phase' in its opaque object. We flip phase values
between 0 and 1 each time the current phase completes. In our GEMM
example, the current phase completes when either TMA load finishes or
tlx.barrier_arrive is called. To overlap the TMA and MMA operations of
multiple iterations for latency hiding, we flip phase every 2 (number of
consumers) iterations as below. TMA load in iter 0 (barFull\[0\]) blocks
the MMA in iter 0. MMA in iter 0 blocks the TMA load in iter 2 but
doesn't block the TMA load in iter 1.

<p align="center">
  <img src="/third_party/tlx/media/image5.PNG"
  style="width:6.38315in;height:3.11992in" />
</p>

Now assemble everything together to
[<u>illustrate</u>](https://www.internalfb.com/excalidraw/EX486624) how
a BMxBN target tile is calculated. Recall we have (1) 8 (BM/2)xBK
sub-tiles from A and 4 BKxBN sub-tiles from B (2) 1 TMA WG (producer)
and 2 MMA WGs (consumer) (3) 4 EmptyA bars, 4 FullA bars, 2 EmptyB bars
and 2 FullB bars because each WGMMA operation needs two 'full' bars for
both operands and each TMA load need one 'empty' bar to proceed.

<p align="center">
<img src="/third_party/tlx/media/image1.PNG" style="width:6.5in;height:2.5in" />
</p>
