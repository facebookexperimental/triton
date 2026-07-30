# SKC Phase B 设计记录（历史、已撤回）

## 状态

自动生成 TLX/CuTe kernel 的 SKC Phase B 不属于 Twill 论文，也不是当前的严格复现路径。
当前路径以软件流水化、带 warp 标注的 manual-CUDA handoff IR 为终点；后续应由专家手工
编译为 CUDA C++。本仓库没有论文手写 CUDA 源码，因此没有可用于复现 §6.1 性能的 kernel。

## 论文边界

Twill §5 负责联合调度并输出 warp-annotated IR。§6.1 的结果来自 expert hand-compilation to
CUDA C++。自动内存分配、layout 选择、channel/barrier 合成和 synchronization lowering 均在
论文范围之外；TLX 仅是 related work。

```text
Twill scheduling -> software-pipelined, warp-annotated handoff IR -> STOP
                                                                     |
                                      missing expert-written CUDA C++
```

因此不存在“当前 Phase B 编译器”的论文忠实度主张。当前
`skc.compiler.prepare_manual_cuda_handoff` 仅输出 handoff IR；`sched2tlx`、历史自动
compiler 或 CuTe 实验即使仍有可运行代码，也只能作为非论文工具研究。

## 已撤回的 CuTe/TLX 路径

旧 `skc_cute` 实现继承 FA4 forward/backward kernel，只改 register quota 和少量 stage/phase
参数，同时丢弃 solver 的 exact cycles、physical warp/lane、MMA order 和 BN=64 geometry。
旧 skeleton binder 同样只绑定 solution 子集，并做 relocation/clamp。以下结论全部撤回：

- `skc_cute` 相对 `fa4_1cta` 的 within-2% 判定；
- solver 配额击败专家默认配置的性能归因；
- identity backward shim 是 solver-compiled kernel；
- TLX skeleton 或 `generated_jos.py` 是 Twill/SKC codegen；
- 任何 TLX/CuTe bar 可替代论文 Figure 8/11 的 Twill bar。

旧 `results_{m4,e3,bwd}.jsonl`、TLX bar 和 skeleton 数值可以作为历史内核实验保留，但不能
用于论文结论。

## 历史原型的拒绝检查

原型曾对缺失 provenance、node set 漂移、partial-lane 可表达性、cycle/dependence/resource
违规及 `DROPPED/FROZEN` 字段 fail closed。这些检查能防止实验 emitter 静默篡改 schedule，
但不能把 emitter 变成论文编译器：它仍需自行决定论文明确留给手工实现的 memory、layout 和
synchronization lowering。

同理，artifact -> schedule graph -> TLX source 的逐节点 round-trip 只验证所记录的调度
元数据；它不验证 CUDA 指令级实现，也不产生论文性能证据。

## 当前完成条件

论文性能复现需要一条不同的后续工作流：

1. 用当前模型得到并审计 handoff IR；
2. 专家依据该 IR 手工编写 CUDA C++，显式记录 allocation、layout、同步和指令选择；
3. 验证手写实现保持调度意图并通过 correctness；
4. 按论文 §6.1 协议测量该实现。

在此之前，Phase B 的准确状态是：**历史自动 lowering 实验已撤回；论文性能未复现。**
