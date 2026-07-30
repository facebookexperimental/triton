# SKC 设计记录（历史、非论文路径）

## 状态

SKC 是本仓库曾探索的自动 lowering 方案，不是 Twill 论文中的编译阶段，也不是当前论文
复现路径。旧文档把“完整 schedule → standalone TLX kernel”描述为 Twill 的严格编译器；
该表述已撤回。

Twill §5 只生成软件流水化、带 warp 标注的 IR。论文 §6.1 由专家把该 IR 手工编译为
CUDA C++；内存分配、layout 和同步的自动 lowering 不在论文范围内。TLX 只出现在 related
work。当前忠实路径因此在 manual-CUDA handoff IR 处停止：

```text
joint solution + DDG + machine model
                |
                v
software-pipelined, warp-annotated handoff IR
                |
                v
STOP: manual CUDA C++ implementation is not present in this repository
```

当前入口为：

```bash
python -m skc \
  --solution solution.json \
  --ddg ddg.json \
  --baseline-graph schedule_graph.json \
  --ir-out pipelined_ir.json \
  --manifest-out manual_cuda_handoff.json
```

`twill-pipelined-warp-ir-v1` 保留 TTGIR-derived op table、II/L/copies、prologue / steady
state / epilogue 边界、逐 instruction cycle/stage/offset/group/lane，以及全部 dependence 和
跨 warp dependence。`twill-manual-cuda-handoff-v1` 明确记录 `executable_generated=false`，
并把 allocation、layout、synchronization 和 instruction selection 留给专家手工 CUDA。

## 当前证据规则

- `paper_joint_solver` 的 artifact 可以用于审计 cycle、warp、dependence 和资源约束。
- artifact 或 schedule graph 的完整 round-trip 只能证明元数据保真，不能证明 CUDA lowering。
- 当前 `skc.compiler.prepare_manual_cuda_handoff` 只是 handoff IR 的兼容入口，不生成
  executable code；已删除的自动 compiler、`sched2tlx`、TLX skeleton 和 CuTe shim 属于历史实验。
- “JOS”或“SKC”遗留文件名及 benchmark key 不表示 Twill/SKC codegen 已发生。
- 没有论文的手写 CUDA 源码，所以当前没有 paper-generated kernel，也没有可归因于 Twill 的
  correctness 或性能结果。

## 历史方案概述

历史 SKC 原型曾要求 `twill-joint-solution-v1` artifact 携带：

1. `ii`、`length`、`copies` 以及每个 DDG node 的 `cycles`；
2. 每个 owned node 的 `warp`、`group_widths` 和 `lane_masks`；
3. DDG edges、loop distance、geometry 和 machine limits；
4. DDG、baseline graph、normalization、machine model 和 solver source 指纹。

原型把 `cycle // II` 映射为 stage，在 stage 内生成 cluster，重算 ring depth，并尝试为跨组值
生成 channel/barrier。它还检查 node set、指纹和若干 round-trip 字段。这些检查对实验工具仍
有工程价值，但原型同时自动选择或合成 allocation、layout、buffer lifetime 和 synchronization，
恰好跨越了论文留给专家手工 CUDA 实现的边界。

## 为什么历史结果不是 codegen 证据

旧 skeleton binder 只消费 solution 的子集，并会替换 geometry、relocate cycle placement 或
clamp stage/protocol。旧 CuTe shim 继承 FA4 的 warp roles、MMA order 和 barrier protocol。
standalone TLX emitter 也有自己的 lane 可表达性、register scaling、buffer 和 barrier 规则。
因此它们运行出的内核既不是论文 §6.1 的手写 CUDA，也不能作为 schedule 的机械忠实 lowering。

性能接近专家 kernel 不能证明编译忠实；性能落后也不能反证 Twill 调度，因为差异可能来自
论文未自动化的 layout、allocation、synchronization 和 instruction-selection 决策。

## 保留用途

历史 SKC/TLX 工件可用于 emitter、barrier、资源模型和 schedule-graph round-trip 的独立研究。
引用这些工件时必须标为“historical non-paper experiment”，并与 Twill solver/IR 证据及论文
性能复现分开报告。
