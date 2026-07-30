# 忠实度评审（2026-07-26）：本分支 vs Twill 论文

## 证据边界

Twill §5 的输出是软件流水化、带 warp 标注的 IR。论文 §6.1 的 GPU 结果来自专家将该 IR
手工编译为 CUDA C++；自动完成内存分配、layout 选择和同步 lowering 明确不在论文范围内。
TLX 只出现在 related work，不是 Twill 的实现后端。

因此，本仓库当前的论文忠实路径在 **manual-CUDA handoff IR** 处结束。本仓库没有论文用于
§6.1 的手写 CUDA 源码，也没有等价的专家手工编译产物；论文性能结论尚未复现。
`sched2tlx`、TLX emitter、TLX skeleton 和曾称为 SKC 的路径均是历史性的非论文实验，不能
作为 Twill/SKC codegen、端到端编译或性能证据。它们的数值只可用于研究本地实验内核。

## 总裁定

当前可评审的是归一化、联合调度约束、搜索过程和 handoff IR 的结构；不可评审或宣称的是
论文的 CUDA lowering 与性能复现。旧 solution、`W_min` 和消融结论来自 P1 修正前的资源
模型，在当前模型重跑前也不是当前求解器结果。

忠实路径为：

```text
DDG + machine model
        |
        v
normalization + joint scheduling
        |
        v
software-pipelined, warp-annotated handoff IR
        |
        v
STOP: expert manual CUDA C++ implementation is absent from this repository
```

任何 schedule graph、TLX source 或 executable kernel 都位于这条边界之外。即使某个历史
实验通过 correctness 或接近某个 baseline，也不能反向证明 Twill schedule 被忠实编译。

## 当前模型修正状态

- memdesc 作为别名化 SMEM/TMEM 存储对象计费。
- CTA 受 65,536 个 32-bit register 的全局上限约束；跨组 register spill 计入 Figure 5。
- SMT 显式约束逻辑组宽度、physical lane mask 和物理 warp 总数；未知寄存器类型不再按 0 计费。
- 归一化已删除旧 10% cost 聚类、`max/32` floor、调用点去重、min-F 二阶段 tie-break 和
  DDG cost 重定价；完整 cost list 按位置进入 §5.2 的单阶段 ZLP。
- RRT 保留整数、多 functional-unit demand，并强制 `cycles(v)` 与其逻辑行数一致。

论文没有规定 raw compiler `latency/occupancy` 经 ZLP 后如何重采样 RRT。本实现采用最大常量
向量段 duration；这是 compiler-adapter 选择，不是论文公式。论文也没有规定单个 tile op
需要多个 warp 时如何扩展 Figure 6；当前模型显式选择 physical lanes。这些解释必须继续作为
实现选择披露。

## 仍未闭合的忠实度问题

**U8｜“禁跨 warp”消融是结构性 UNSAT。** `--no-cross-warp` 对所有边强制 same-warp，
而 VARIABLELATENCY 将 TMA load 放入独占组并排除其他 op。任何 TMA→compute 图因此在命题层
即 UNSAT，与 II、L 和资源无关。旧报告的 11/11 UNSAT 不能呈现为独立搜索发现。

**U6｜`max_probes_per_ii=6` 截断 Algorithm 1 的 L 窗口。** 论文要求遍历同一
`ceil(L/I)` 窗口；默认只探 6 个 L 会留下最优性缺口。旧解都在前 3 个探针命中，但仍需用
完整窗口重跑确认。

**U3｜论文 forward bar 不完整。** SWP-only 和 SWP+heuristic-WS 没有实现或测量，因而没有
检验论文关于联合求解必要性的完整性能比较。缺 bar 也不能由 TLX 实验补足。

**U4｜backward 低寄存器预算没有论文方式的硬件复现。** 旧 `bwd_lr_solution_v6.json`
只记录历史求解结果；没有专家手工 CUDA 实现，所以论文的 spill/性能叙事未被验证。

**U9｜CONCURRENCY 窗口使用 occupancy 而非 latency。** 论文窗口量来自 `cycles(o)`；
dump 中二者不同时，当前窗口可能偏窄。

**U5/U10/U11｜历史脚本和证据存档不完整。** `run_ablations.sh` 的 warp 参数陈旧；部分
UNSAT、去对称性和 bwd 扫描只留 stdout。历史 TLX/CuTe 数据不能填补这些形式化证据缺口。

## 历史非论文实验

旧 `generated_jos.py`、TLX default、TLX skeleton、CuTe shim 和 identity-shim 数据全部属于
本地工程实验。“JOS”或“SKC”出现在旧文件名、bar 名或 JSON key 中只是遗留命名，不表示
论文系统输出。尤其是：

- 旧 skeleton 只绑定 solution 子集并 relocation/clamp 几何、cycle placement 或协议；
- 旧 CuTe Phase B 继承 FA4 roles、MMA 顺序和 barrier protocol；
- `sched2tlx` 自行承担论文未自动化的 allocation/layout/synchronization 决策。

这些实验可以保留用于 emitter/debugging 比较，但不得进入论文复现判定、Figure 8/11 映射或
within-2% 等性能结论。

## 后续闭环条件

1. 用当前模型和完整 L 窗口重跑 solver、消融与去对称性检查，并归档机器可读证据。
2. 将输出限制为可审计的 software-pipelined、warp-annotated handoff IR。
3. 由专家依据 handoff IR 手工实现 CUDA C++，明确记录 allocation、layout、barrier 和指令选择。
4. 对该手写 CUDA 实现做 correctness 与论文协议下的性能测量。

在第 3、4 步完成前，最准确的结论是：**调度层复现仍在校验，性能层未复现。**
