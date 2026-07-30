# Twill 联合调度复现状态报告
## B200 单平台 · 2026-07-26 证据边界修订

## 1. 结论摘要

Twill §5 的产物是软件流水化、带 warp 标注的 IR。论文 §6.1 的性能结果不是自动 codegen
结果，而是专家将该 IR 手工编译成 CUDA C++ 后测得；memory allocation、layout 和
synchronization lowering 的自动化不在论文范围内。TLX 只出现在 related work。

本仓库当前的论文忠实路径在 **manual-CUDA handoff IR** 处结束。本仓库没有论文使用的
手写 CUDA 源码或等价的专家手工编译实现，因此论文 Figure 8/11 和 within-2% 等性能主张
尚未复现。

`sched2tlx`、TLX emitter、TLX skeleton、CuTe shim 以及旧称 SKC/JOS 的 executable bar
都是历史性的非论文实验。它们不能作为 Twill/SKC codegen、端到端 correctness 或性能证据。
旧文件名和 benchmark key 中的 `jos`/`skc` 仅为遗留命名。

| 层次 | 当前状态 | 可作何种证据 |
|---|---|---|
| 归一化与联合求解 | P1 后模型已修正，需重跑 | 形式化模型与搜索证据 |
| §5 handoff IR | 当前忠实终点 | software-pipelined、warp-annotated 调度证据 |
| 专家手写 CUDA C++ | 仓库中不存在 | 无 |
| 论文性能复现 | 未完成 | 无 |
| TLX/sched2tlx/SKC skeleton | 历史本地实验 | 仅限相应实验内核 |

## 2. 当前论文忠实路径

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
STOP: expert manual CUDA C++ implementation is absent
```

当前模型已经加入别名化 SMEM/TMEM、CTA 全局 register 上限、跨组 spill、physical group
width 和 lane mask，并删除旧 cost 聚类、floor、去重和二阶段 tie-break。旧
`*_solution*.json` 由 P1 修正前模型生成；以下求解数字只作为历史记录保留，不能描述当前
模型，也不能直接支撑论文声明。

## 3. 历史求解结果（P1 前模型，待重跑）

| case | 历史 `(II*, L*, copies)` | 历史 `W_min` | 历史消融 | 墙钟 |
|---|---|---|---|---|
| forward sub-tiled | (66, 146, 3) | 4 | W=3 UNSAT | ~10 min |
| forward non-sub-tiled | (62, 173, 3) | 5 | W=4 UNSAT | 785 s |
| backward HD=128 | (95, 273, 3) | 4 | W=3 UNSAT | 366 s |
| backward low-register | (95, 273, 3) | 8 | W=7 UNSAT, 28.5 s | 未记录全搜索 |

历史 sub-tiled 解在 `(66,146,W=4)` 具有 TMA 隔离、双 softmax、独立 rescale 和
ping-pong 特征；一个额外钉约束模板在 `(66,148–150,W=5)` SAT。这些是旧模型的可满足性
观察，不是论文 CUDA 实现或性能证据。同一 `(I,L,W)` 通常有多个合法划分，旧实验选择的
某个划分不能被称为论文专家实现。

下列问题仍要求重跑或重新解释：

- `--no-cross-warp` 与 VARIABLELATENCY 的组合使 TMA→compute 图结构性 UNSAT；旧 11/11
  结果不是独立搜索发现。
- 默认 `max_probes_per_ii=6` 未遍历 Algorithm 1 的完整 L 窗口。
- SWP-only、SWP+heuristic-WS 和 backward low-register 的论文 bar 未实现。
- 部分 UNSAT、去对称性和 backward 扫描没有机器可读存档。

## 4. 论文性能状态

**未复现。** 仓库没有论文 §6.1 的专家手写 CUDA C++ 源码，因而没有可测的 Twill
implementation。FA4 和 cuDNN 可作为独立 baseline，但与历史 TLX bar 的比较不能替代论文
的 Twill bar。自动 emitter 的快慢也不能验证或否定 Twill 调度，因为 emitter 自行承担了
论文未自动化的 allocation、layout、synchronization 和 instruction-selection 决策。

要闭合性能层，必须从当前 handoff IR 出发由专家手工实现 CUDA C++，记录上述正交决策，
再按论文协议验证 correctness 和性能。

## 5. 历史 sched2tlx/TLX 测量（非论文实验）

以下数据保留用于本地 emitter 与 kernel 工程分析。它们不是 Figure 8/11 复刻，也不能标成
JOS、Twill 或 SKC 性能。

**Forward，B=4、H=32、D=128、fp16 non-causal，中位 TFLOPS**

| 历史本地 bar | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|
| Triton tutorial, WS off | 750 | 794 | 813 | 827 |
| local Meta-WS diagnostic | 481 | 522 | 560 | 605 |
| Triton sub-tiled control | 338 | 375 | 406 | 426 |
| TLX baseline emit | 505 | 608 | 688 | 724 |
| legacy `jos` sched2tlx emit | 184 | 200 | 210 | 215 |
| cuDNN baseline | 1188 | 1238 | 1256 | 1294 |
| FA4 baseline | 962 | 991 | 1029 | 1047 |

**Backward，同一 shape 族，中位 TFLOPS**

| 历史本地 bar | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|
| TLX baseline emit | 244 | 258 | 265 | 269 |
| legacy `jos_bwd` sched2tlx emit | SKIP | SKIP | SKIP | SKIP |
| cuDNN baseline | 878 | 979 | 1051 | 1085 |
| FA4 baseline | 896 | 1038 | 1107 | 1141 |

`jos_bwd` 的历史 skip 来自 `L0_smem_3` barrier 被引用但未分配。TLX backward 只计时
dK/dV/dQ kernel，并排除 host 上的 M/D preprocessing；cuDNN/FA4 使用
`fwd+bwd - fwd median`。这些 methodology 不同的数字不能作端到端直接对比。

对 legacy `generated_jos.py` 与 TLX baseline 的 ncu 分析曾将本地 3.25× gap 分解为约
2.54× 指令数和 1.3× 较低 issue rate；还观察到 barrier stall 5.00 vs 1.09 cyc/inst、
880 B spill stores、996 B spill loads，以及一个 skew 变体 deadlock。这些结论只诊断两个
历史 emitted TLX kernel，不诊断 Twill handoff IR 或论文 CUDA 实现。

## 6. 历史 TLX skeleton/SKC 数字（非论文实验）

早期 skeleton binder 测得下列 forward 结果；它只绑定 solution 子集，并对 geometry、cycle
placement 和 protocol 做 relocation/clamp，因此不是 schedule lowering：

| 历史本地 bar | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|
| tutorial baseline | 741 | 786 | 818 | 839 |
| TLX skeleton default | 789 | 827 | 855 | 871 |
| legacy solver-bound skeleton | 699 | 754 | 801 | 844 |
| BN64 variant | 714 | 749 | 767 | 780 |
| QK2 no-skew variant | 699 | 735 | 762 | 774 |
| FA4 baseline | 962 | 991 | 1029 | 1047 |

历史 backward skeleton 为 208、220、230、236 TFLOPS；TLX baseline 为 244、258、265、269。
这些数字只能说明手写/参数化 TLX skeleton 本身。它们既不验证 solver partition，也不能充当
论文 Figure 11 bar。

旧 CuTe Phase B 继承 FA4 forward/backward roles、MMA order 和 barrier protocol，同时丢弃
exact cycles、physical lanes 和 BN=64 geometry。其 within-2%、identity backward、register
quota 和 split-P 等归因均已撤回。artifact round-trip 或 fail-closed 检查最多验证元数据，
不能把自动 TLX/CuTe emitter 变成论文编译器。

## 7. 工件解释

- `subtiled_joint_solution_v6.json`、`fwd_joint_solution_v6.json`、
  `bwd_joint_solution_v6.json` 等是 P1 前历史 solution，需用当前模型重跑。
- `bench/results_{fwd,bwd}.json` 与 `bench/GAP_ANALYSIS.md` 是历史本地 benchmark/profiling。
- `sched2tlx` generated source、TLX skeleton 和旧 SKC manifest 都是非论文实验工件。
- FA4/cuDNN 数值是独立 baseline 测量，不是 Twill 输出。

## 8. 下一步

1. 用当前资源模型和完整 Algorithm 1 搜索窗口重跑 solver 与消融。
2. 固化并审计 software-pipelined、warp-annotated handoff IR。
3. 由专家手工编写缺失的 CUDA C++ implementation。
4. 对该手写实现运行 correctness、deadlock 和论文性能协议。

在完成第 3、4 步前，不应再使用“性能复现成功/失败”“TLX 自动路径复现 Twill”或
“SKC faithful compiler”等表述。
