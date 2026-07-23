# 忠实度评审（2026-07-23）：本分支 vs 论文 arXiv:2512.18134

> 方法：5 维度评审（归一化 ILP / 模调度+Algorithm 1 / SMT 约束系统 / 实验协议 / 声明审计），
> 逐条对照论文原文与代码；58 条 findings，非 faithful 项逐条对抗验证（41 CONFIRMED / 4 DOWNGRADED）。

## 总裁定

**核心形式化忠实；实验层有 8 处经确认的未披露缺口，其中 2 处触及科学结论的解释力
（无一达到推翻 1:1 主张的 infidelity 级）。**

终局分布：faithful 13 · faithful-with-note 18 · deviation-disclosed 20 · **deviation-undisclosed 8** · infidelity 0。

忠实面（验证确认）：§5.2 归一化 ILP 逐式复刻（含 SCIP、U=300、pairwise 交叉积目标）；
Fig 4/5/6 约束系统逐条对应（整数 t[v,i] ↔ 布尔网格双射论证成立）；Algorithm 1 结构、
MinII 起搜、W 最小化；实验形状/dtype/量测法与论文一致；两项用户授权偏差（仅 B200、
emitter/TLX 代替手工 CUDA）全程一致披露；SKC/skc_cute 明确标注为超出论文的复现层替代。

## 经确认的未披露缺口（按科学分量排序）

**U8（最重）｜消融③"禁跨 warp"是结构性必然 UNSAT，非搜索发现。**
`--no-cross-warp` 对所有边断言 samewarp，而 VARIABLELATENCY 是 iff（TMA 载入独占 W_vl、
其他 op 排除在外）——任何 TMA→compute 图在命题层即 UNSAT，与 I/L/资源无关。
报告把 11/11 真 UNSAT 呈现为与论文一致的实验结果，未披露其真值由构造保证。
（注：若论文的 iff 编码相同，其自身消融同样结构性成立——这本身值得写入欠定处清单。）

**U6｜max_probes_per_ii=6 截断 Algorithm 1 的 L 窗口。**
论文内层循环要求探完同 ceil(L/I) 的整个窗口；代码默认每 II 只探 6 个 L 且 CLI 未覆盖。
若 SAT 点在窗口后段，搜索会返回严格更差的 II——最优性保证有洞。且 plan 文档 §16 声称
探满，与代码矛盾。（实际记录的解都在窗口前 3 个 L 内命中，未实际受害。）

**U3｜论文 fwd 九根 bar 缺三根。**
JOS-SWP（仅软件流水）、SWP+启发式 WS 两根从未实现/量测——论文"联合求解是关键、
拆开无收益"的论断在复现里从未被检验；plan §3 曾映射、§9 曾列为验收项，掉落未记录。

**U4｜bwd 降寄存器预算 bar 未量测。**
bwd_lr_solution_v6.json（W=8）存在但从未落地成内核；论文的 ptxas 溢出叙事
（默认预算溢出→降预算修复且略快）从未在硬件上观察。REPORT §1 以"方向一致"呈现，
未记录 bar 缺失。

**U1｜multiset 语义：审计声称与代码相反。**
plan §16 明言"论文的 C 是逐指令 multiset、按重数计入 U 预算"，normalize.py 也实现了
重数加权；但 ddg.py:275 唯一调用点先 `sorted(set(...))` 去重——重数从未生效，
实际语义是 distinct-value（分辨率系统性偏细）。影响喂给下游所有 II/L/W 的归一化成本。

**U9｜同步 op 的 CONCURRENCY 窗口用 occ 而非 lat。**
论文窗口量是 cycles(o)（与 COMPLETION 同源）；代码 win=max(1, occ[o])。
dump 缺省时二者相等，否则窗口偏窄。

**U5｜run_ablations.sh 陈旧**：warp 数还是 v6 前的（--num-warps 4 在 v6 下是 SAT），
按脚本跑不出报告里的 UNSAT 判定（真实判定来自搜索的 W 下降路径）。

**U10/U11｜证据存档缺口**：消融③ 11/11、去破缺 PASS、bwd 108-UNSAT 扫描无落盘工件
（只有 stdout）；Phase B fwd 基线行（fa4_1cta 与 stock）无原始 JSONL（有 m4 恒等行
间接佐证，2048 处 1092 vs 1123 有出入）。验证员将 U11 降级为 disclosed（可复算），
但 2048 差异应复测。

## 验证员的 4 处降级（原判过重）

spill 同池归一化 → faithful-with-note（论文 s∈[0,S] 未定单位，同池是合理解读且已披露）；
边延迟聚类 KeyError → faithful-with-note（现有三 fixture 均不可达，属潜在 bug 非偏差）；
外层 II 上限 8 + 1h 墙钟 → disclosed（plan §12 预注册过资源类上限）；
Phase B fwd 基线工件 → disclosed（可从原始 ms 复算、m4 行佐证）。

## 建议修复（按性价比）

1. U8：消融③补一段结构性归因披露（或改为仅对非 VL 边禁跨 warp 的弱化版重跑，
   得到非平凡判定）。
2. U6：以 max_probes_per_ii=999 重跑三 case 搜索确认解不变，然后改默认或改文档。
3. U1：改 ddg.py 传重数（或改 plan §16 的声明与代码一致），重跑 v6 确认 II/L/W 稳定。
4. U3/U4：补 SWP-only、heur-WS、bwd-LR 三根 bar（bwd-LR 需先过 emitter/SKC）。
5. U5/U10：刷新 run_ablations.sh 至 v6 口径并归档三组缺失工件；复测 Phase B fwd 基线
   2048 点。
6. U9：CONCURRENCY 窗口改 lat（或披露 occ 选择的理由），重跑敏感性检查。
