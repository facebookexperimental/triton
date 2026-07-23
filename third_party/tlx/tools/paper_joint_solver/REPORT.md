# 复现报告：《Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs》(arXiv:2512.18134)
## B200 单平台 · TLX (sched2tlx emitter) 落地 · 2026-07-21/22

> 论文系统代号 **JOS**（Joint Optimal Scheduler）。范围与落地方式为用户指定的两项偏差
> （仅 Blackwell/B200；codegen 用 emitter→TLX 而非论文的手工 CUDA C++），其余按论文 1:1。
> 全部审计与过程记录见 `/projects/kzhou6/hwu27/repro-plan-b200-tlx.md` §14–§20。

---

## 1. 结论摘要

**论文的"发现层"声明全部复现成功；"性能层"声明在 TLX 自动落地下未达成，且该落差本身
实证了论文选择手工编译的核心理由。**

| 论文声明 | 复现判定 | 证据 |
|---|---|---|
| 联合求解器能从第一性原理重发现专家 WS 策略 | ✅ | 前向最优点 4/5 FA4 特征；完整 FA4 模板在最优点 +2 单位处 SAT（fa4_like=True） |
| 减少 warp 数 → UNSAT | ✅（真 unsat 判定） | subtiled/bwd：W=3 UNSAT；fwd：W=4 UNSAT |
| 不 sub-tile → 在 ZLP 可行最小 I 处 SMT UNSAT | ✅（逐字对应） | fwd II=60 处 L 全窗 16/16 真 UNSAT（并量化：II 需退至 62，−3.3% 吞吐） |
| 禁跨 warp 通信 → UNSAT | ✅ | subtiled 全窗 11/11 真 UNSAT |
| 求解时间数十秒~数分钟（Xeon 8570 单核） | ✅ 同量级同款 CPU | bwd 366 s（论文 269 s）；subtiled 全搜索 ~10 min；单判定 3–15 s |
| 降低寄存器预算 → 更多 warp 组 | ✅ 方向一致 | bwd：预算减半 → W_min 4→8（W=7 真 UNSAT） |
| 生成调度实现物性能贴近手工实现（fwd within 2% of FA4） | ❌（TLX 落地下） | JOS 215 vs FA4 1047 TFLOPS @16K（0.21×），亦低于 TLX-Default 724（0.30×） |
| "Triton 自动路径无法正确编译或性能差"（论文手工编译的理由） | ✅ 被实证 | 见 §5 归因：自动落地的正交决策吃掉全部调度收益；且 stock AutoWS 管线在本 build 编译失败 |

---

## 2. 求解层结果（终稿 v6 口径）

| case | (II*, L*, copies) | W_min | 减 warp 消融 | 去破缺复核 | U | 求解墙钟 |
|---|---|---|---|---|---|---|
| 前向 sub-tiled（图 8 主对象） | (66, 146, 3) | 4 | W=3 真 UNSAT (4.5 s) | PASS | 300 | 全搜索 ~10 min |
| 前向非 sub-tiled | (62, 173, 3) | 5 | W=4 真 UNSAT | — | 150† | 785 s |
| 反向 (HD=128) | (95, 273, 3) | 4 | W=3 真 UNSAT | PASS | 300 | 366 s |
| 反向 -LR（寄存器预算减半） | (95, 273, 3) | **8**（W=7 UNSAT 28.5 s） | — | — | 300 | — |

† U=300 时公式触 150G 内存闸；U 为论文自有的用户参数，逐 case 记录。

**策略结构对照（前向 sub-tiled vs 论文图 9 的 FA4 策略）**
- Algorithm-1 最优点 (66,146,W=4)：TMA 变延迟隔离组 ✓ / 双 softmax 组 ✓ / 独立 rescale ✓ / ping-pong 相位交错 ✓ / MMA 专用组 ✗（分散于 softmax 组）。
- 完整 FA4 模板（加共置+分离探针钉住）：**(66, 148–150, W=5) SAT，5 s，fa4_like=True**——论文报告的
  精确结构在最优点 +2~4 归一化单位处可满足并被实证；其在最优 L 处 UNSAT 的原因是跨组 spill 链 +4 单位。
- 反向：VL 基建组 + 3 计算组，对应论文降预算叙事的三组结构；ping-pong 由发射序涌现。
- 纯可满足性语义注记：同一 (I,L,W) 点通常有多个合法划分（论文与我们各取一抽取）；见 §5 对性能后果的量化。

**图 9 复刻**：`fig9_subtiled_v6.dot` / `fig9_subtiled_fa4exact.dot`（warp 组着色依赖图）。

---

## 3. 性能测量（图 8 / 图 11 复刻；B=4, H=32, D=128, fp16 非因果；中位 TFLOPS，q20/q80 差 <2%）

**前向（图 8 对应）**

| bar | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| Triton（tutorial, WS off） | 750 | 794 | 813 | 827 |
| Triton-WS† | 481 | 522 | 560 | 605 |
| Triton-Tiled | 338 | 375 | 406 | 426 |
| TLX-Default（emitter 基线调度） | 505 | 608 | 688 | 724 |
| **JOS**（求解器 FA4 结构 → emitter） | 184 | 200 | 210 | **215** |
| cuDNN | 1188 | 1238 | 1256 | **1294** |
| FA4 官方 | 962 | 991 | 1029 | **1047** |

† stock AutoWS 管线在本 build 对 tutorial 全配置编译失败（`tmem_alloc destroyed but still has uses`），
以 `TRITON_USE_META_WS=1` + 修剪配置绕行（已记偏差）。

**反向（图 11 对应）**

| bar | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| TLX-Default | 244 | 258 | 265 | 269 |
| JOS | SKIP（emitter 新 gap：`L0_smem_3` barrier 引用未分配；签名已记录） | | | |
| cuDNN | 878 | 979 | 1051 | 1085 |
| FA4 官方 | 896 | 1038 | 1107 | 1141 |

环境注记：2026 软件栈上 cuDNN 前向反超 FA4 24%（论文 2025 栈两者接近）；时钟稳定 boost
1950–1965 MHz（不可锁频的代偿记录在 JSON env 探针中）。

---

## 4. 与论文数字的对照要点

- 论文图 8：其系统 ≈ FA4 −2%，显著高于 Triton。我们：JOS(TLX) = 0.21× FA4、0.26× Triton-best。
- 差异不在求解器（结构已重发现），在**落地路径**：论文由专家手工编译 CUDA 并补齐全部"正交优化"
  （TMA multicast、布局、指令选择等——论文自己声明这些 out of scope）；我们按用户指定走 emitter 自动路径。
- 关键旁证：**TLX-Default（同一 emitter 的基线调度）自身即落后 FA4 31%**——自动路径的天花板；
  JOS 在其上进一步付出通道/寄存器/串行回退的代价（§5）。

## 5. JOS 性能差距归因（ncu + ptxas 实证；GAP_ANALYSIS.md）

3.4× 差距完整分解：**2.54× 指令数膨胀 × 1.3× 发射率下降 ≈ 3.3×（实测 3.25×）**。按杠杆排序：

1. **划分抽取将 TMEM 卸载+双 rescale 压在 MMA 发射组**（每迭代 16 次 barrier_wait；全核 35 vs 基线 12）：
   barrier 停顿 5.00 vs 1.09 cyc/inst，SM 吞吐 24.5% vs 41.2%。模型层根因：可满足性抽取对
   "发射槽二阶压力"无代价项——**论文的最优性 = min-II/min-W，不含指令数/停顿目标**，合法解间性能差可达 3×+。
2. **寄存器超订**：21 warps/CTA → 152→104 缩放 → ptxas 溢出 880B/996B（基线 0/0）；
   long_scoreboard 15.7 vs 8.0 cyc/inst。
3. **组内串行回退**（skew 环因 TMEM 预算被弃）；变体实验显示 skew 发射路径自身有同步 bug（独立缺陷，触发即死锁）。
4. 合成通道的同组 SMEM 往返 + 死代码 wg4（写图合成的可优化项）。
5. **负结果**：环深不是瓶颈（加深无增益，DRAM 空闲 49 GB/s）。

## 6. 忠实度审计摘要（全文见方案文档 §16–§19）

- **A 类（可证等价）**：对称性破缺（去破缺复核全 PASS）、整数编码↔布尔网格双射、MinII 起搜、贪心仅作 warm-start。
- **B 类（论文授权的机器模型决策）**：TMEM 独立功能单元、成本实测、10% 聚类 + 1/32 下限、逐 case U。
- **C 类（欠定处解释，均已披露）**：min-F 字典序 tie-break、"报告策略 = 最小 W"、spill 代价同池归一化、
  CONCURRENCY 异步发射窗口（论文图 2 语义）、寄存器携带循环变量同组（emitter 实现性）。
- **D 类（资源上限）**：ILP/SMT 限时、150G 内存闸——所有 UNSAT 结论仅采信求解器真 unsat，从未以超时充数。

## 7. 工件索引

- 求解器：`paper_joint_solver/`（normalize/ddg/modulo_ilp/greedy_ims/joint_smt/search/graph_writer/viz/strategy_report + tests 23 项）
- 解：`subtiled_joint_solution_v6.json`、`subtiled_fa4exact_solution.json`、`fwd_joint_solution_v6.json`、`bwd_joint_solution_v6.json`、`bwd_lr_solution_v6.json`
- 内核：`examples/case3_FA_fp16{,_subtiled}/generated{,_jos}.py`、`case4_FA_bwd/generated_hd128.py`（全部 B200 正确性 PASS）
- 测量：`bench/RESULTS.md`、`results_{fwd,bwd}.json`、`GAP_ANALYSIS.md`；可视化 `fig9_*.dot`
- emitter 修复（全部字节级回归）：tensordesc 正则、iter_arg 去重、TMEM bridge 槽位、acc-TMEM 别名、
  混合环深槽位、跨组寄存器通道合成、ghost 缓冲深度钳制

## 7.5 论文欠定处补记（复现过程暴露）

除 §6 C 类已列各项外：论文的调度模型严格单 SM（§3.1 "entire SM"，全文零提及 cluster/CTA-pair），
但其手工实现层使用了 cluster 特性（§6.2.1 TMA multicasting，标注"正交优化"），且未披露其 Blackwell
手工实现是否使用 2-CTA tcgen05——而 FA4 基线内部使用 cluster 特性，within-2% 声明与此直接相关。
本复现的裁定：模型/求解器/JOS 内核严格 1-CTA（与论文模型同范围）；基线按原样黑盒测量（与论文同法）；
实现层仅允许"保调度"正交优化（multicast/persistent 类），2-CTA 类"改调度"优化默认排除（SKC 设计 §2 B2）。

## 8. 遗留项

1. bwd JOS 内核的 `L0_smem_3` barrier 分配 gap（签名已记录，修复后可补 jos_bwd bar）。
2. skew 发射路径的同步 bug（触发即死锁；修复后可解除串行回退杠杆 #3）。
3. M_lse epilogue store 缺失（正确性门禁不检查该输出）——SKC 路径已补齐（§9）。
4. 性能寻优方向（超出论文范围）：在联合系统上加二阶目标（min 跨组通道数 / min 指令数代理），
   或对合法解族做小规模实测择优——论文框架天然支持加约束迭代。

## 9. SKC 落地结果（2026-07-22 补充；设计见 SKC_DESIGN.md）

SKC Phase A 已实现（`skc/` 包：RoleClassifier → ScheduleBinder → SkeletonInstantiator，
骨架 = blackwell_fa_ws 协议参数化），同一 FA4-exact 解重新落地。同 harness 前向数据（TFLOPS）：

| bar | 2048 | 4096 | 8192 | 16384 | 说明 |
|---|---|---|---|---|---|
| 原版 tutorial（同 config 同 harness） | 741 | 786 | 818 | 839 | M1 对照基线 |
| SKC-default（M1，骨架默认参数） | 789 | 827 | 855 | **871** | ≥651 门大幅通过，≥ 原版 |
| **SKC-JOS（M2，解绑定实例）** | 699 | 754 | 801 | **844** | vs emitter 路径 215 = **3.9×** |
| 变体 BN64（M3：仅解几何） | 714 | 749 | 767 | 780 | 几何代价 −10% |
| 变体 QK2 无 skew（M3） | 699 | 735 | 762 | 774 | 深度单独 ≈ 中性 |
| FA4 官方 | 962 | 991 | 1029 | 1047 | |

**M3 归因（@16K）**：解绑定的三个参数增量中，(a) 解的 tile 几何 BLOCK_N=64（fixture 产物）
−91；(b) QK 环深 2 单独 ≈ 0；(c) **PV skew（跨迭代 QK 预发射，解的核心调度洞见）+70**——
在解自身几何上，解的调度（844）显著优于朴素发射序（774/780），即求解器发现的重叠结构在真硬件上
成立。几何损失源于成本模型未定价 BLOCK_N=64 的 mbarrier/发射开销，非调度逻辑缺陷。
硬件还实证了论文的 sub-tiling 机制：BN=128 下 QK 双缓冲 TMEM 放不下（768>512 列），
只有 sub-tile 几何才腾得出双缓冲+skew 的空间——与论文"不 sub-tile → UNSAT"的论证同构。

**结论更新**：图 8 的 JOS bar 由 215（通用 emitter）修订为 **844（SKC）** = FA4 的 0.81×、
TLX-Default 的 1.17×、原 emitter 的 3.9×。§1 表中"性能贴近手工实现"一项由 ❌ 升为
**部分达成**（0.81× FA4，非 within-2%）；Phase A（TLX 后端）天花板即此，within-2% 需
Phase B（CuTe-DSL 后端）。§5 归因的杠杆 #1（TMEM/rescale 压 MMA 组）、#3（skew 死锁）、
#4（SMEM 通道往返）在 SKC 中被 R1/R3/R4 结构性消除；skew 死锁的根因即 §8.2，SKC 的
显式 acc_empties 等待给出了正确实现。

### 9.1 M4：bwd 骨架与图 11 JOS bar（补全 §3 反向表）

骨架 = case4 手写 fa_bwd_dkdv_ws 参数化（4 role：load / 专职 MMA 5-dot skewed / compute / dQ 归约）。
**图 11 更新（TFLOPS）**：

| bar | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| **JOS-bwd（SKC，原 emitter 版 SKIP）** | 208 | 220 | 230 | **236** |
| TLX-bwd-Default（emitter） | 244 | 258 | 265 | 269 |
| cuDNN / FA4 | 878/896 | | | 1085/1141 |

**家族求解的三项发现**（bwd_skc_solution.json，判别探针序列）：
1. **R1 专职 MMA 发射组在 bwd 零 II 代价**：钉定 5-dot 专职组的解 SAT 于 (II=95, L=273)——与
   v6 自由最优完全同点。
2. **模型拒绝骨架的单体 compute 任务**：{exp2, dpT-load, dS-sub} 两两可共置、三者同组在
   II≤105 真 UNSAT（整链钉定的全模板 II≤111×L≤317 共 108 真 UNSAT）——模型要求 softmax+dS
   链摊 ≥2 个 warp 组，与论文 bwd 的 3 计算组结构一致，**且预言了硬件实测**：单 compute 任务的
   手写结构 236 < emitter 分摊调度 269。求解器对 bwd 分区的判断被硬件验证。
3. **两处模型↔硬件账目差**（绑定期钳制并审计）：DDG 解算几何 BLOCK_M=64 → 本 build 64 行
   TMEM tile 不支持，保调度放大 128；Q 环活性 3 copies → 模型 SMEM 预算缺 ~17KB
   barrier/padding 项（Q=3 实测 OOM），钳到 2。净绑定 = 骨架默认配置获 solver 认证，
   skc_bwd_jos ≡ skc_bwd_default（236）。

bwd 诚实判读：Phase A 的 bwd 天花板由 TLX 手写内核水平决定（无论单体 236 还是分摊 269，
均 ≈0.21–0.24× FA4）——比 fwd 离论文 within-2% 更远，Phase B 是唯一路径。原 §3 反向表的
jos_bwd SKIP（emitter `L0_smem_3` gap）由本节数据取代；§8.1 遗留项相应关闭。

## 10. Phase B 结果：CuTe-DSL 后端（skc_cute，2026-07-22）

设计见 SKC_PHASE_B_DESIGN.md；实现 = 零 fork 的 subclass shim 绑定 FA4 自己的内核类
（全树指纹 pin @2409214a，恒等门 ±0.03%，审计哈希入每条计时 JSON）。CUDA 13.0（与论文一致）。

### 10.1 三层证据判定

**E1 收敛（全部成立）**：求解器独立推导 == FA4 专家值——KV 环深 3、S/P 双缓冲 2、双 softmax
链、专职 MMA 发射者、bwd Q_stage 2 + compute 摊 2 组；且解算活性配额推导独立落在专家邻域
（softmax ≈198/thread vs 专家 176–192）。

**E2 配额绑定（正向通过，可证伪实验）**：上游 _TUNING_CONFIG 缺 1-CTA nc hd128 键（回退
192/80/48 未调优）。解算活性推导的 200/88/24 在**全部形状为最优候选**：8K +0.99%、16K +0.40%
（复跑 +0.38% 可复现）、2K 中位 +5.3%（分布双峰，记不稳定）。**求解器在专家自己的骨架上、
专家未调的配置点上击败上游默认。** bwd 负结果如实报：2-CTA 调优配额迁移到 1-CTA 反而 −1.5%
（1-CTA 默认 reduce=152 已是对的）。

**E3 扰动排序（fwd @16K，模型预测方向全部命中）**：
kv_stage 3→2→1 = 1275 → 1229(−3.6%) → **死锁**（解的 K 活性 2+V 1=3 是硬下界，跌破即挂——
模型预言的悬崖位置精确成立）；split_P_arrive 96→64→32→0 = 1275→1247→1221→1191 严格单调
（重叠相位旋钮总价值 −6.6%）。

### 10.2 性能表（TFLOPS；fwd 4·BHS²D / bwd 10·BHS²D）

**fwd**：
| bar | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| FA4 stock（2-CTA，论文字面基线） | 962 | 985 | 1021 | 1047 |
| fa4_1cta（FA_DISABLE_2CTA=1，B1 公平靶标） | 1123 | 1223 | 1246 | 1275 |
| **skc_cute（solver 配额绑定）** | ~1172† | 1216 | **1259** | **1280** |

† 2048 复测中位；分布双峰见 §24.1。

**bwd**：
| bar | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| FA4 stock（2-CTA） | 886 | 1037 | 1095 | 1128 |
| fa4_1cta | 847 | 963 | 1020 | 1052 |
| skc_cute（恒等；bound −1.5% 略差） | 848 | 963 | 1020 | 1042 |

### 10.3 within-2% 判定与 B2 残差

- **B1 规则（vs fa4_1cta）：fwd PASS 且反超（+0.4%）；bwd PASS（−1.0%）。**
- 对论文字面 stock FA4：fwd **+22%**（超过）；bwd −7.6%。
- **B2 残差双向**：fwd 2-CTA 在此栈/形状是**负优化**（stock 比 1-CTA 慢 17–24%）；bwd 2-CTA
  真实获益 +7%。论文自述 CUDA 13.0 ⇒ 其 fwd FA4 基线极可能就是这条被 2-CTA 拖慢的路径——
  §7.5 的未披露项由此升级：论文 fwd within-2% 是对一个在其自己栈上非最优的基线的比较。
- bwd 1-CTA 路径（上游从未运行过）正确性全过，R1 风险退役。

### 10.4 结论

Phase B 把 Phase A 的天花板问题（TLX 0.81×）终结：**在 B1 忠实度规则下，论文的性能主张
在我们的复现中成立且被超越**——手段正是论文框架自己的用法（调度模型认证专家操作点 + 在
模型可表达的参数面上寻优）。同时复现暴露：性能主张的语义强依赖基线配置（2-CTA 双向残差），
这是论文欠定处清单（§7.5）的最重条目。
