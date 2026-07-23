# SKC 设计方案：Schedule-Directed Kernel Compiler
## （把论文的"专家手工编译"机械化为可重复的调度制导编译器）

> 背景：复现中 JOS 调度经通用 emitter 落地仅得 215 TFLOPS（FA4 的 0.21×），而论文的路径是
> 专家把求解器输出的流水线+warp 标注 IR **手工编译**成 CUDA C++ 并补齐正交优化。
> 本方案回答：能否"手写一个编译器"——即把那位专家的做法固化成程序。
> 结论：能，且仓库里已有 ~80% 的地基。但必须分两阶段管理预期（§2）。

---

## 1. 证据驱动的设计需求（每条来自 GAP_ANALYSIS.md 的实测归因）

| 归因（实测） | 设计需求 |
|---|---|
| #1 划分抽取把 TMEM 卸载+rescale 压上 MMA 发射组（barrier 停顿 4.6×） | **R1** MMA 发射者必须是 1-warp 专职任务（24 regs），只发 tcgen05 与 barrier，永不碰 TMEM 卸载 |
| #2 寄存器超订（21 warps/CTA → 152→104 缩放 → 880B 溢出） | **R2** 按 role 定额寄存器（计算组 152/168、基建组 24/88），编译期证明 ΣCTA ≤ 64K，不缩放 |
| #3 skew 环被弃 → 组内无流水（且 skew 发射路径有死锁 bug） | **R3** 组内软件流水由骨架**构造保证**（显式多缓冲+相位），不依赖事后 skew 变换 |
| #4 合成通道走 SMEM 往返 + 死代码组 | **R4** 跨组标量/向量（alpha/m/l）走 **TMEM tile 协议**（真实 FA4/TLX 手写内核的做法）；role 全覆盖无死组 |
| #5 环深非瓶颈（负结果） | 深度作为 solver 参数直通即可，不需要新机制 |
| 通用 emitter 的失败模式：逐 op 翻译 + 事后综合同步 | **R0（总纲）** 不做逐 op 翻译；做**骨架实例化**：专家玩法库 + 调度参数绑定 |

## 2. 目标与预期管理（诚实的天花板分析）

- **Phase A（本方案主体）**：JOS 调度 → TLX 骨架编译器。目标 = 达到 TLX 手写内核天花板
  （repo 实测/回归门：blackwell_fa_ws 家族 651–720 TFLOPS 级）。收益 215 → ~700（**3.3×**），风险低，
  全部基建复用。**但 TLX 手写天花板 ≈ 0.63–0.69× FA4(1047)** ——论文的 within-2% 在此阶段不可达。
- **Phase B（可选二期）**：同一编译器换后端到 **CuTe-DSL**（FA4 官方所用，本地已装 nvidia-cutlass-dsl，
  vLLM vendored flash_fwd_sm100.py 全源可参考）。按忠实度分两档：
  - **B1（保调度正交优化，论文方法论内）**：TMA multicast（论文 §6.2.1 自己在手工实现中使用过的 cluster
    特性）、persistent/CLC 外层循环、指令选择、布局/softmax 优化——算子集合与内层调度结构不变，
    实现的仍是求解器算出的单 SM 调度。
  - **B2（改调度优化，默认排除）**：2-CTA tcgen05（CTA-pair MMA）等——tile 分解与同步拓扑改变后，
    实现物不再是求解的那个调度，1:1 关系断裂。论文模型不含 cluster（全文零提及），仅在用户明示
    "接受超出论文的扩展"时启用并记偏差表。
  - 预期管理注记：FA4 内部若受益于 2-CTA，排除 B2 后对 FA4 的可达上限存在结构性残差；同时这暴露论文
    自身欠定处——未披露其 Blackwell 手工实现是否使用 CTA-pair，而 within-2% 声明与此直接相关（记入报告）。
  工作量约为 Phase A 的 2–3×。
- **非目标**：通用任意图编译（保持与论文同等的诚实范围——论文也只手工编译了 FMHA fwd/bwd 两个内核族；
  我们做的是"每内核族写一个骨架"，族内由 solver 参数驱动）。

## 3. 架构（Phase A）

```
solver 解 (JSON: II, cycles, warp, depths)
   │
   ├─ 1. RoleClassifier  —— 把 solver 的 warp 组映射到骨架 role：
   │      {VL-load, MMA-issuer, softmax×k, correction, epilogue}
   │      · 依据：组内 op 类型指纹（与 strategy_report.classify 同源）
   │      · 不可映射 → 显式拒绝并回馈 solver（见 §5），绝不静默降级
   ├─ 2. ScheduleBinder  —— 从 cycles/II 提取骨架参数：
   │      · 各 role 任务体内的发射序（cycle 排序 → 程序序）
   │      · ping-pong 相位（两 softmax 组的 exp 相位差 → 交错启动偏移）
   │      · 环深（K/V/QK/P 各缓冲 depth ← solver liveness 或 sweep 参数）
   │      · prologue 深度（⌈L/II⌉ − 1 级预热）
   ├─ 3. SkeletonInstantiator —— 模板化 TLX 代码生成（Jinja 级文本模板 + 片段库）：
   │      · fwd 骨架：以 third_party/tlx/tutorials/blackwell_fa_ws.py 为基底
   │        （default=correction / softmax replicate=k / mma 1-warp@24 / load 1-warp；
   │         qk/p/alpha/l/m/acc 全 TMEM tile；barrier 协议照抄——这是已验证 651+ 的同步拓扑）
   │      · bwd 骨架：以 case4 handwritten.py（fa_bwd_dkdv_ws）为基底
   │      · R2 寄存器定额表编译期校验；R3 多缓冲显式展开
   └─ 4. Verifier —— 三重门：
          · 正确性（vs SDPA，沿用现有 harness）
          · 结构断言（生成代码的 barrier 拓扑 == 骨架协议；无未绑定参数）
          · 性能门（≥ TLX-Default 724 为过门线；目标 ≥ 手写基线 651）
```

## 4. 与通用 emitter 的本质区别（为什么这次能行）

| | 通用 emitter（sched2tlx） | SKC |
|---|---|---|
| 翻译单位 | 逐 op | 骨架（专家玩法整体） |
| 同步 | 事后从图综合 mbarrier | 骨架自带已验证协议（651+ TFLOPS 实证） |
| 跨组数据 | 事后合成 SMEM 通道 | TMEM tile 协议（FA4 同款） |
| 寄存器 | 事后缩放（152→104+溢出） | role 定额，编译期证明 |
| 组内流水 | 事后 skew 变换（有 bug） | 构造性多缓冲 |
| 适用面 | 任意图（低质量） | 内核族（高质量）——与论文的手工编译同范围 |

## 5. 与求解器的闭环（论文框架的正确用法）

骨架的 role 系统反向定义了 solver 的"可表达族"约束——**已有钩子直接可用**：
- role 模板 = `colocate`（MMA 集合、每条 softmax 链）+ `separate`（softmax↔softmax、MMA↔softmax）
  ——即本次复现中 FA4-exact 探针用过的那组约束，作为默认开启；
- R2 定额 → 现有 REGISTERLIMIT 的 per-role 预算参数；
- 由此 solver 输出**必然可被 SKC 实例化**，"图上可行、落地劣化"的类别性问题被结构性消除。
- 审计口径：这与论文 §5 的定位一致——求解器输出交给"实现指定 WS 策略的下游编译器"，
  SKC 就是那个下游编译器，只是我们把它写出来了。

## 6. 里程碑与验收

| | 内容 | 验收 |
|---|---|---|
| M1 | fwd 骨架提取参数化（从 blackwell_fa_ws.py 重构为模板，默认参数） | 默认参数实例 ≥ 651 TFLOPS（等价原内核），正确性 PASS |
| M2 | RoleClassifier + ScheduleBinder 接 JOS 解（FA4-exact 抽取） | JOS 参数实例正确性 PASS；≥ TLX-Default 724 |
| M3 | 参数敏感性：solver 环深/相位 vs 默认值 A/B | 报告 solver 参数的净贡献（这是论文没做的增量实验） |
| M4 | bwd 骨架（case4 手写基底）+ jos_bwd bar 补全 | 图 11 的 JOS bar 出数 |
| M5（Phase B 决策点） | 若需 within-2%：CuTe-DSL 后端立项 | 单独设计评审 |

预估工作量：M1–M2 各 1–2 天当量；M3 半天；M4 2 天（bwd 拓扑复杂）。风险最大项：M2 的
role 映射对 solver 抽取的鲁棒性（缓解：§5 的闭环约束默认开启后，抽取空间已被模板收窄）。

## 7. 风险

1. TLX 手写天花板不足以支撑论文数字（已知，Phase B 兜底）。
2. 骨架参数化过程中破坏原内核性能（M1 验收即防线：默认参数必须复现原性能）。
3. solver 相位/发射序与骨架静态结构冲突（ScheduleBinder 只取"可绑定子集"：发射序、相位、深度；
   不可绑定的 cycle 细节明示丢弃并记审计——与论文把调度交给人工实现时的信息损失同类）。

## 8. 实施结果（2026-07-22；Phase A M1–M3 完成）

实现于 `skc/`：`skeleton_fwd.py`（协议参数化骨架，全部绑定量为 constexpr，无文本模板）、
`roles.py`、`binder.py`、`instantiate.py`、`__main__.py`（CLI），`tests/test_skc.py` 4 项 CPU 单测。

- **M1 ✅**：默认参数实例 871 TFLOPS @16K，同 harness 原版内核 839——"复现原性能"成立
  （651 门大幅通过）。正确性 PASS 且 M_lse 输出补齐。
- **M2 ✅**：FA4-exact 解绑定实例 = {几何 BN=64（DDG tmem_load shape 提取）、KV=3（K 活性 2 + V 1，
  与手调默认重合）、QK=2（跨 II 活性）、**PV skew=1（稳态 mod-II 发射序）**、R2 配额表}，
  844 TFLOPS @16K，> TLX-Default 724 门；比通用 emitter 同解落地（215）**3.9×**。
- **M3 ✅**：解参数净贡献 @16K——skew **+70**（解的核心洞见，硬件实证）；QK 深度单独 ≈0；
  解几何 BN=64 −91（成本模型未定价的实现开销）。骨架默认 871 仍为 Phase A 最优。
- 绑定期发现并修复的骨架推广问题：acc 环与 QK 环解耦（acc 原地累加不随深度分裂）、
  alpha/p/l/m 槽位公式一般化（QK=1/BN=128 时与原版逐位一致）、skew 下 correction 需显式等
  acc_empties(i−1)（原协议的隐式 TC 序在 QK 预发射后失效——即通用 emitter skew 死锁 bug 的根因，
  此处给出正确同步）。本 build 约束：TMEM 64 行 tile 不支持 → SUB_M 64→128 保调度放大（记审计）。
- **验收对照**：R0–R4 全部落实；"图上可行、落地劣化"类问题未再出现（emitter 归因杠杆
  #1/#3/#4 结构性消除）。paper within-2% 仍未达（0.81× FA4）——Phase A 天花板结论成立，
  升级路径为 M5/Phase B（CuTe-DSL 后端）。
- 工件：`skc_fwd_default.py` / `skc_fwd_jos.py` / `skc_fwd_var_bn64.py` / `skc_fwd_var_qk2.py`
  （自带绑定审计头），bench 新 bar `skc_default`/`skc_jos`/`skc_var_*`，数据入
  `bench/results_fwd.json`。
- **M4 ✅**（bwd）：`skeleton_bwd.py`（case4 手写基底参数化）+ `classify_bwd`/`bind_bwd`；
  图 11 JOS bar 出数 236 TFLOPS @16K（原 emitter 版 SKIP）。家族求解（§5 闭环实操）：
  v6 自由解按 R0 拒绝 → 判别探针序列定位——R1 专职 MMA 组零 II 代价（SAT 于自由最优同点
  (95,273)），单体 compute 任务被模型拒绝（{exp2, dpT-load, dS-sub} 三方共置真 UNSAT，
  预言了手写 236 < emitter 分摊 269 的硬件差）；两处模型账目差钳制并审计（64 行 TMEM tile、
  ~17KB SMEM 开销）。详见 REPORT.md §9.1。M5（Phase B CuTe-DSL）待评审。
