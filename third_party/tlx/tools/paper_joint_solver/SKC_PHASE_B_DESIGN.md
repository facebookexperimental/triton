# SKC Phase B 设计：CuTe-DSL 后端（M5 评审稿）
## 把 within-2%-of-FA4 变成一个可判定、可证伪的实验

> 设计过程：4-agent 源码勘察（FA4 fwd/bwd sm100 内核 + pipeline/launch 基建，全部行号取证）
> → 3 个独立方案（就地绑定 / 忠实 fork / 自写骨架）→ 3 个对抗评审 → 本合成稿。
> 评审共识：**就地绑定（subclass shim，零 fork）胜出**；自写骨架被其设计者自证 dominated
> （Phase A 教训：协议质量主导，3.9× 提升全部来自复用已验证协议）；fork 的增量价值不抵
> SASS 校验不可达 + helper 漂移盲区两处硬伤。

---

## 0. 三个决定性事实（勘察结论，全部有行号）

1. **FA4 fwd 在 hd128 fp16 非因果的默认路径是 2-CTA**（`interface.py:589-603`：cluster (2,1)、
   tcgen05 CtaGroup.TWO、MMA tiler M=256 跨双 CTA；fwd **不用** TMA multicast，源码注释
   `# no multicast`）。唯一自动禁用条件是 CUDA 12（`utils.py:70-88`）；**论文自述 "All
   experiments use CUDA 13.0" ⇒ 其 FA4 基线即 2-CTA 路径**，而其手工内核是否用 2-CTA 未披露
   （REPORT §7.5）。厂商自带开关 `FA_DISABLE_2CTA=1` 给出**同一内核的 1-CTA 变体**。
2. **FA4 的结构与我们求解器的裁决收敛**：fwd = 专职 1-warp MMA 发射者(w12) + 双 softmax
   warpgroup + correction 组 + TMA load——即 FA4-exact 解的 W=5 结构；bwd = 专职发射者 +
   **compute 摊 2 个 warpgroup** + reduce 组——正是 M4 判定（R1 零代价、单体 compute 真 UNSAT）。
3. **FA4 的参数面几乎为 ScheduleBinder 而生**：`q_stage/kv_stage/s_stage`（环深，SMEM 公式派生，
   1-CTA hd128 时 kv_stage=3——与我们 fwd 解的 KV 活性=3 相同）、寄存器配额表 `_TUNING_CONFIG`
   （源码自注 "agent-editable"）、`split_P_arrive=96`（子迭代相位点）、warp role 元组。
   且 **1-CTA 非因果 hd128 的配额键 `(False,False,128,False)` 在表里缺失**——上游没调过这个点，
   落到 192/80/48 默认——这是求解器在专家自己骨架上可能击败专家默认值的唯一真实杠杆。

## 1. 可证伪性设计（评审团最重要的修正）

三个评审一致指出原始方案的头条实验是**准同义反复**：binder 把求解器解映射回 FA4 自己的 1-CTA
默认参数点、跑在 FA4 自己的代码里，within-2% "by construction" 必过。修正为三层证据，各自可失败：

- **E1 收敛层（VERIFY，无 GPU）**：断言求解器独立推导的量 == FA4 专家值：kv_stage 3、s_stage 2、
  q_stage 2、专职发射者、bwd Q_stage 2 / compute 摊 2 组。任何一条不等即 FAIL——
  "调度模型从第一性原理推出专家操作点"是论文的真正论题，此层是其可判定形式。
- **E2 绑定层（可失败的性能主张）**：唯一真实杠杆 = R2 寄存器配额绑定到未调优的 1-CTA 键上
  （对照 192/80/48 默认）+ kv_stage 降钳扫描 + `split_P_arrive` ∈ {32,64,96} 扫描。
  预注册预测见 §8；solver 配额**跑输**上游默认即为负结果，如实报。
- **E3 序数层（off-optimum 排序）**：对每个可绑参数做扰动矩阵，检验**性能退化方向与模型预测的
  排序一致**（例：kv_stage 3→2→1 应单调劣化且 1 显著差于 2；split_P_arrive 96→0 应劣化）。
  模型预测排序错误即 FAIL。这是"模型懂硬件"的主要证据，取代不可失败的头条。

## 2. 对照设计（B1/B2 裁定的落地）

| bar | 配置 | 角色 |
|---|---|---|
| `fa4_2cta` | 原样 FA4（我们已测 fwd 1047 / bwd 1141） | 论文字面基线；B2 残差的分子 |
| `fa4_1cta` | 同一 FA4 + `FA_DISABLE_2CTA=1`（子进程 env，import 前置） | **B1 公平主对照**（厂商自带 1-SM 变体，非稻草人：CUDA-12 build 就跑它） |
| `skc_cute` | shim 绑定实例（1-CTA） | 被测物 |

判定口径：**within-2% 主张在 B1 规则下针对 `fa4_1cta` 检验**；`fa4_2cta − fa4_1cta` 差额
作为"B2 归因残差"与判定并列报告（预注册，防摘樱桃质疑）——它就是论文单 SM 模型无法表达的
那部分硬件收益的实测定价。诚实声明：若审稿人拒绝对 "FA4" 的 B1 重释，本设计对论文字面主张
不给出判定（fidelity charter 的必然代价，写入报告）。

## 3. 架构（就地绑定，零 fork）

新包 `third_party/tlx/tools/paper_joint_solver/skc_cute/`：

```
fa4_pin.py       — pin flash-attention commit + 整个 flash_attn/cute/ 源树指纹
                   （评审修正：不是挑 6 个文件——未 pin 文件漂移会静默改行为）；
                   驱动 import 时校验，不匹配 → 硬拒绝
binder_cute.py   — bind_fwd/bind_bwd：消费冻结解 JSON（subtiled_fa4exact / bwd_skc）
                   → binding JSON {overrides, verifies, dropped, audit}；
                   CPU 侧 validate() 编码全部静态不变量：
                   fwd 512==2*sm+corr+other、TMEM 2*n_block+q_stage*hd_v<=512、
                   s_stage>=q_stage、split_P_arrive%32==0；bwd reduce+2*compute+max(load,mma)<=512；
                   仅接受 hd128（评审修正：kv_stage 钳制在 hd192/128 有 uneven_kv_smem
                   派生态残留，非 hd128 一律硬拒绝）
shim_fwd.py      — class SKCForwardSm100(FlashAttentionForwardSm100)：
                   两个安全写点——__init__ 尾部（配额、split_P_arrive、s0_s1_barrier）
                   + 重载 _setup_attributes() 于 super() 后（kv_stage 降钳/s_stage 校验）。
                   写点位于双编码（SharedStorage 尺寸 + pipeline.create）两处消费者的上游，
                   结构性消解 dual-encoding 风险（评审 source-check 确认）。
                   评审修正：is_persistent/q_stage 是 interface 层计算的 ctor 参数，
                   shim 层改会与 launch 几何脱钩 → 一律 VERIFY-only
shim_bwd.py      — 同法：__init__ 尾配额；_setup_attributes 绑 Q_stage
driver.py        — install(binding)：monkey-patch interface 模块全局符号
                   （评审确认 interface.py 按名解析类，patch 点真实有效）；
                   断言 disk cache 未启用（cache_utils.py:35 in-process dict）
                   → 每子进程一 binding，compile_key 无跨绑定串染
bench/skc_cute_worker.py — 复制 fa4_worker 协议 + --binding + --disable-2cta；
                   审计哈希回显进计时 JSON（绑定与测量密码学挂钩）
tests/test_skc_cute.py — CPU：pin 校验、不变量、审计三态完备性（bound/verified/dropped）
```

**绑定分类学**（每参数四态 + 拒绝语义）：`BIND`（配额、kv_stage 降钳、split_P_arrive）/
`VERIFY`（kv=3、s=2、q=2、专职发射者、tile 128×128、bwd Q_stage）/
`DROPPED`（II/L 精确 cycle——与 Phase A 同类的移交信息损失，审计留痕）/
`FROZEN`（warp 元组：named-barrier 索引算术硬编码 4-warp 组；MMA 发射序：融合三役 mbarrier 环
+ 蓄意省略的 O-full 信号以静态序为正当性；`producer_tail` 上游自己注释 "hangs"——只继承不修）。
语义注记入审计：solver 的双链是**一个** Q block 的子块，FA4 q_stage=2 是**两个** Q block
ping-pong——结构同、依赖粒度异，故 VERIFY 而非 BIND。

## 4. 里程碑与验收

| | 内容 | 验收门 |
|---|---|---|
| M1 | pin + binder + shims + CPU 测试 | 单测全过；E1 收敛断言全部成立（无 GPU） |
| M2 | **恒等门**：空 override 的 shim 实例 vs `fa4_1cta` | 差 < 噪声（隔离 shim 开销；此门失败则后续全部无效） |
| M3 | `fa4_1cta` 两 bar 出数（fwd+bwd 4 seqlen）+ 正确性 | bwd 1-CTA 是上游**未曾运行**的路径（无 CUDA-12 回退）——正确性门失败则触发 R5 预案 |
| M4 | 绑定实例出数：`skc_cute` fwd/bwd | E2 判定（vs `fa4_1cta` within-2%？配额绑定是否胜过上游未调默认？） |
| M5 | E3 扰动矩阵（kv 3→2→1、split_P 96→64→32→0、bwd Q 2→1、配额 ±16） | 模型预测排序与实测排序一致；每点 120s watchdog + killgpu.sh，挂=弃点不弃法 |
| M6 | 校准套件 `calib/`：CuTe-DSL 微内核测每 op 成本 → `solver_costs_cute.json` 重解 | 重解结构不变（或变化被解释）——自写骨架方案的唯一存活遗产 |
| M7 | 报告：三层证据 + B2 残差定价 + 论文 §7.5 未披露项的量化收口 | 文档合入 REPORT.md §10 |

预估：M1-M2 各 1 天当量；M3-M4 各 0.5 天（排 GPU）；M5 1-2 天；M6 2 天（可选）；M7 0.5 天。

## 5. 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | bwd 1-CTA 路径上游未运行过（正确性/挂起未知） | M3 独立正确性门；失败预案：bwd 仅报 vs `fa4_2cta` 并声明残差不可归因，within-2% 对 bwd 不判定 |
| R2 | compute-sanitizer 不覆盖 TMEM/tcgen05 异步代理/mbarrier 相位（静默错果而非挂起） | 每个扰动点跑 SDPA 残差门（不是只在默认点）；错果=弃点 |
| R3 | 上游漂移 | 全源树指纹 pin + 硬拒绝；venv 内安装与 clone 的一致性在 pin 校验里显式检查 |
| R4 | 编译缓存把 2-CTA 工件误标 1-CTA | 每子进程一 binding + disk-cache-off 断言 + 从编译对象反查 cluster shape 记录进 JSON |
| R5 | within-2% 在 B1 下"必过"的同义反复质疑 | §1 三层证据结构；预注册预测（§8）先于测量存档 |

## 8. 预注册预测（先于任何 GPU 运行存档）

1. `fa4_1cta` fwd 落在原版 1047 的 85–95%（失去 M=256 CTA-pair 摊销 + kv_stage 6→3）；
   bwd 残差更大（2-CTA 是 bwd 默认且有专职 relay warp）。
2. `skc_cute`（恒等绑定）与 `fa4_1cta` 差 < 1%（by construction，故不作为证据）。
3. 配额绑定（E2）：**有真实机会 +1~3% 超过 `fa4_1cta`**（上游 1-CTA nc 键未调优）；若跑输，
   如实报负结果。
4. E3 排序：kv_stage 单调、split_P_arrive 单调、bwd Q_stage 2>1——模型与硬件排序一致。
5. B1 规则下 within-2%（vs `fa4_1cta`）预计 **PASS**；对原版 2-CTA FA4 的字面主张残差
   预计 5–15%（fwd），全部归因 B2——即论文单 SM 模型的表达边界的实测定价。
