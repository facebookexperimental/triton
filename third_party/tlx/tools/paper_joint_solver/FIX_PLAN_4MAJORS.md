# 修复计划:4 个 major 论文一致性缺陷

基于 2026-07-27 的六维度忠实度审查(每项发现经对抗复核确认),计划本身另经 3 个
批判者按真实代码与 dump 实证修订。文中 file:line 以当前分支 tip(`d07c30f40`,
`paper_joint_solver/` 与审查时逐字节相同)为准。

待修复的 4 项(均为已确认的 major):

1. handoff IR 的 `cross_warp_dependencies` 按 group 粒度分类,而论文 §4.3 与本仓库
   求解器/emit 门禁均为物理 warp(lane)粒度;
2. IR 未物化论文 §5 意义上的 software-pipelined 程序(⌈L/I⌉ 份展开被留给专家);
3. streaming(§5.3)在 forward 输入上永不触发(任意入边即取消资格,含标量地址
   算术与 signal-only token 边);
4. CONCURRENCY 窗口对 TC/TMA 有未披露的松弛(`win=1` 而非论文 Figure 6 的
   `cycles(o)`),且已存档解中含论文约束禁止的放置。

---

## 前置事实(批判阶段实证确认)

- **基线是红的**:`tests/test_ddg_and_modulo.py` 有 2 个确定性失败——
  `test_joint_lane_symmetry_orders_membership_columns`(:307,lane 列未排序)和
  `test_joint_uses_producer_specific_spill_cost`(:847,期望 sat 实得 unsat,疑似
  aa4 窗口收紧提交时回归)。后者恰在 Fix B 的地盘上。
- **9 个在库 `*_solution*.json` 全是 legacy schema**,测试断言它们抛
  `LegacySolutionError`,`refit_check.py:58` 还读取其一——**重跑结果必须写新文件名,
  严禁覆盖**。
- pre-commit 对 `third_party/tlx/tools/` 全局排除,不会格式化本目录。

---

## Phase 0:基线与分诊(半天)

1. 按 `run_ablations.sh` 的 `SOLVER_LIB_PATH` 约定设置求解器动态库路径,跑
   `pytest tests/ -q`,记录基线(预期 2 failed / 129 passed)。
2. **Phase 0b(必做)**:分诊上述 2 个红测试——修复或显式 `xfail(reason=...)` 并
   记录。否则 Fix B 的回归与既有损坏无法区分。
   `test_joint_uses_producer_specific_spill_cost` 的 unsat 要先弄清是窗口收紧的
   正当结果还是编码 bug。
3. 顺手把 `importorskip` 模式改为同时捕获 `YicesAPIException`(可选,1 行)。

---

## Phase 1:求解器侧(Fix A + A2 + B,可与 Phase 2 并行)

### Fix A:streaming 判定(`ddg.py:369`)

改 `has_incoming = {edge.dst for edge in edges}` 为:

```python
has_incoming = {e.dst for e in data_edges if e.src not in prob.emitter_infra}
```

`data_edges`(:356)已排除 signal-only;`emitter_infra`(:350-354)来自 baseline
graph 的 `warp_group < 0` 标记(fwd 两个 dump 中就是 muli/addi 节点 0/1)。依据:
论文 §5.3 的 G(Figure 9)里没有地址算术节点,TMA load 是图源点。

**批判者实证修订:**

- 归一化池**不会**移动——只有两条 556 延迟出边被清零(fwd 12→0、subtiled 1→0),
  `normalization_f` 三个 dump 全不变。旧解由 `solver_sources_sha256` 翻转 + 这两条
  边失效,不是全局 C′ 漂移。
- bwd 有惊喜:`tt.descriptor_reduce`(id 35)也被标 infra(tile 级 op!)。今天无害
  (零出边),但要加**不变量断言**:被豁免的 infra 生产者只能以标量结果喂 VL 节点;
  并加 bwd 回归测试钉住 `streaming=={0,2}`(7、9 因真实的 `tt.addptr` tile 指针
  入边被正确排除)。
- 测试:`test_any_incoming_dependence_disqualifies_streaming`(:701-739)**扩展而非
  重写**(它的入边是真数据边,Fix A 后照样通过);`test_load_and_derive` 里 :66 的
  内联 oracle 复刻了旧谓词,要同步更新;需要新建带 `warp_group<0` 节点的
  schedule_graph fixture 辅助函数(现有 `_write_ddg` 不支持,无 conftest)。
  正确引用:清零测试在 :670-698。
- **旁路调用方**:`viz.py:160`、`strategy_report.py:68`、`desym_check.py:23` 不传
  baseline graph → Fix A 后它们渲染的模型与求解模型分叉。给它们穿透
  `--baseline-graph`,或在 emitter_infra 为空但 DDG 有 infra 候选时大声警告。

### Fix A2(新增决策点):infra 边的 WS 语义豁免

`joint_smt.py:841-848` 的 CONCURRENCY spill-gate 把 infra 生产者也算作寄存器前驱
(fwd 中 `regs[1]=1`,且节点 1 因 VARIABLELATENCY iff 永远不可能与 load 同组 →
gate 恒真);`joint_smt.py:793-824` 的 CROSS-WARPSPILLS 同样给 infra 边 1→2 计
spill。Fix B 去掉 carve-out 后,这会在 W_vl 上给两个 TMA load 制造论文里不存在的
12 周期互斥窗(论文的 G 中 load 是源点,gate 永不触发)。

**建议决策(推荐前者)**:与 Fix A 同一原则——infra 边只保留 DEPENDENCE 排序语义,
**豁免全部 WS 语义**(streaming 取消资格、CONCURRENCY gate 前驱、CROSS-WARPSPILLS
计价),在 FIDELITY_REVIEW 披露为 adapter 规则;或保留并显式披露偏差。必须在
Phase 3 重跑前定,它改变 fwd 非 subtiled 的解。

### Fix B:CONCURRENCY 窗口(`joint_smt.py:860-861`)

```python
win = prob.lat[o]          # 论文 Fig 6:[t-(cycles(o)-1), t],无条件
if win == 0: continue      # cycles(o)=0 → 空窗(论文公式自身行为)
```

删除 :854-859 的 Fig-2 论据注释,换成论文公式引用。可选:把旧行为放进显式非论文
旗标(默认关)供消融。

**批判者实证修订(重要)**:这**不是纯收紧**——subtiled case 有 45/55 节点
`lat==0`(25 个原生 0 + 20 个被 ZLP 归一化截断到 0,`normalize.py:48` 的 lb=0 与
论文 C′≥0 一致)。现行代码给它们 `win=max(1,lat)=1`(禁止同刻共发射),
skip-on-zero 则完全豁免——所以 Fix B 同时收紧 TC/TMA、放松约 82% 的 op。结论不变
(这就是论文公式的字面语义),但:

- FIDELITY_REVIEW 必须把放松的一半也披露;
- 删掉"可行域缩小"的预期;旧 (II\*,L\*) 可能仍 SAT 而非变 UNSAT;
- 加测试:lat-0 op 在 t_v 被接受(新行为)+ lat≥1 在窗内被拒(回归,复现已存档
  反例的形状:同 warp MMA lat=2 @ t−1 vs blocking tmem_load @ t)。

**Emit-gate 加固(强烈推荐)**:在 `schedule_plan.py` 的门禁(现有 COMPLETION /
DEPENDENCE / CAPACITY / VARIABLELATENCY 旁)加直线程序上的 CONCURRENCY 复检,语义
与求解器**逐字节相同**(含 skip-on-zero 和 A2 决策)。O(V²·copies²),V≤55 无成本。
**先 dry-run** `test_skc_cute.py` 的两个手工 fixture 构造器(:104-156、:249-284——
它们的 cycle 从未经过 CONCURRENCY 检查),若违窗则把 fixture 重排纳入本任务,避免
实现中途翻车。

---

## Phase 2:IR 侧(Fix C + D,单次 v2 schema bump)

### Fix C:lane 粒度的 cross-warp

1. **先规范化 lane**(批判者发现的前置缺陷):solver 恰好输出有序 tuple,但下游
   无人强制。在 `schedule_plan.py:87`(`_lane_dict`)排序 + `_schema.py` v2 的
   `_validate_instructions` 要求有序。此后 tuple 相等 ≡ 集合相等,顺带修掉现有
   gate 的"置换 lane 幻影 spill"分叉。
2. `ScheduledEdge`(:109-118)加 `producer_lanes` / `consumer_lanes` / `spill_cost`;
   `spill_cost` 对 signal-only 边置 0(镜像 `joint_smt.py:798` 的豁免——现 gate
   :391-397 对 token 边多算 spill,一并对齐并记录)。
3. `pipelined_ir.py:82-88` 谓词:
   `producer_group != consumer_group or producer_lanes != consumer_lanes`。
4. `_schema.py` v2:验证 lane 字段、按 lane 谓词重算 cross 集合;`spill_cost >= 0`、
   非 cross 边必须为 0;把 :434-437 的时序检查强化为
   `available >= cycle[src] + latency + spill_cost`(否则 spill_cost 是死数据)。
5. **audit.py 的真实工作量**:channel/program_order 规则**自动**跟随 cross 列表
   (都 keyed off `cross_dependency_map`),无需改;真正要改的是
   `_mapping_dependencies`(:427-434)和 `_sync_dependencies`(:972-979)的
   `facts` 元组纳入新字段,以及 `scaffold.py:217-236` 的 `_sync_template` 显式枚举
   新字段(否则专家手册里字段过期也能过审)。
6. **schema bump 双点同步**:`pipelined_ir.py:20-21` 与 `_schema.py:11-12` 是独立
   字面量——同一提交内一起 bump(或让 `_schema` 导入前者),可在 `skc/compiler.py`
   加 import 时断言二者一致;`SKC_DESIGN.md:34,36` 同步。solution 的输入 schema
   **不动**(输入格式未变)。

### Fix D:物化流水展开(`pipelined_program` 节)

`_build_ir` 新增(纯调度算术,不越 §6.1 边界——批判者确认):

- `instances`:`{node, copy, cycle: node.cycle + copy*ii, region}`,
  copy ∈ [0, copies),即论文 Figure 3 的 op-table;
- `instance_dependencies`:消费者 (v,i) → 生产者 (u, i−δ);越界(j<0)记
  `external + carried_distance`(δ≥0 已由 schema 保证,只可能下越界);
- `steady_state.slots`:稳态窗内每节点恰一实例,
  `iteration_lag = (copies-1) - copy == stage`,即 Figure 1f 的 `V[i-1]` 重命名事实。

`_schema.py` v2 从 (nodes, ii, copies, regions) 整体重算展开并**对象相等**比较
(与现有 region 重算同模式,:280-281——不是字节比较)。

**测试性质(批判者修正)**:steady 恰含 |nodes| 个实例、每节点一次且 lag==stage;
prologue+epilogue 合计 |nodes|·(copies−1)。(原"region 人口=窗宽"性质不成立,
量纲都不同。)

**测试清单(两修合并)**:`test_skc.py:337` / `:554` 的机制选择辅助函数改 lane
谓词;v1 字符串硬编码点 `test_skc.py:84,197-199,625-626,701-703`、
`test_skc_cute.py:178,192`;`test_skc_cute.py:287-295` 扩展为混合 lane 边必须进
cross 列表且 sync 审计要求 channel。注意 `test_skc.py` 的"fixture"是代码内 dict
构造器(`_ir()` :73),改构造器而非重生成文件。

---

## Phase 3:重跑求解(后台,串行,每 case 独立日志)

命令逐 case 钉死(provenance 会锁 `normalization_u` 和 machine manifest):

| case | 命令要点 |
|---|---|
| fwd subtiled | `--baseline-graph …subtiled/schedule_graph.json --warp-fixed-overhead 4` |
| fwd 非 subtiled | 同上 + **决定** `--normalization-u 150`(沿历史)或披露改用默认 300 |
| bwd | case4 的 ddg + graph |
| bwd-LR | **选定并披露** `--reg-budget < 8160`——legacy 的 8192 现 CLI 直接拒绝(上限 32×255=8160),且 8192=256/线程根本不算"reduced";论文只说"reduced budget"无数值 |

外加重跑 `run_ablations.sh` 的 4 组 UNSAT 消融(warp 参数按 FIDELITY_REVIEW U5 先
更新)。显式保留 `--ilp-seconds/--smt-seconds/--max-wall-s`。历史耗时 4–17 分钟/case
(254–990 s),且全部早于 full-L-window 默认——**按更慢预算**,串行跑
(refit_check 的 150 GiB RLIMIT 先例说明内存有压力)。**输出一律新文件名**
(如 `solutions/<case>_v7.json`),并决定 `refit_check.py` 的去向(更新期望元组,
或注记 CHANGE 是预期结论)。

---

## Phase 4:端到端验证(审计必须"拒绝"而非"通过")

对每个新解:

1. `python -m skc handoff --solution … --ddg … --baseline-graph … --ir-out … --manifest-out …`
2. `python -m skc scaffold --ir … --handoff … --out-dir …`
3. **负向检查:各 audit 必须拒绝草稿**(`test_skc.py:747` 已把"scaffold 产物永不
   过审"钉为规范;让 audit-bundle 通过意味着伪造专家批准,恰恰违反论文边界)
4. 全量 pytest。

audit 通过路径只由测试中的合成"已批准"fixture 覆盖。

---

## Phase 5:文档与收尾

- `FIDELITY_REVIEW.md`:删除已过时的 U6/U9 文本;新增披露:恢复 Fig-6 精确窗口 +
  zero-lat 空窗的放松面、streaming 谓词定义(data 边 + 非 infra 生产者)、A2 的
  infra 边 WS 豁免(若采纳)、lane 粒度 cross 列表、`pipelined_program` 物化、
  `spill_cost` 的 signal-only 置零。
- `REPORT.md`:历史表标注"旧模型",新增新解表格;`SKC_DESIGN.md` schema 字符串 → v2。
- 格式:pre-commit 对本目录是 no-op,需要的话手动 `ruff`/`yapf`。
- 不提交(除非另行要求)。

---

## 验收标准

1. subtiled dump 上 `streaming == {2,3}` 且出边延迟为 0、RRT 保留;bwd 上
   `streaming == {0,2}`。
2. 已存档反例形状(同 warp MMA lat=2 @ t−1 + blocking 消费者 @ t)被求解器与
   emit gate 双双拒绝;lat-0 op 同刻共存被接受。
3. 混合 lane 同组边进入 `cross_warp_dependencies` 且 sync 审计强制 channel;
   同组同 lane 边保持 program_order 合法。
4. v2 IR 的 `pipelined_program` 通过 schema 整体重算相等性检查;steady 恰
   |nodes| 实例。
5. 全测试绿(或仅剩 Phase 0b 显式 xfail);四 case 重跑 + 消融归档,全部走通
   handoff → scaffold → 负向审计。

**工作量预估**:Phase 0–2 约 2–3 个专注工作日(测试/fixture churn 是大头),
Phase 3 一夜后台,Phase 4–5 半天。唯一真正的**决策点是 A2**(推荐豁免 infra 边的
WS 语义,与 Fix A 同一原则,且不豁免会在 fwd 引入论文没有的 12 周期互斥)。
