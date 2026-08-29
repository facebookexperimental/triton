# curated-ddg-0.2 — 冻结 schema(单一权威文档)

> REV 2026-08-05 — 本文档为 W0 冻结产物,依 R4 裁定(2026-08-05,用户批准)制定。
> 驱动裁定:R4(token 解糖、per-node footprint、occupancy 保留、min_ii 删除)、
> R12(provenance 嵌套字段名 `curation_source.{ddg_sha256,baseline_graph_sha256,curator_sources_sha256}`)、
> R6(paper CLI 的 `--baseline-graph` 三配对规则)、R7(spill 零值语义归属)、
> R8(emitter 通路不消费 curated 工件)、R2(probe 只吃 curated G)。
> spec-2 / spec-4 / spec-5 / spec-6 **引用本文档,不得复述 schema**;凡引用处与本文冲突,以本文为准。
> 与各 spec 裁定前文本的差异清单见 §7,供各 spec 修订者对齐。
> REV2 2026-08-05:§1.5 文件名括注改为点后缀 `${stem}.curated_ddg.json` /
> `${stem}.curation_manifest.json`(裁定 K,spec-7 命名权威);无其他改动。

**状态**:FROZEN。任何字段增删、单位变更、语义变更都要求版本号提升
(`curated-ddg-0.3`)并重新过裁定;实现期内不得静默修改。

**依据**:raw 夹具 `sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json`
(schema `ddg-0.1`,已实测结构:node 字段 `{id, op_ref, op_kind, pipeline, latency,
self_latency, occupancy, min_warps, is_super_node}`,edge 字段
`{src, dst, kind, distance, latency}`,顶层 `{@generated, schema_version, config,
kernel, ops, loops}`)与现行 loader
`paper_joint_solver/ddg.py::load_problem`(2026-08-05 逐行核对)。

---

## §0 范围与角色

整编阶段(curation,spec-2 所有)把 **raw ddg.json + baseline schedule_graph.json**
变换为论文形状的 curated G,产出一对工件:

```
raw ddg.json ─┐
              ├─ [python -m paper_joint_solver.curate_ddg] ─→ curated_ddg.json
baseline      │                                            └→ curation_manifest.json
schedule_graph┘
```

- **curated_ddg.json** — 论文 G=(V,E) + 机器无关的逐节点/逐边标注。paper 规范路径
  (`solve_joint` / `solve_joint_audit` / `python -m paper_joint_solver.probe` /
  `strategy_report` / `schedule_plan` 重校验)的**唯一**图输入。
- **curation_manifest.json** — 每一条整编判断的审计记录。任何"被丢弃了什么/
  为什么"的下游观察(strategy_report 的 staging 观察等)从这里读,不回读 raw。

**R8 边界**:emitter 管线(`solve_joint_emitter`、`python -m
paper_joint_solver.emitter`、`run_emitter_cases.sh`、`emit_bench_kernels.py`、
`graph_writer.py`)**继续消费 raw ddg + baseline graph**,不消费本 schema 的任何
工件(graph_writer 的 timing-coverage 硬门会拒绝被整编删除的节点)。

---

## §1 顶层布局、版本与哈希链

### curated_ddg.json

```json
{
  "schema_version": "curated-ddg-0.2",
  "curation_source": {
    "ddg_sha256": "<raw ddg.json 的 sha256>",
    "baseline_graph_sha256": "<baseline schedule_graph.json 的 sha256>",
    "curator_sources_sha256": "<curate_ddg 及其私有库源码的合并 sha256>"
  },
  "loops": [
    {
      "loop_id": 0,
      "trip_count": 32,
      "ddg": { "nodes": [ ... ], "edges": [ ... ] }
    }
  ]
}
```

### curation_manifest.json

```json
{
  "schema_version": "curation-manifest-0.2",
  "curation_source": { 同上三元组,逐字节相同 },
  "loops": [ { "loop_id": 0, "nodes": [...], "edges": [...],
               "footprints": [...], "observations": {...} } ]
}
```

要点:

1. **版本串**:curated 文件 `"curated-ddg-0.2"`;manifest `"curation-manifest-0.2"`。
   loader 开头硬校验 curated 版本串,不匹配即报错并提示先跑 curate_ddg。
2. **哈希链字段名 = `curation_source`(嵌套对象)**,三个键名如上,repo 级唯一拼写
   (R12)。下游 solution/plan 工件的 provenance **逐字转录**该对象为
   `provenance.curation_source.{ddg_sha256, baseline_graph_sha256,
   curator_sources_sha256}`;spec-7 的验证块按嵌套名读取。
   (spec-2 0.1 草稿的顶层键名 `source` 作废,见 §7。)
3. **loops 数组**与 raw 的 loops 同序同索引(`loop_index` 寻址语义不变);
   `loop_id` 从 raw 透传(可选,溯源用);`trip_count` 从 raw 透传
   (int 或 null)。raw loop 的其余元数据(`min_ii`、`res_mii`、`rec_mii`、
   `is_outer`、`induction_var`、`lower_bound`、`upper_bound`、`step`、
   `trip_count_estimated`)**一律不携带**(R4;见 §4-D3)。
4. **确定性**:同输入必须字节级同输出。JSON `sort_keys=True` + 末尾单换行;
   nodes 按 `id` 升序;edges 按 `(src, dst, distance, latency)` 升序,平行重边
   之间保持 raw 文件相对次序(稳定排序)。manifest 同规则。
5. 文件名:schema 只规定内容;run 脚本的工件命名(`${stem}.curated_ddg.json` /
   `${stem}.curation_manifest.json`,lit1 词干)归 spec-7。

---

## §2 节点字段表(loops[i].ddg.nodes[])

所有字段**必填**(标注 opt 者除外);不存在任何缺省合成路径——一切缺省都在整编
阶段落定并记入 manifest。整编规则编号 C1–C9 指 spec-2 WI-1 的规则序列
(C5 经 R4 修订为"regs + footprint 归属解析",见 §7)。

| 字段 | 类型 | 单位 / 语义 | 产出整编规则 |
|---|---|---|---|
| `id` | int | raw 节点 id 透传;仅 C1 幸存者出现 | C1 + C9 |
| `op_ref` | str | raw op 引用(如 `"op_269446992"`);仅溯源与 paper 路径 IR 发射时对 baseline ops 表的映射(§6);solver 语义不得消费 | C9 |
| `op_kind` | str | TTGIR op 名(如 `"tt.descriptor_load"`);观察性字段(strategy_report 判据、报告);**solver 不得据此做任何再分类**——所有 op_kind 驱动的判断(TMEM 重指派、variable_latency、footprint 规则)已在整编时落定为下列显式字段 | C9 |
| `pipeline` | str | 功能单元,∈ `machine.capacities ∪ {"NONE"}`(现行 `{"TC","TMA","TMEM","SFU","CUDA","NONE"}`);**TMEM 重指派已应用**(raw 里 `tmem_load`/`tmem_store` 的 `CUDA` → `"TMEM"`) | C3 |
| `latency` | int ≥ 0 | raw cycles(论文 cycles(v) 的归一化前输入);rrt 每行 span 与之相等 | C9(校验) |
| `occupancy` | int ≥ 0, ≤ latency | raw cycles busy。**观察性字段**(R4 明令保留;strategy_report 的 `tma_isolated_rule` 消费 `occupancy == 0`);solver 的资源语义只走 `rrt`,不读此字段 | C9 |
| `rrt` | object | `{functional_unit: [int usage_at_cycle_0, ...]}`;**恒显式**——raw 有显式 rrt 则校验透传,否则由 occupancy 合成 `[1]*occ + [0]*(lat-occ)` 前载形状(仅当 pipeline ∈ capacities 且 latency > 0,否则 `{}`);manifest 记 `"explicit"` / `"synthesized_from_occupancy"` | C4 |
| `min_warps` | int ≥ 1 | 该 op 需要的最小 warp 数;raw 透传,缺省 1 在整编时落定。REGISTERLIMIT 的 `ceil(regs/min_warps)` 记账(R5)读此字段 | C9 |
| `regs` | int ≥ 0 | 32 位寄存器字(整 tile 结果值);由整编私有库 resource_model 的结果类型推导 | C5 |
| `spill_cost` | int ≥ 0 | raw cycles;**每个节点恒显式**,含 0(R7:零值也进成本池 C,池语义归 spec-1)。优先级:raw 显式注解 > regs>0 者取 machine 缺省 30 > memory-backed/无结果者 0;命中规则记 manifest | C7 |
| `smem_footprint` | int ≥ 0 | **字节**。别名已在整编解析:每个 SMEM 存储物的 footprint 只挂在唯一 owner 节点上,其余成员(视图、复用者)为 0;solver 的字面 per-value 求和因而物理正确(对 `machine.smem_bytes` 记账) | C5 |
| `tmem_footprint` | int ≥ 0 | **TMEM 列**(B200 原生分配单位;字节→列的 ceil 在整编完成,换算基数 `tmem_column_bytes=512` 记入 manifest `amount_basis`)。别名/accumulator 归属同上,owner 唯一;solver 对 `machine.tmem_cols` 记账 | C5 |
| `variable_latency` | bool | 论文 sec 4.3 的 variable-latency 类;整编外延 = TMA loads(spec-2 I4);solver/probe 只读 flag,不再推导 | C8 |
| `streaming` | bool | 论文 sec 5.3 字面规则 "no incoming data dependencies" 在 **curated E** 上求值(无任何豁免——infra 边已删、token 边已解糖为普通边);蕴含 `variable_latency` | C8 |

**不存在的节点字段**(出现即 loader 校验错误的候选;至少不得被任何消费者读):
`self_latency`、`is_super_node`、`explicit_spill_cost`、`warp_group`、
`spillcost`(别名拼写在整编吸收)、任何 `signal`/`infra`/`memory_objects` 相关字段。

阻塞相关不设节点字段:blocking 是逐边标注(§3);其来源规则(src.pipeline ∈
{TC, TMA} 或 token 完成)只在整编时使用。

---

## §3 边字段表(loops[i].ddg.edges[])与解糖契约

| 字段 | 类型 | 单位 / 语义 | 产出整编规则 |
|---|---|---|---|
| `src`, `dst` | int | curated 节点 id(必须存在于同 loop 的 nodes) | C1/C2 透传;C1.3 重连合成 |
| `distance` | int ≥ 0 | 迭代距离(论文 δ);raw 的 `kind`("data"/"loop_carried")丢弃——carried 语义完全由 `distance > 0` 表达 | C9 |
| `latency` | int ≥ 0 | raw cycles(论文 d);streaming 出边清零发生在 **solver**(论文 sec 5.3 成本模型规则),不在 curated 文件 | C9 |
| `blocking` | bool | 消费方需要阻塞同步;规则:`src.pipeline ∈ {"TC","TMA"}`(异步单元)或该边在整编时判定为 token 完成边;命中的规则名记 manifest | C6 |

**不存在的边字段**:`kind`、`src_result_idx`、`signal_only`。
**`Edge.signal_only` 全 repo 废除**(R4):loader 的 Edge dataclass、joint_smt /
schedule_plan / strategy_report / search 的一切 `signal_only` 读者按各自 spec 删除。

**平行重边**合法且逐条保留(与 raw dump 的平行边语义一致);边的稳定身份 =
数组位置(loader 赋 `index`,实现细节,不是文件字段)。

### token 边解糖契约(C2)

raw 中 "token 边" 的判定(`ops[src.op_ref].result_types[src_result_idx] ==
"!ttg.async.token"`)**只在整编阶段发生**,且是 raw 顶层 `ops` 表的唯一语义消费点。
输出中 token 边成为**普通依赖边**——同 `src/dst/distance/latency`,`blocking=true`
(规则 `async_token_completion`),与其他边完全同质。论文的 E 是同质 `(u,v,d,δ)`
元组;curated E 之后不存在边类别,下游一切 `data_edges` / `liveness_edges` /
`ws_data_edges` 派生集合的正确替换都是 **E 本身**。

### B5(TMEM liveness)承载方式

Fig 5 的 def-use liveness 在**全 curated E** 上量化。MMA 完成令牌延长 accumulator
活性的语义(B5)由解糖后的 use 边承载:footprint owner 节点(§2 C5)的出边
——含解糖 token 边——把它的活性区间延展到最后消费者。整编义务(fail-closed):

- footprint 归属解析必须核验:owner 在 Fig 5 liveness 下的活性区间覆盖该存储物
  **全体成员**的使用区间(三个 canonical fixture 已核实:token src 全为存储物
  成员、现行 liveness 边集 == 全边集,解糖即足够,无需附加边);
- 若某成员使用不被 owner 区间覆盖,整编**必须合成**一条显式 use 边
  (owner→该消费者,`distance`/`latency` 取被代表边的值),manifest 记
  `action:"synthesized", reason:"footprint_liveness_use"`。canonical fixtures
  预期不触发;触发即显形于 golden fixture diff(决议 3)。

---

## §4 从 raw 丢弃的内容与其 manifest 记录形状

每一类丢弃都有 manifest 通道;curated 文件本身不含任何"被丢弃物"的痕迹。

**D1 — emitter-infra 节点**(baseline `warp_group < 0`;C1)。节点连同全部关联边
删除;kept→infra*→kept 纯链做传递重连(d、δ 沿链求和)。记录形状:

```json
{"id": 0, "op_kind": "arith.muli", "action": "dropped",
 "reason": "emitter-infra (baseline warp_group<0)"}
{"src": 0, "dst": 1, "action": "removed", "reason": "src dropped (emitter-infra)"}
{"src": 34, "dst": 40, "action": "synthesized",
 "reason": "rewired_through_dropped_chain", "via": [35],
 "distance": 0, "latency": 52}
```

**D2 — token 边类别**(C2)。边本身保留(解糖),类别消失。记录形状:

```json
{"src": 27, "dst": 28, "action": "kept",
 "resolution": "token_to_plain_dependence",
 "blocking": true, "blocking_rule": "async_token_completion"}
```

**D3 — loop 元数据**:`min_ii`、`res_mii`、`rec_mii`、`is_outer`、
`induction_var`、`lower_bound`、`upper_bound`、`step`、`trip_count_estimated`。
solver 自行计算 MinII(`Problem.min_ii()`);`Problem.raw_min_ii` 死字段随之删除
(R12)。原值作为观察进 manifest:

```json
"observations": {"dropped_loop_fields":
  {"min_ii": 1204, "res_mii": 1204, "rec_mii": 1033, "is_outer": false, ...}}
```

**D4 — `memory_objects` / `StorageObject`**(solver 侧概念,R4 删除)。多成员
存储物记账被逐节点 `smem_footprint` / `tmem_footprint` 取代(§2);归属判断进
manifest `footprints[]`(§5)。`joint_smt` 的 object 记账、`Problem.memory_objects`
字段、`schedule_plan` 校验镜像的 object 逻辑按 spec-4 WI-9 / spec-6 删除。

**D5 — raw 顶层 `ops` 表、`config`、`kernel`、`@generated`**。paper 路径不携带;
需要时经 `curation_source.ddg_sha256` 回指 raw 文件。(`ops` 表在 paper 路径的
唯一残余消费点是 IR 发射,经 **baseline graph** 获得,见 §6 第 1 行。)

**D6 — 字段级丢弃**:节点 `self_latency`、`is_super_node`;边 `kind`、
`src_result_idx`。固定清单,不逐条记录,manifest 观察记一行:

```json
"observations": {"dropped_field_names":
  ["node.self_latency", "node.is_super_node", "edge.kind", "edge.src_result_idx"]}
```

---

## §5 curation_manifest.json — 逐判断记录 schema

每 loop 四个 section。所有记录都是**判断**(judgment)的转录:动作 + 命中规则名。

### nodes[]

- dropped 形状见 §4-D1。
- kept 形状(每个 curated 节点一条,全字段判断可审计):

```json
{"id": 17, "action": "kept",
 "pipeline": {"from": "CUDA", "to": "TMEM"}, "pipeline_rule": "tmem_port_unit",
 "rrt": "synthesized_from_occupancy",
 "spill_cost": {"value": 0, "rule": "memory_backed_zero"},
 "variable_latency": false, "streaming": false}
```

`rrt` ∈ {`"explicit"`, `"synthesized_from_occupancy"`};
`spill_cost.rule` ∈ {`"explicit"`, `"default_regs_producer"`, `"memory_backed_zero"`};
`pipeline_rule` 仅在发生重指派时出现(`"tmem_port_unit"`)。

### edges[]

`action` ∈ {`"kept"`, `"removed"`, `"synthesized"`};kept 边必带 `blocking` +
(若 true)`blocking_rule` ∈ {`"async_producer_unit_TC"`, `"async_producer_unit_TMA"`,
`"async_token_completion"`};解糖边带 `resolution: "token_to_plain_dependence"`;
合成边带 `via`(D1)或 `reason:"footprint_liveness_use"`(§3 B5)。

### footprints[](取代 0.1 草稿的 `memory_objects` section)

每个解析出的存储物一条**归属记录**——curated 文件里的逐节点 footprint 值由此
派生,manifest 保全多成员真相:

```json
{"object": "tmem:%acc", "kind": "tmem",
 "amount": 64,
 "amount_basis": {"bytes": 32768, "tmem_column_bytes": 512, "rule": "max_union"},
 "owner": 17, "members": [17, 27, 28],
 "rules": ["tmem_alloc", "mma_operand2_accumulator", "tmem_store_operand0"],
 "liveness_cover": "desugared_use_edges"}
```

- `owner` = footprint 挂载的唯一节点(该节点的 `{kind}_footprint` += `amount`;
  members 其余为 0)。`object` 是 resource_model 的存储物键(可与 owner 不同,
  如 `smem:{local_alloc id}` 挂载到产出数据的 descriptor_load);owner 的选取
  必须满足 §3 的 B5 覆盖义务。
- `amount_basis` 记录字节值与(TMEM)列换算基数;smem 物 `amount` 即字节,
  无换算基数。
- `rules` = resource_model 归属规则命中序列(`tmem_alloc` /
  `mma_operand2_accumulator` / `tmem_store_operand0` / `local_alloc` /
  `descriptor_load`)。
- `liveness_cover` ∈ {`"desugared_use_edges"`, `"synthesized"`}(§3 B5 核验结论)。
- 整编硬门(local_alloc/tmem_alloc 零 footprint、unclaimed memory handle)保留在
  整编阶段为 fail-closed 输入门,触发即整编错误(不产出工件)。

### observations{}

计数与固定清单:`dropped_node_kinds`(kind→count)、`token_edges_resolved`、
`edges_synthesized`、`dropped_loop_fields`(§4-D3)、`dropped_field_names`
(§4-D6)。

### 成对工件示例(节选,fwd_subtiled 形状)

curated_ddg.json:

```json
{"schema_version": "curated-ddg-0.2",
 "curation_source": {"ddg_sha256": "9f3a…", "baseline_graph_sha256": "77c1…",
                     "curator_sources_sha256": "b02e…"},
 "loops": [{"loop_id": 0, "trip_count": 32, "ddg": {
   "nodes": [
     {"id": 2, "op_ref": "op_269446992", "op_kind": "tt.descriptor_load",
      "pipeline": "TMA", "latency": 556, "occupancy": 96,
      "rrt": {"TMA": [1, 1, "…×96", 0, "…×460"]},
      "min_warps": 1, "regs": 0, "spill_cost": 0,
      "smem_footprint": 16384, "tmem_footprint": 0,
      "variable_latency": true, "streaming": true},
     {"id": 27, "op_ref": "op_269850112", "op_kind": "ttng.tmem_store",
      "pipeline": "TMEM", "latency": 4, "occupancy": 4,
      "rrt": {"TMEM": [1, 1, 1, 1]},
      "min_warps": 1, "regs": 0, "spill_cost": 0,
      "smem_footprint": 0, "tmem_footprint": 0,
      "variable_latency": false, "streaming": false},
     "…"],
   "edges": [
     {"src": 2, "dst": 10, "distance": 0, "latency": 556, "blocking": true},
     {"src": 27, "dst": 28, "distance": 0, "latency": 4, "blocking": true},
     "…"]}}]}
```

curation_manifest.json:

```json
{"schema_version": "curation-manifest-0.2",
 "curation_source": {"ddg_sha256": "9f3a…", "baseline_graph_sha256": "77c1…",
                     "curator_sources_sha256": "b02e…"},
 "loops": [{"loop_id": 0,
   "nodes": [
     {"id": 0, "op_kind": "arith.muli", "action": "dropped",
      "reason": "emitter-infra (baseline warp_group<0)"},
     {"id": 1, "op_kind": "arith.addi", "action": "dropped",
      "reason": "emitter-infra (baseline warp_group<0)"},
     {"id": 2, "action": "kept", "pipeline": {"from": "TMA", "to": "TMA"},
      "rrt": "synthesized_from_occupancy",
      "spill_cost": {"value": 0, "rule": "memory_backed_zero"},
      "variable_latency": true, "streaming": true},
     "…"],
   "edges": [
     {"src": 0, "dst": 1, "action": "removed",
      "reason": "src dropped (emitter-infra)"},
     {"src": 1, "dst": 2, "action": "removed",
      "reason": "src dropped (emitter-infra)"},
     {"src": 27, "dst": 28, "action": "kept",
      "resolution": "token_to_plain_dependence",
      "blocking": true, "blocking_rule": "async_token_completion"},
     {"src": 2, "dst": 10, "action": "kept",
      "blocking": true, "blocking_rule": "async_producer_unit_TMA"},
     "…"],
   "footprints": [
     {"object": "smem:4", "kind": "smem", "amount": 16384,
      "amount_basis": {"bytes": 16384, "rule": "max_union"},
      "owner": 2, "members": [2, 4], "rules": ["local_alloc", "descriptor_load"],
      "liveness_cover": "desugared_use_edges"},
     {"object": "tmem:%acc", "kind": "tmem", "amount": 64,
      "amount_basis": {"bytes": 32768, "tmem_column_bytes": 512,
                       "rule": "max_union"},
      "owner": 17, "members": [17, 27, 28],
      "rules": ["tmem_alloc", "tmem_store_operand0", "mma_operand2_accumulator"],
      "liveness_cover": "desugared_use_edges"}],
   "observations": {
     "dropped_node_kinds": {"arith.addi": 1, "arith.muli": 1},
     "token_edges_resolved": 3, "edges_synthesized": 0,
     "dropped_loop_fields": {"min_ii": 1204, "res_mii": 1204, "rec_mii": 1033},
     "dropped_field_names": ["node.self_latency", "node.is_super_node",
                             "edge.kind", "edge.src_result_idx"]}}]}
```

(示例哈希与部分数值为占位;golden fixture 的真实值在 W1 提交时定格——决议 3,
三个 fixture case 的 curated_ddg.json + curation_manifest.json 入库为回归锚。)

---

## §6 消费者与可读面(pinned)

| 消费者(裁定名) | 输入 | 可读 | 明确禁读 |
|---|---|---|---|
| `solve_joint`(paper 规范入口,spec-3/4)与 `python -m paper_joint_solver` CLI | curated_ddg.json | 全部语义字段:pipeline, latency, rrt, min_warps, regs, spill_cost, smem_footprint, tmem_footprint, variable_latency, streaming;边全字段;trip_count。`op_kind`/`op_ref`/`occupancy` 仅可转录进 stats/provenance,**不得驱动任何再分类/过滤/缺省** | manifest;raw ddg;raw `ops` 表。`--baseline-graph` 仅与 `--ir-out`/`--handoff-manifest-out` 三配对合法(R6),且只供 pipelined_ir 经 `op_ref` 映射 baseline ops 表构建 IR |
| `solve_joint_audit`(spec-3;FA4 refit 仪器) | 同上 + probe 输入(exact_warp_sets/colocate/separate) | 同 solve_joint | 同上 |
| `python -m paper_joint_solver.probe`(R2,UNSAT 消融证据) | **仅 curated_ddg.json**,无 `--baseline-graph` | 同 solve_joint | baseline graph;manifest;raw ddg |
| `strategy_report`(schema `paper-joint-strategy-v2`,含 `--curation-manifest`;spec-5) | curated_ddg.json + solution + curation_manifest.json(可选) | solver 面之外另可读:`op_kind`、`occupancy`(判据观察);manifest 的 `nodes[].action=="dropped"` 记录与 removed-edge 记录(bwd `descriptor_reduce` staging 观察);`footprints[]`(观察) | raw ddg;baseline graph |
| `schedule_plan.load_schedule_plan` / skc 重校验(spec-6) | curated_ddg.json + solution | 同 solve_joint(REGISTERLIMIT 镜像读 regs/min_warps,R5;MEMORYCAPACITY 镜像读逐节点 footprint);校验 solution `provenance.curation_source` 三元组与 curated 文件 `curation_source` 逐键相等 | 同 solve_joint |
| `viz` | curated_ddg.json(+ manifest 可选) | 同 strategy_report | raw ddg;baseline graph |
| **非消费者**(R8):`solve_joint_emitter`、`python -m paper_joint_solver.emitter`、`run_emitter_cases.sh`、`emit_bench_kernels.py`、`graph_writer.py` | raw ddg + baseline graph | —(不得打开 curated 工件) | curated_ddg.json;curation_manifest.json |

通则:
- **paper 路径任何组件不得回读 raw ddg.json**(哈希链校验除外——只算 sha,不解析)。
- solver/probe 不含任何过滤、分类、缺省逻辑;curated 字段即最终语义输入
  (spec-2 WI-6 的 G1–G5 契约以本 schema 为字段权威)。
- spill 池语义(零值入 C、无 `>0` 过滤、无生产者门控)归 spec-1;本 schema 只
  保证前提:每节点 `spill_cost` 显式存在、含 0(R7)。

---

## §7 与裁定前 spec 文本的差异(各 spec 修订者对齐清单)

R4 为准;下列旧文本必须按本文档改写,不得残留:

1. **spec-2 WI-1 的 0.1 草稿 schema**:
   - `"schema_version": "curated-ddg-0.1"` → `"curated-ddg-0.2"`;manifest 同步 0.2。
   - 顶层 `source` → **`curation_source`**(R12 嵌套名,repo 唯一拼写)。
   - loops 条目的 `"min_ii"` **删除**(草稿仍携带;R4 明令 drop,`raw_min_ii`
     一并删)。
   - loop 级 `memory_objects` 数组 **删除**;改为 §2 的逐节点
     `smem_footprint`/`tmem_footprint` + manifest `footprints[]` 归属记录
     (草稿的"多成员 StorageObject 进 curated 文件、solver 重建 StorageObject"
     路线作废;WI-2 第 9 条"memory_objects 改从 loop 读入 + StorageObject import
     保留"相应作废——`Problem.memory_objects` 与 `StorageObject` 从 solver 消失,
     spec-4 WI-9 的 `footprint(v, kind)` 读取器取而代之)。
   - 节点 `occupancy` **保留**为 curated 字段(草稿声明不输出;R4 改判:
     strategy_report 消费)。
   - C5 的产出改述为"regs + footprint 归属解析(owner 唯一化)+ B5 覆盖核验",
     manifest 的 `memory_objects` section 更名 `footprints`(记录形状 §5)。
   - manifest 增加 §4-D3/D6 的 observations 键。
2. **spec-2 C2 / 全 repo**:`Edge.signal_only` 废除已是草稿立场,维持;注意
   spec-2 WI-6 表中 search.py 行的 "all-ws" 域替换文本因 R1 作废(域语义硬编码
   reg-data,flag 删除)——归 spec-3 修订。
3. **spec-4 WI-9**:footprint 字段名定格为 `smem_footprint` / `tmem_footprint`
   (其草稿写 `"smem"`/`"tmem"`);单位定格为 smem=字节、**tmem=列**——与其自身
   tmem-column-granularity 处置(ceil 留在整编、capacity 单位 `machine.tmem_cols`
   不变)一致;R4 裁定摘要中的"字节"措辞按此解读(见 residual note)。其
   `liveness_edges = [e for e in prob.edges if not e.signal_only]` 过渡文本作废:
   `signal_only` 不存在,liveness 直接量化全 `prob.edges`。
4. **spec-5 I-9(b) 作废**:curated 边**没有** `signal_only` 注记。
   strategy_report 判据文本中 "non-signal data edges" 全部改为 "all curated
   edges";`test_classify_backward_tmem_source_walk_ignores_signal_edges` /
   `test_signal_only_edge_is_not_reduction_staging` 保留意图需改造为
   manifest/全边语义(归 spec-5 修订)。staging 观察从 manifest 读
   (本文档 §6 strategy_report 行)。I-9(c) 的字段假设改为 §2 表
   (`op_kind`/`pipeline`/`occupancy`/`regs` 均在,另有 §2 全表)。
5. **spec-6**:重校验镜像的输入面按 §6 `load_schedule_plan` 行(warp_sets 工件
   形状归 R6/spec-6,本 schema 不涉解工件)。

---

## §8 残留注记

- R4 裁定摘要把逐节点 footprint 概括为"smem/tmem 字节";本 schema 依 spec-4
  WI-9 的容量单位契约把 `tmem_footprint` 定格为**列**(字节值保全在 manifest
  `amount_basis.bytes`)。如 lead 坚持双字段皆字节,则 ceil 移回 solver,与
  spec-4 的 no-op-refuted 处置冲突——默认按本文执行。
- golden fixture(决议 3)提交时,本 schema §5 示例中的占位数值不回填;golden
  文件本身即数值权威。
