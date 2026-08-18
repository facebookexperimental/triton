// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule=mode=modulo 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SCHEDULE
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule=mode=modulo 2>/dev/null \
// RUN:   | triton-opt -split-input-file -tritonamdgpu-pipeline 2>&1 \
// RUN:   | FileCheck %s
// RUN: triton-opt %s -tritonamdgpu-dump-plan-value-graph='output-path=- strict=true' \
// RUN:   -o /dev/null | FileCheck %s --check-prefix=PLAN
// RUN: triton-opt %s -tritonamdgpu-dump-plan-value-graph='output-path=- strict=true' \
// RUN:   -o /dev/null > %t.plan.original
// RUN: sed 's/%a_ptrs/%renamed_ptrs/g; s/%a_ld/%renamed_load/g' %s \
// RUN:   | triton-opt -tritonamdgpu-dump-plan-value-graph='output-path=- strict=true' \
// RUN:     -o /dev/null > %t.plan.renamed
// RUN: diff %t.plan.original %t.plan.renamed
// RUN: triton-opt %s -o %t.no-analysis.mlir
// RUN: triton-opt %s \
// RUN:   -tritonamdgpu-dump-plan-value-graph='output-path=%t.plan.json strict=true' \
// RUN:   -o %t.with-analysis.mlir
// RUN: diff %t.no-analysis.mlir %t.with-analysis.mlir
// RUN: triton-opt %S/../../Conversion/amd/in_thread_transpose.mlir \
// RUN:   -split-input-file \
// RUN:   -tritonamdgpu-dump-plan-value-graph='output-path=- strict=true' \
// RUN:   -o /dev/null | FileCheck %s --check-prefix=PLAN-TRANSPOSE
// RUN: triton-opt %s \
// RUN:   -tritonamdgpu-dump-plan-value-graph='output-path=%t.schedule.plan.json strict=true' \
// RUN:   -o /dev/null
// RUN: env PYTHONPATH=%S/../../../third_party/tlx/tools/plan_ir \
// RUN:   python3 -m tlx_plan.cli schedule-delta \
// RUN:   --value-graph %t.schedule.plan.json --kernel schedule_apply_fixture \
// RUN:   --output %t.schedule.identity.json
// RUN: python3 -c "import json; p=json.load(open('%t.schedule.identity.json')); o=p['blocks'][0]['desired_order']; o[0],o[1]=o[1],o[0]; open('%t.schedule.legal.json','w').write(json.dumps(p))"
// RUN: triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-schedule='input-path=%t.schedule.legal.json report-path=%t.schedule.report.json strict=true' \
// RUN:   | FileCheck %s --check-prefix=APPLY
// RUN: FileCheck %s --check-prefix=APPLY-REPORT < %t.schedule.report.json
// RUN: triton-opt %S/../../Conversion/amd/in_thread_transpose.mlir \
// RUN:   -split-input-file \
// RUN:   -tritonamdgpu-apply-plan-schedule='input-path=%t.schedule.legal.json strict=true allow-missing-kernel=true' \
// RUN:   -o /dev/null
// RUN: python3 -c "import json; p=json.load(open('%t.schedule.identity.json')); o=p['blocks'][0]['desired_order']; o[1],o[2]=o[2],o[1]; open('%t.schedule.invalid.json','w').write(json.dumps(p))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-schedule='input-path=%t.schedule.invalid.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=APPLY-REJECT
// RUN: triton-opt %s \
// RUN:   -tritonamdgpu-dump-plan-value-graph='output-path=%t.pipeline.plan.json strict=true pass-position=before_update_async_wait_count' \
// RUN:   -o /dev/null
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.plan.json')); f=next(x for x in p['functions'] if x['function']=='async_lds_modulo_slots'); l=next(x['id'] for x in f['operations'] if x['kind']=='scf.for'); g=f['async_groups'][0]['id']; d={'schema_version':'plan-pipeline-delta/0.1','kernel':f['function'],'input_value_graph_fingerprint':f['semantic_fingerprint'],'pass_position':'before_update_async_wait_count','loops':[{'loop':l,'transactions':[{'group':g,'action':'set_prefetch_distance','distance':1,'buffer_depth':2}],'staging':[]}]}; open('%t.pipeline.legal.json','w').write(json.dumps(d))"
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.legal.json report-path=%t.pipeline.report.json strict=true' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --check-prefix=PIPELINE-REPORT < %t.pipeline.report.json
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.legal.json')); p['loops'][0]['transactions'][0]['buffer_depth']=3; open('%t.pipeline.depth3.json','w').write(json.dumps(p))"
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.depth3.json report-path=%t.pipeline.depth3.report.json strict=true' \
// RUN:   | FileCheck %s --check-prefix=PIPELINE-DEPTH3
// RUN: FileCheck %s --check-prefix=PIPELINE-DEPTH3-REPORT < %t.pipeline.depth3.report.json
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.legal.json')); p['loops'][0]['transactions'][0].update(distance=1,buffer_depth=1); open('%t.pipeline.depth1.json','w').write(json.dumps(p))"
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.depth1.json report-path=%t.pipeline.depth1.report.json strict=true' \
// RUN:   | FileCheck %s --check-prefix=PIPELINE-DEPTH1
// RUN: FileCheck %s --check-prefix=PIPELINE-DEPTH1-REPORT < %t.pipeline.depth1.report.json
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.depth3.json')); p['loops'][0]['transactions'][0]['distance']=2; open('%t.pipeline.distance2.json','w').write(json.dumps(p))"
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.distance2.json report-path=%t.pipeline.distance2.report.json strict=true' \
// RUN:   | FileCheck %s --check-prefix=PIPELINE-DISTANCE2
// RUN: FileCheck %s --check-prefix=PIPELINE-DISTANCE2-REPORT < %t.pipeline.distance2.report.json
// RUN: python3 -c "p=open('%s').read(); a=p.rindex('  tt.func @async_lds_modulo_slots'); b=p.index('\n  tt.func ',a+1); s=p[a:b].replace('      ttg.barrier all\n','',2); open('%t.pipeline.no-barriers.mlir','w').write(p[:a]+s+p[b:])"
// RUN: triton-opt %t.pipeline.no-barriers.mlir \
// RUN:   -tritonamdgpu-dump-plan-value-graph='output-path=%t.pipeline.no-barriers.plan.json strict=true pass-position=before_update_async_wait_count' \
// RUN:   -o /dev/null
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.no-barriers.plan.json')); f=next(x for x in p['functions'] if x['function']=='async_lds_modulo_slots'); l=next(x['id'] for x in f['operations'] if x['kind']=='scf.for'); g=f['async_groups'][0]['id']; d={'schema_version':'plan-pipeline-delta/0.1','kernel':f['function'],'input_value_graph_fingerprint':f['semantic_fingerprint'],'pass_position':'before_update_async_wait_count','loops':[{'loop':l,'transactions':[{'group':g,'action':'set_prefetch_distance','distance':1,'buffer_depth':3}],'staging':[]}]}; open('%t.pipeline.no-barriers.json','w').write(json.dumps(d))"
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %t.pipeline.no-barriers.mlir \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.no-barriers.json report-path=%t.pipeline.no-barriers.report.json strict=true' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --check-prefix=PIPELINE-NO-BARRIERS-REPORT < %t.pipeline.no-barriers.report.json
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.legal.json')); p['loops'][0]['transactions'][0]['buffer_depth']=400; open('%t.pipeline.invalid.json','w').write(json.dumps(p))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.invalid.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PIPELINE-REJECT
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.plan.json')); f=next(x for x in p['functions'] if x['function']=='register_to_lds_staging'); ops={x['id']:x['kind'] for x in f['operations']}; v=next(x for x in f['values'] if ops.get(x['origin'].get('operation'))=='arith.addf' and {ops[u['operation']] for u in x['uses']}=={'arith.mulf','arith.subf'}); l=next(x['id'] for x in f['operations'] if x['kind']=='scf.for'); cs=sorted({u['operation'] for u in v['uses']}); d={'schema_version':'plan-pipeline-delta/0.1','kernel':f['function'],'input_value_graph_fingerprint':f['semantic_fingerprint'],'pass_position':'before_update_async_wait_count','loops':[{'loop':l,'transactions':[],'staging':[{'value':v['id'],'action':'register_to_lds','consumers':cs,'buffer_depth':1,'alignment':16}]}]}; open('%t.pipeline.staging.json','w').write(json.dumps(d))"
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.json report-path=%t.pipeline.staging.report.json strict=true' \
// RUN:   | FileCheck %s --check-prefix=PIPELINE-STAGING
// RUN: FileCheck %s --check-prefix=PIPELINE-STAGING-REPORT < %t.pipeline.staging.report.json
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.staging.json')); p['loops'][0]['staging'][0]['consumers'].pop(); open('%t.pipeline.staging.incomplete.json','w').write(json.dumps(p))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.incomplete.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PIPELINE-STAGING-INCOMPLETE
// RUN: python3 -c "import json; d=json.load(open('%t.pipeline.staging.json')); p=json.load(open('%t.pipeline.plan.json')); f=next(x for x in p['functions'] if x['function']=='register_to_lds_staging'); value=next(x for x in f['values'] if x['id']==d['loops'][0]['staging'][0]['value']); producer=value['origin']['operation']; selected=set(d['loops'][0]['staging'][0]['consumers']); extra=next(x['id'] for x in f['operations'] if x['kind']=='arith.addf' and x['id'] != producer and x['id'] not in selected); d['loops'][0]['staging'][0]['consumers'].append(extra); open('%t.pipeline.staging.nonuse.json','w').write(json.dumps(d))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.nonuse.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PIPELINE-STAGING-NONUSE
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.staging.json')); p['loops'][0]['staging'][0]['buffer_depth']=2; open('%t.pipeline.staging.depth.json','w').write(json.dumps(p))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.depth.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PIPELINE-STAGING-DEPTH
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.staging.json')); p['loops'][0]['staging'][0]['alignment']=12; open('%t.pipeline.staging.alignment.json','w').write(json.dumps(p))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.alignment.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PIPELINE-STAGING-ALIGNMENT
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.plan.json')); f=next(x for x in p['functions'] if x['function']=='register_to_lds_capacity'); ops={x['id']:x['kind'] for x in f['operations']}; v=next(x for x in f['values'] if ops.get(x['origin'].get('operation'))=='arith.addf'); l=next(x['id'] for x in f['operations'] if x['kind']=='scf.for'); cs=sorted({u['operation'] for u in v['uses']}); d={'schema_version':'plan-pipeline-delta/0.1','kernel':f['function'],'input_value_graph_fingerprint':f['semantic_fingerprint'],'pass_position':'before_update_async_wait_count','loops':[{'loop':l,'transactions':[],'staging':[{'value':v['id'],'action':'register_to_lds','consumers':cs,'buffer_depth':1,'alignment':16}]}]}; open('%t.pipeline.staging.capacity.json','w').write(json.dumps(d))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.capacity.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PIPELINE-STAGING-CAPACITY
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.plan.json')); f=next(x for x in p['functions'] if x['function']=='register_to_lds_scalar'); ops={x['id']:x['kind'] for x in f['operations']}; v=next(x for x in f['values'] if ops.get(x['origin'].get('operation'))=='arith.index_cast'); l=next(x['id'] for x in f['operations'] if x['kind']=='scf.for'); cs=sorted({u['operation'] for u in v['uses']}); d={'schema_version':'plan-pipeline-delta/0.1','kernel':f['function'],'input_value_graph_fingerprint':f['semantic_fingerprint'],'pass_position':'before_update_async_wait_count','loops':[{'loop':l,'transactions':[],'staging':[{'value':v['id'],'action':'register_to_lds','consumers':cs,'buffer_depth':1,'alignment':4}]}]}; open('%t.pipeline.staging.scalar.json','w').write(json.dumps(d))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.scalar.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PIPELINE-STAGING-SCALAR
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.plan.json')); f=next(x for x in p['functions'] if x['function']=='register_to_lds_derived'); ops={x['id']:x['kind'] for x in f['operations']}; v=next(x for x in f['values'] if ops.get(x['origin'].get('operation'))=='arith.addf' and any(ops[u['operation']]=='amdg.extract_slice' for u in x['uses'])); s=next(u['operation'] for u in v['uses'] if ops[u['operation']]=='amdg.extract_slice'); sv=next(x for x in f['values'] if x['origin'].get('operation')==s); cs=sorted({u['operation'] for u in sv['uses']}); l=next(x['id'] for x in f['operations'] if x['kind']=='scf.for'); d={'schema_version':'plan-pipeline-delta/0.1','kernel':f['function'],'input_value_graph_fingerprint':f['semantic_fingerprint'],'pass_position':'before_update_async_wait_count','loops':[{'loop':l,'transactions':[],'staging':[{'value':v['id'],'action':'register_to_lds','consumers':cs,'buffer_depth':1,'alignment':16}]}]}; open('%t.pipeline.staging.derived.json','w').write(json.dumps(d))"
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.derived.json report-path=%t.pipeline.staging.derived.report.json strict=true' \
// RUN:   | FileCheck %s --check-prefix=PIPELINE-STAGING-DERIVED
// RUN: FileCheck %s --check-prefix=PIPELINE-STAGING-DERIVED-REPORT < %t.pipeline.staging.derived.report.json
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.plan.json')); f=next(x for x in p['functions'] if x['function']=='register_to_lds_unsupported'); ops={x['id']:x['kind'] for x in f['operations']}; v=next(x for x in f['values'] if ops.get(x['origin'].get('operation'))=='arith.addf' and any(ops[u['operation']]=='arith.mulf' for u in x['uses'])); c=next(x['id'] for x in f['operations'] if x['kind']=='arith.subf'); l=next(x['id'] for x in f['operations'] if x['kind']=='scf.for'); d={'schema_version':'plan-pipeline-delta/0.1','kernel':f['function'],'input_value_graph_fingerprint':f['semantic_fingerprint'],'pass_position':'before_update_async_wait_count','loops':[{'loop':l,'transactions':[],'staging':[{'value':v['id'],'action':'register_to_lds','consumers':[c],'buffer_depth':1,'alignment':16}]}]}; open('%t.pipeline.staging.unsupported.json','w').write(json.dumps(d))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.unsupported.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PIPELINE-STAGING-DERIVATION-REJECT
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.plan.json')); f=next(x for x in p['functions'] if x['function']=='register_to_lds_no_gain'); ops={x['id']:x['kind'] for x in f['operations']}; v=next(x for x in f['values'] if ops.get(x['origin'].get('operation'))=='arith.addf' and any(ops[u['operation']]=='amdg.extract_slice' for u in x['uses'])); s=next(u['operation'] for u in v['uses'] if ops[u['operation']]=='amdg.extract_slice'); sv=next(x for x in f['values'] if x['origin'].get('operation')==s); cs=sorted({u['operation'] for u in sv['uses']}); l=next(x['id'] for x in f['operations'] if x['kind']=='scf.for'); d={'schema_version':'plan-pipeline-delta/0.1','kernel':f['function'],'input_value_graph_fingerprint':f['semantic_fingerprint'],'pass_position':'before_update_async_wait_count','loops':[{'loop':l,'transactions':[],'staging':[{'value':v['id'],'action':'register_to_lds','consumers':cs,'buffer_depth':1,'alignment':16}]}]}; open('%t.pipeline.staging.no-gain.json','w').write(json.dumps(d))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.no-gain.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PIPELINE-STAGING-NO-GAIN
// RUN: python3 -c "import json; p=json.load(open('%t.pipeline.plan.json')); f=next(x for x in p['functions'] if x['function']=='register_to_lds_layout_chain'); ops={x['id']:x['kind'] for x in f['operations']}; v=next(x for x in f['values'] if ops.get(x['origin'].get('operation'))=='arith.addi' and any(ops[u['operation']]=='tt.reshape' for u in x['uses'])); req=next(x for x in f['values'] if ops.get(x['origin'].get('operation'))=='tlx.require_layout'); cs=sorted({u['operation'] for u in req['uses']}); l=next(x['id'] for x in f['operations'] if x['kind']=='scf.for'); d={'schema_version':'plan-pipeline-delta/0.1','kernel':f['function'],'input_value_graph_fingerprint':f['semantic_fingerprint'],'pass_position':'before_update_async_wait_count','loops':[{'loop':l,'transactions':[],'staging':[{'value':v['id'],'action':'register_to_lds','consumers':cs,'buffer_depth':1,'alignment':16}]}]}; open('%t.pipeline.staging.layout-chain.json','w').write(json.dumps(d))"
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-pipeline='input-path=%t.pipeline.staging.layout-chain.json report-path=%t.pipeline.staging.layout-chain.report.json strict=true' \
// RUN:   | FileCheck %s --check-prefix=PIPELINE-STAGING-LAYOUT-CHAIN
// RUN: FileCheck %s --check-prefix=PIPELINE-STAGING-LAYOUT-CHAIN-REPORT < %t.pipeline.staging.layout-chain.report.json
//
// Modulo runs before the guarded legacy scheduler. A successful modulo schedule
// is preserved; the standard AMD pipeline lowers and expands it.

// SCHEDULE: remark: amd-modulo:{{.*}}II={{[0-9]+}} maxStage=1{{.*}}serialized num_stages=2
// SCHEDULE-NOT: triton.warp_pipeline.border
// CHECK-LABEL: tt.func @early_lower
// The standard pipeline may choose register pipelining when this single-load
// fixture is not profitable/legal for LDS async copy. Verify the load is peeled
// into the prologue and forwarded as a loop-carried tensor to the stage-1 dot.
// CHECK:       tt.load {{.*}}amd.pipeliner_part = "prologue"
// CHECK:       scf.for {{.*}}iter_args({{.*}}tensor<256x64xf16, #blocked>)
// CHECK:         tt.load
// CHECK:         ttg.convert_layout {{.*}}tensor<256x64xf16, #blocked>
// CHECK:         tt.dot

// PLAN-DAG: "schema_version": "plan-value-graph/0.4"
// PLAN-DAG: "kind": "loop_init"
// PLAN-DAG: "iteration_distance": 1
// PLAN-DAG: "kind": "loop_backedge"
// PLAN-DAG: "kind": "loop_exit"
// PLAN-DAG: "kind": "loop_forward"
// PLAN-DAG: "kind": "branch_yield"
// PLAN-DAG: "kind": "convert_layout"
// PLAN-DAG: "logical_bytes": 262144
// PLAN-DAG: "physical_register_bytes": null
// PLAN-DAG: "artifact_stage": "final_structured_ttgir"
// PLAN-DAG: "static_intervals_are_physical_cycles": false
// PLAN-DAG: "lds_logical_bytes_are_physical_allocation": false
// PLAN-DAG: "async_lifetime_extended_through_wait": true
// PLAN-DAG: "async_lifetimes_are_physical_cycles": false
// PLAN-DAG: "live_segments": [
// PLAN-DAG: "view_kind": "ttg.memdesc_index"
// PLAN-DAG: "kind": "modulo"
// PLAN-DAG: "modulus": 2
// PLAN-DAG: "possible_slots": [
// PLAN-DAG: "effect": "read"
// PLAN-DAG: "effect": "write"
// PLAN-DAG: "effect": "allocate"
// PLAN-DAG: "effect": "free"
// PLAN-DAG: "physical_lds_offset": null
// PLAN-DAG: "direction": "lds_write"
// PLAN-DAG: "retained_group_count": 1
// PLAN-DAG: "kind": "completion_wait"
// PLAN-DAG: "kind": "visibility_barrier"
// PLAN-DAG: "kind": "lds_consumer"
// PLAN-DAG: "kind": "reuse_release_barrier"
// PLAN-DAG: "kind": "slot_overwrite"
// PLAN-DAG: "iteration_distance": 2
// PLAN-DAG: "precision": "conservative_cross_region"
// PLAN-DAG: "code": "async_write_without_completion"
// PLAN-DAG: "kind": "loop_carried_ssa"
// PLAN-DAG: "kind": "memory_raw"
// PLAN-DAG: "kind": "memory_war"
// PLAN-DAG: "kind": "memory_waw"
// PLAN-DAG: "kind": "async_completion"
// PLAN-DAG: "kind": "barrier_visibility"
// PLAN-DAG: "kind": "consumer_release"
// PLAN-DAG: "kind": "slot_reuse"
// PLAN-DAG: "peak_live_sets": [
// PLAN-DAG: "logical_tensor_bytes_are_per_wave_vgpr_bytes": false
// PLAN-DAG: "physical_vgpr_peak": null
// PLAN-DAG: "physical_lds_bytes": null
// PLAN-DAG: "max_logical_slot_depth": 2
// PLAN-DAG: "importance": "important"
// PLAN-DAG: "status": "open"
// PLAN-TRANSPOSE: "kind": "in_thread_transpose"
// APPLY-LABEL: tt.func @schedule_apply_fixture
// APPLY: %[[C2:.*]] = arith.constant 2 : i32
// APPLY-NEXT: %[[C1:.*]] = arith.constant 1 : i32
// APPLY-NEXT: arith.addi %[[C1]], %[[C2]]
// APPLY-REPORT: "accepted": true
// APPLY-REPORT: "moved_operations": 2
// APPLY-REJECT: schedule delta reverses distance-zero dependency
// PIPELINE-REPORT: "accepted": true
// PIPELINE-REPORT: "changes_buffer_depth": false
// PIPELINE-REPORT: "changes_iteration_storage": false
// PIPELINE-REPORT: "changes_prefetch_distance": false
// PIPELINE-REPORT: "changes_synchronization": false
// PIPELINE-REPORT: "moved_operations": {{[1-9][0-9]*}}
// PIPELINE-REPORT: "output_value_graph_fingerprint": "{{[0-9a-f]+}}"
// PIPELINE-REPORT: "skipped_inconsistent_dependencies": 0
// PIPELINE-DEPTH3-LABEL: tt.func @async_lds_modulo_slots
// PIPELINE-DEPTH3: ttg.local_alloc : () -> !ttg.memdesc<3x16x16xf16
// PIPELINE-DEPTH3: arith.constant 3 : i32
// PIPELINE-DEPTH3: ttg.memdesc_index
// PIPELINE-DEPTH3-REPORT: "accepted": true
// PIPELINE-DEPTH3-REPORT: "changes_buffer_depth": true
// PIPELINE-DEPTH3-REPORT: "changes_iteration_storage": true
// PIPELINE-DEPTH3-REPORT: "changes_synchronization": true
// PIPELINE-DEPTH3-REPORT: "post_rewrite_ddg_verified": true
// PIPELINE-DEPTH3-REPORT: "rewritten_slot_indices": 2
// PIPELINE-DEPTH3-REPORT: "ring_mutations": 1
// PIPELINE-DEPTH3-REPORT: "post_rewrite_audit_passed": true
// PIPELINE-DEPTH1-LABEL: tt.func @async_lds_modulo_slots
// PIPELINE-DEPTH1: ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16
// PIPELINE-DEPTH1: ttg.async_wait {num = 0 : i32}
// PIPELINE-DEPTH1: ttg.barrier
// PIPELINE-DEPTH1: ttg.local_load
// PIPELINE-DEPTH1: ttg.barrier
// PIPELINE-DEPTH1: ttg.async_copy_global_to_local
// PIPELINE-DEPTH1: ttg.async_commit_group
// PIPELINE-DEPTH1-REPORT: "accepted": true
// PIPELINE-DEPTH1-REPORT: "logical_lds_bytes_after": 512
// PIPELINE-DEPTH1-REPORT: "post_rewrite_audit_passed": true
// PIPELINE-DISTANCE2-LABEL: tt.func @async_lds_modulo_slots
// PIPELINE-DISTANCE2: arith.constant 1 : i32
// PIPELINE-DISTANCE2: arith.addi
// PIPELINE-DISTANCE2: arith.constant 3 : i32
// PIPELINE-DISTANCE2: arith.remui
// PIPELINE-DISTANCE2: ttg.async_wait {num = 2 : i32}
// PIPELINE-DISTANCE2-REPORT: "accepted": true
// PIPELINE-DISTANCE2-REPORT: "changes_prefetch_distance": true
// PIPELINE-DISTANCE2-REPORT: "updated_waits": 1
// PIPELINE-NO-BARRIERS-REPORT: "accepted": true
// PIPELINE-NO-BARRIERS-REPORT: "inserted_barriers": 2
// PIPELINE-NO-BARRIERS-REPORT: "post_rewrite_audit_passed": true
// PIPELINE-REJECT: requested LDS ring depths exceed the target LDS capacity
// PIPELINE-STAGING-LABEL: tt.func @register_to_lds_staging
// PIPELINE-STAGING: ttg.local_alloc {{.*}}alignment = 16
// PIPELINE-STAGING: scf.for
// PIPELINE-STAGING: arith.addf
// PIPELINE-STAGING-NEXT: ttg.local_store
// PIPELINE-STAGING-NEXT: ttg.barrier
// PIPELINE-STAGING: ttg.local_load
// PIPELINE-STAGING: arith.mulf
// PIPELINE-STAGING: arith.subf
// PIPELINE-STAGING-NEXT: ttg.barrier
// PIPELINE-STAGING: ttg.local_dealloc
// PIPELINE-STAGING-REPORT-DAG: "accepted": true
// PIPELINE-STAGING-REPORT-DAG: "changes_iteration_storage": true
// PIPELINE-STAGING-REPORT-DAG: "changes_synchronization": true
// PIPELINE-STAGING-REPORT-DAG: "changes_new_staging": true
// PIPELINE-STAGING-REPORT-DAG: "staging_mutations": 1
// PIPELINE-STAGING-REPORT-DAG: "inserted_barriers": 2
// PIPELINE-STAGING-REPORT-DAG: "post_rewrite_ddg_verified": true
// PIPELINE-STAGING-REPORT-DAG: "post_rewrite_audit_passed": true
// PIPELINE-STAGING-REPORT-DAG: "materialization_scope": "register_to_lds_staging"
// PIPELINE-STAGING-INCOMPLETE: staging_does_not_shorten_lifetime
// PIPELINE-STAGING-NONUSE: named register-to-LDS consumer does not use the staged value
// PIPELINE-STAGING-DEPTH: M1.5b.4 supports only single-slot register staging
// PIPELINE-STAGING-ALIGNMENT: pipeline staging alignment must be a power of two
// PIPELINE-STAGING-CAPACITY: register-to-LDS staging exceeds the target LDS capacity
// PIPELINE-STAGING-SCALAR: register-to-LDS staging requires a produced ranked tensor
// PIPELINE-STAGING-DERIVED-LABEL: tt.func @register_to_lds_derived
// PIPELINE-STAGING-DERIVED: %[[PRODUCER:.*]] = arith.addf
// PIPELINE-STAGING-DERIVED-NEXT: ttg.local_store %[[PRODUCER]]
// PIPELINE-STAGING-DERIVED: ttg.local_load
// PIPELINE-STAGING-DERIVED-COUNT-1: amdg.extract_slice
// PIPELINE-STAGING-DERIVED: arith.mulf
// PIPELINE-STAGING-DERIVED: arith.subf
// PIPELINE-STAGING-DERIVED-REPORT-DAG: "derived_operations_cloned": 1
// PIPELINE-STAGING-DERIVED-REPORT-DAG: "derived_operations_pruned": 1
// PIPELINE-STAGING-DERIVED-REPORT-DAG: "logical_live_range_shortened": true
// PIPELINE-STAGING-DERIVED-REPORT-DAG: "selected_consumer_operands": 4
// PIPELINE-STAGING-DERIVED-REPORT-DAG: "unselected_consumers_preserved": 1
// PIPELINE-STAGING-DERIVED-REPORT-DAG: "post_rewrite_audit_passed": true
// PIPELINE-STAGING-DERIVATION-REJECT: named register-to-LDS consumer is reached through an unsupported derived operation
// PIPELINE-STAGING-NO-GAIN: staging_does_not_shorten_lifetime
// PIPELINE-STAGING-LAYOUT-CHAIN-LABEL: tt.func @register_to_lds_layout_chain
// PIPELINE-STAGING-LAYOUT-CHAIN: ttg.local_load
// PIPELINE-STAGING-LAYOUT-CHAIN: tt.reshape
// PIPELINE-STAGING-LAYOUT-CHAIN: tt.trans
// PIPELINE-STAGING-LAYOUT-CHAIN: ttg.convert_layout
// PIPELINE-STAGING-LAYOUT-CHAIN: tlx.require_layout
// PIPELINE-STAGING-LAYOUT-CHAIN-REPORT-DAG: "derived_operations_cloned": 4
// PIPELINE-STAGING-LAYOUT-CHAIN-REPORT-DAG: "derived_operations_pruned": 4
// PIPELINE-STAGING-LAYOUT-CHAIN-REPORT-DAG: "logical_live_range_shortened": true
// PIPELINE-STAGING-LAYOUT-CHAIN-REPORT-DAG: "selected_consumer_operands": 2

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#slot_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#reshape_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [0, 1]}>
#reshape_dst = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#reshape_trans = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#slot_shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @early_lower(
      %a_ptrs: tensor<256x64x!tt.ptr<f16>, #blocked>,
      %b: tensor<64x256xf16, #dot1>,
      %c_init: tensor<256x256xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<256x256xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c_init)
        -> (tensor<256x256xf32, #mma>) {
      %a_ld = tt.load %a_ptrs : tensor<256x64x!tt.ptr<f16>, #blocked>
      %a = ttg.convert_layout %a_ld : tensor<256x64xf16, #blocked>
              -> tensor<256x64xf16, #dot0>
      %d = tt.dot %a, %b, %acc :
          tensor<256x64xf16, #dot0> * tensor<64x256xf16, #dot1>
          -> tensor<256x256xf32, #mma>
      scf.yield %d : tensor<256x256xf32, #mma>
    }
    tt.return %res : tensor<256x256xf32, #mma>
  }

  tt.func @structured_control(%cond: i1, %initial: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %selected = scf.if %cond -> i32 {
      %then = arith.addi %initial, %c1 : i32
      scf.yield %then : i32
    } else {
      %else = arith.subi %initial, %c1 : i32
      scf.yield %else : i32
    }
    %result = scf.while (%iter = %selected) : (i32) -> i32 {
      %keep_going = arith.cmpi slt, %iter, %initial : i32
      scf.condition(%keep_going) %iter : i32
    } do {
    ^bb0(%iter: i32):
      %next = arith.addi %iter, %c1 : i32
      scf.yield %next : i32
    }
    tt.return %result : i32
  }

  tt.func @lds_modulo_slots(%data: tensor<16x16xf16, #slot_blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable>
    scf.for %i = %c0 to %c4 step %c1 {
      %i_i32 = arith.index_cast %i : index to i32
      %current_index = arith.remsi %i_i32, %c2_i32 : i32
      %previous_index = arith.subi %c1_i32, %current_index : i32
      %current = ttg.memdesc_index %alloc[%current_index] : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      %previous = ttg.memdesc_index %alloc[%previous_index] : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      ttg.local_store %data, %current : tensor<16x16xf16, #slot_blocked> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      %loaded = ttg.local_load %previous : !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable> -> tensor<16x16xf16, #slot_blocked>
    }
    ttg.local_dealloc %alloc : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable>
    tt.return
  }

  tt.func @async_lds_modulo_slots(
      %ptrs: tensor<16x16x!tt.ptr<f16>, #slot_blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable>
    scf.for %i = %c0 to %c4 step %c1 {
      %i_i32 = arith.index_cast %i : index to i32
      %current_index = arith.remsi %i_i32, %c2_i32 : i32
      %previous_index = arith.subi %c1_i32, %current_index : i32
      %current = ttg.memdesc_index %alloc[%current_index] : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      %previous = ttg.memdesc_index %alloc[%previous_index] : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      %copy = ttg.async_copy_global_to_local %ptrs, %current : tensor<16x16x!tt.ptr<f16>, #slot_blocked> -> <16x16xf16, #slot_shared, #smem, mutable>
      %commit = ttg.async_commit_group tokens %copy
      %wait = ttg.async_wait {num = 1 : i32}
      ttg.barrier all
      %loaded = ttg.local_load %previous token %wait : !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable> -> tensor<16x16xf16, #slot_blocked>
      ttg.barrier all
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    ttg.barrier all
    ttg.local_dealloc %alloc : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable>
    tt.return
  }

  tt.func @async_branch(
      %cond: i1, %ptrs: tensor<16x16x!tt.ptr<f16>, #slot_blocked>) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
    %outer_copy = ttg.async_copy_global_to_local %ptrs, %alloc : tensor<16x16x!tt.ptr<f16>, #slot_blocked> -> <16x16xf16, #slot_shared, #smem, mutable>
    %outer_commit = ttg.async_commit_group tokens %outer_copy
    scf.if %cond {
      %inner_copy = ttg.async_copy_global_to_local %ptrs, %alloc : tensor<16x16x!tt.ptr<f16>, #slot_blocked> -> <16x16xf16, #slot_shared, #smem, mutable>
      %inner_commit = ttg.async_commit_group tokens %inner_copy
    }
    %wait = ttg.async_wait %outer_commit {num = 0 : i32}
    ttg.barrier all
    ttg.local_dealloc %alloc : !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
    tt.return
  }

  tt.func @register_to_lds_staging(
      %input: tensor<16x16xf16, #slot_blocked>)
      -> tensor<16x16xf16, #slot_blocked> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %result = scf.for %i = %c0 to %c4 step %c1
        iter_args(%iter = %input) -> tensor<16x16xf16, #slot_blocked> {
      %producer = arith.addf %iter, %input : tensor<16x16xf16, #slot_blocked>
      %unrelated = arith.addf %input, %input : tensor<16x16xf16, #slot_blocked>
      %left = arith.mulf %producer, %producer : tensor<16x16xf16, #slot_blocked>
      %right = arith.subf %producer, %input : tensor<16x16xf16, #slot_blocked>
      %combined = arith.addf %left, %right : tensor<16x16xf16, #slot_blocked>
      scf.yield %combined : tensor<16x16xf16, #slot_blocked>
    }
    tt.return %result : tensor<16x16xf16, #slot_blocked>
  }

  tt.func @register_to_lds_capacity(
      %input: tensor<256x256xf16, #slot_blocked>)
      -> tensor<256x256xf16, #slot_blocked> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %result = scf.for %i = %c0 to %c4 step %c1
        iter_args(%iter = %input) -> tensor<256x256xf16, #slot_blocked> {
      %producer = arith.addf %iter, %input : tensor<256x256xf16, #slot_blocked>
      %consumer = arith.mulf %producer, %producer : tensor<256x256xf16, #slot_blocked>
      scf.yield %consumer : tensor<256x256xf16, #slot_blocked>
    }
    tt.return %result : tensor<256x256xf16, #slot_blocked>
  }

  tt.func @register_to_lds_scalar() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %value = arith.index_cast %i : index to i32
      %consumer = arith.addi %value, %value : i32
    }
    tt.return
  }

  tt.func @register_to_lds_derived(
      %input: tensor<64x128xf16, #blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %producer = arith.addf %input, %input : tensor<64x128xf16, #blocked>
      %unselected = arith.addf %producer, %input : tensor<64x128xf16, #blocked>
      %pad0 = arith.mulf %unselected, %input : tensor<64x128xf16, #blocked>
      %pad1 = arith.addf %pad0, %input : tensor<64x128xf16, #blocked>
      %slice = amdg.extract_slice %producer [0, 0] : tensor<64x128xf16, #blocked> to tensor<32x64xf16, #blocked>
      %left = arith.mulf %slice, %slice : tensor<32x64xf16, #blocked>
      %right = arith.subf %slice, %slice : tensor<32x64xf16, #blocked>
    }
    tt.return
  }

  tt.func @register_to_lds_unsupported(
      %input: tensor<16x16xf16, #slot_blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %producer = arith.addf %input, %input : tensor<16x16xf16, #slot_blocked>
      %derived = arith.mulf %producer, %input : tensor<16x16xf16, #slot_blocked>
      %consumer = arith.subf %derived, %input : tensor<16x16xf16, #slot_blocked>
    }
    tt.return
  }

  tt.func @register_to_lds_no_gain(
      %input: tensor<64x128xf16, #blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %producer = arith.addf %input, %input : tensor<64x128xf16, #blocked>
      %slice = amdg.extract_slice %producer [0, 0] : tensor<64x128xf16, #blocked> to tensor<32x64xf16, #blocked>
      %pad0 = arith.mulf %input, %input : tensor<64x128xf16, #blocked>
      %pad1 = arith.addf %pad0, %input : tensor<64x128xf16, #blocked>
      %consumer = arith.subf %slice, %slice : tensor<32x64xf16, #blocked>
    }
    tt.return
  }

  tt.func @register_to_lds_layout_chain(
      %input: tensor<1x16xi32, #reshape_src>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %producer = arith.addi %input, %input : tensor<1x16xi32, #reshape_src>
      %pad0 = arith.muli %input, %input : tensor<1x16xi32, #reshape_src>
      %pad1 = arith.addi %pad0, %input : tensor<1x16xi32, #reshape_src>
      %reshape = tt.reshape %producer allow_reorder : tensor<1x16xi32, #reshape_src> -> tensor<2x8xi32, #reshape_trans>
      %trans = tt.trans %reshape {order = array<i32: 1, 0>} : tensor<2x8xi32, #reshape_trans> -> tensor<8x2xi32, #reshape_dst>
      %convert = ttg.convert_layout %trans : tensor<8x2xi32, #reshape_dst> -> tensor<8x2xi32, #reshape_dst>
      %required = tlx.require_layout %convert : tensor<8x2xi32, #reshape_dst> -> tensor<8x2xi32, #reshape_dst>
      %consumer = arith.addi %required, %required : tensor<8x2xi32, #reshape_dst>
    }
    tt.return
  }

  tt.func @schedule_apply_fixture() -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %sum = arith.addi %c1, %c2 : i32
    tt.return %sum : i32
  }
}
