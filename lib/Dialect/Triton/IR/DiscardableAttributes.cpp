#include "triton/Dialect/Triton/IR/DiscardableAttributes.h"

namespace mlir::triton {

static constexpr AutoWSLoopAttrInfo kAutoWSLoopAttrs[] = {
    {kNumStagesAttrName, AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.loop_unroll_factor", AutoWSLoopAttrPropagation::NotForwarded},
    {kDisallowAccMultiBufferAttrName,
     AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.flatten", AutoWSLoopAttrPropagation::NotForwarded},
    {kWarpSpecializeAttrName, AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.multi_cta", AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"llvm.loop_annotation", AutoWSLoopAttrPropagation::NotForwarded},
    {"tt.data_partition_factor", AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.list_schedule_pick", AutoWSLoopAttrPropagation::NotForwarded},
    {"tt.mem_plan_pick", AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.merge_epilogue", AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.merge_epilogue_to_computation",
     AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.merge_correction", AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.separate_epilogue_store",
     AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.tmem_alloc_algo", AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.smem_alloc_algo", AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.smem_budget", AutoWSLoopAttrPropagation::ForwardToInnerLoop},
    {"tt.smem_circular_reuse", AutoWSLoopAttrPropagation::ForwardToInnerLoop},
};

ArrayRef<AutoWSLoopAttrInfo> getAutoWSLoopAttrs() { return kAutoWSLoopAttrs; }

SmallVector<NamedAttribute>
filterAutoWSLoopAttrs(Operation *op, AutoWSLoopAttrPropagation propagation) {
  SmallVector<NamedAttribute> attrs;
  for (const AutoWSLoopAttrInfo &attrInfo : getAutoWSLoopAttrs()) {
    if (attrInfo.propagation != propagation)
      continue;
    if (Attribute attr = op->getDiscardableAttr(attrInfo.name))
      attrs.emplace_back(attrInfo.name, attr);
  }
  return attrs;
}

SmallVector<NamedAttribute>
filterDiscardableAttrs(Operation *op, ArrayRef<StringRef> allowList) {
  SmallVector<NamedAttribute> propagatedAttrs;
  for (auto attrName : allowList) {
    Attribute attr = op->getDiscardableAttr(attrName);
    if (attr)
      propagatedAttrs.emplace_back(attrName, attr);
  }
  return propagatedAttrs;
}

} // namespace mlir::triton
