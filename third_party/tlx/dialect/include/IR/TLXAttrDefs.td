#ifndef PROTON_ATTRDEFS
#define PROTON_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "TLXDialect.td"

class TLX_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TLX_Dialect, name, traits, baseCppClass> {
}

//===----------------------------------------------------------------------===//
// Dummy Layout Attributes for Deferred Layout Resolution
//===----------------------------------------------------------------------===//

def TLX_DummyTMemLayoutAttr : TLX_Attr<"DummyTMemLayout", []> {
  let mnemonic = "dummy_tmem_layout";
  let summary = "Placeholder layout for Tensor Memory (TMEM) to be resolved after inlining";

  let description = [{
    This attribute represents a placeholder layout for tensors in Tensor Memory.
    It is generated during initial lowering when we don't have enough context
    to determine the final layout. After function inlining, a pass will resolve
    this to a concrete TensorMemoryEncodingAttr based on the tensor shape.

    Parameters:
    - shape: The shape of the tensor (e.g., [128, 64])
    - elementType: The element type (e.g., f16, bf16, f32)
    - unpacked: Whether the tensor uses unpacked format (default true)

    TODO: Remove `unpacked` parameter and infer it in the C++ resolution pass
    based on usage context.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "bool":$unpacked
  );

  let assemblyFormat = "`<` $shape `,` $elementType `,` `unpacked` `=` $unpacked `>`";
}

def TLX_DummySMemLayoutAttr : TLX_Attr<"DummySMemLayout", []> {
  let mnemonic = "dummy_smem_layout";
  let summary = "Placeholder layout for Shared Memory (SMEM) to be resolved after inlining";

  let description = [{
    This attribute represents a placeholder layout for tensors in Shared Memory.
    It is generated during initial lowering when we don't have enough context
    to determine the final layout. After function inlining, a pass will resolve
    this to either:
    - SwizzledSharedEncodingAttr (for 1D tensors)
    - NvMmaSharedEncodingAttr (for 2D+ tensors)

    Parameters:
    - shape: The shape of the tensor
    - elementType: The element type
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let assemblyFormat = "`<` $shape `,` $elementType `>`";
}

def TLX_DummyRegisterLayoutAttr : TLX_Attr<"DummyRegisterLayout", []> {
  let mnemonic = "dummy_register_layout";
  let summary = "Placeholder layout for register-distributed tensors to be resolved after inlining";

  let description = [{
    This attribute represents a placeholder layout for tensors distributed
    across registers. It is generated during initial lowering when we don't
    have enough context to determine the final distribution layout.

    After function inlining, a pass will resolve this to a concrete layout such as:
    - BlockedEncodingAttr (default blocked distribution)
    - MmaEncodingAttr (for MMA operation results)
    - DotOperandEncodingAttr (for dot operation inputs)

    Parameters:
    - shape: The shape of the tensor
    - elementType: The element type
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let assemblyFormat = "`<` $shape `,` $elementType `>`";
}

def TLX_DummyMMALayoutAttr : TLX_Attr<"DummyMMALayout", []> {
  let mnemonic = "dummy_mma_layout";
  let summary = "Placeholder layout for MMA accumulator to be resolved after inlining";

  let description = [{
    This attribute represents a placeholder layout for MMA (Matrix Multiply-Accumulate)
    accumulator tensors. It is generated during initial lowering when we don't have
    enough context to determine the final MMA layout.

    After function inlining, a pass will resolve this to NvidiaMmaEncodingAttr
    by computing the appropriate instruction shape based on the operand types
    and number of warps. This is used for Hopper (version 3) MMA operations.

    Parameters:
    - shape: The shape of the accumulator tensor
    - elementType: The element type of the accumulator
    - operandAElementType: The element type of operand A (needed for instruction shape)
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "Type":$operandAElementType
  );

  let assemblyFormat = "`<` $shape `,` $elementType `,` `opA` `=` $operandAElementType `>`";
}

def TLX_DummyDotOperandLayoutAttr : TLX_Attr<"DummyDotOperandLayout", []> {
  let mnemonic = "dummy_dot_operand_layout";
  let summary = "Placeholder layout for dot operand to be resolved after inlining";

  let description = [{
    This attribute represents a placeholder layout for dot/MMA operand tensors.
    It is generated during initial lowering when we don't have enough context
    to determine the final layout.

    After function inlining, a pass will resolve this to DotOperandEncodingAttr
    by finding the associated MMA operation and using its layout as the parent.

    Parameters:
    - shape: The shape of the operand tensor
    - elementType: The element type
    - opIdx: Operand index (0 for A, 1 for B)
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "unsigned":$opIdx
  );

  let assemblyFormat = "`<` $shape `,` $elementType `,` `opIdx` `=` $opIdx `>`";
}

#endif // PROTON_ATTRDEFS
