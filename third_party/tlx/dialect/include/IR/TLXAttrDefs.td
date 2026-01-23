#ifndef TLX_ATTRDEFS
#define TLX_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "TLXDialect.td"

class TLX_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TLX_Dialect, name, traits, baseCppClass> {
}

//===----------------------------------------------------------------------===//
// Dummy Layout Attributes for Deferred Layout Resolution
//===----------------------------------------------------------------------===//

def TLX_DummyRegisterLayoutAttr : TLX_Attr<"DummyRegisterLayout", []> {
  let mnemonic = "dummy_register_layout";
  let summary = "Placeholder layout for register-distributed tensors to be resolved after inlining";

  let description = [{
    This attribute represents a placeholder layout for tensors distributed
    across registers. It is generated during initial lowering when we don't
    have enough context to determine the final distribution layout.

    After function inlining, a pass will resolve this to a concrete layout such as:
    - BlockedEncodingAttr (default blocked distribution)
    - TMEM-compatible BlockedEncodingAttr (for tensors loaded from TMEM)
    - MmaEncodingAttr (for MMA operation results)
    - DotOperandEncodingAttr (for dot operation inputs)

    Parameters:
    - shape: The shape of the tensor
    - elementType: The element type
    - tmemCompatible: If true, create a layout compatible with TMEM load/store
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "bool":$tmemCompatible
  );

  let assemblyFormat = "`<` `[` $shape `]` `,` $elementType `,` $tmemCompatible `>`";
}

def TLX_DummyTMEMLayoutAttr : TLX_Attr<"DummyTMEMLayout", []> {
  let mnemonic = "dummy_tmem_layout";
  let summary = "Placeholder layout for TMEM tensors to be resolved during layout propagation";

  let description = [{
    This attribute represents a placeholder layout for tensors in Tensor Memory (TMEM).
    It is used when we don't know the final TMEM layout at allocation time.

    During layout propagation, this will be resolved to a concrete TMEM layout:
    - TensorMemoryEncodingAttr (for regular TMEM data)
    - TensorMemoryScalesEncodingAttr (for scales in scaled MMA operations)

    The resolution depends on how the TMEM buffer is used (e.g., as scales in tmem_copy).
  }];

  let parameters = (ins);

  let assemblyFormat = "";
}

#endif // TLX_ATTRDEFS
