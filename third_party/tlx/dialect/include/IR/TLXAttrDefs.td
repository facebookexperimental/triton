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

#endif // TLX_ATTRDEFS
