#ifndef PROTON_ATTRDEFS
#define PROTON_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "TLXDialect.td"

class TLX_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TLX_Dialect, name, traits, baseCppClass> {
}

#endif // PROTON_ATTRDEFS
