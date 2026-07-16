#ifndef TLX_ATTRDEFS
#define TLX_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "TLXDialect.td"

class TLX_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TLX_Dialect, name, traits, baseCppClass> {
}

def TLX_LayoutEncodingTrait : AttrInterface<"LayoutEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Get the CGA layout backing this encoding.",
                    "::mlir::triton::gpu::CGAEncodingAttr", "getCGALayout">,
    InterfaceMethod<"Get the rank of the layout.", "unsigned", "getRank",
                    (ins), [{}], [{
      return $_attr.getCGALayout().getRank();
    }]>
  ];
}

def TLX_DeclareLayoutEncodingMethods : DeclareAttrInterfaceMethods<
  TLX_LayoutEncodingTrait, ["getCGALayout"]>;

// Mirror of ttg::SharedEncodingTrait so TLX-owned attributes can advertise
// themselves as shared-memory encodings (accepted by the MemDescType verifier).
def TLX_SharedEncodingTrait : AttrInterface<"SharedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Return the default alignment for the layout.",
                    "int32_t", "getAlignment", (ins), [{}], [{ return 16; }]>,
  ];
}

def TLX_DeclareSharedEncodingMethods : DeclareAttrInterfaceMethods<
  TLX_SharedEncodingTrait, ["getAlignment"]>;

// Mirror of ttg::PinnedEncodingTrait so a TLX attribute can advertise itself as
// user-pinned to core TritonGPU passes. Must stay in sync with the core def
// (same cppNamespace + methods) so both resolve to the same interface.
def TLX_PinnedEncodingTrait : AttrInterface<"PinnedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Return the concrete layout this wrapper pins.",
                    "::mlir::Attribute", "getPinnedLayout">,
  ];
}

def TLX_DistributedEncodingTrait : AttrInterface<"DistributedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Get the order of reps.",
                    "SmallVector<unsigned>", "getRepOrder">,
    InterfaceMethod<"Return total element size per thread.",
                    "unsigned", "getTotalElemsPerThread",
                    (ins "ArrayRef<int64_t>":$shape),
                    /*defaultImplementation=*/[{
      return ::mlir::triton::gpu::toLinearEncoding($_self, shape)
          .getTotalElemsPerThread(shape);
    }]>,
    InterfaceMethod<"Return element size per thread in each dimension.",
                    "SmallVector<unsigned>", "getElemsPerThread",
                    (ins "ArrayRef<int64_t>":$shape),
                    /*defaultImplementation=*/[{
      return ::mlir::triton::gpu::toLinearEncoding($_self, shape)
          .getElemsPerThread(shape);
    }]>,
    InterfaceMethod<"Convert to LinearLayout.",
                    "::mlir::triton::LinearLayout", "toLinearLayout",
                    (ins "ArrayRef<int64_t>":$shape)>,
  ];
}

//===----------------------------------------------------------------------===//
// Dummy Layout Attributes for Deferred Layout Resolution
//===----------------------------------------------------------------------===//

def TLX_DummyRegisterLayoutAttr : TLX_Attr<"DummyRegisterLayout"> {
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

def TLX_DummyTMEMLayoutAttr : TLX_Attr<"DummyTMEMLayout"> {
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

//===----------------------------------------------------------------------===//
// No-verify Layout Attribute
//===----------------------------------------------------------------------===//

def TLX_NoVerifyLayoutAttr
    : TLX_Attr<"NoVerifyLayout",
               [TLX_DistributedEncodingTrait, TLX_LayoutEncodingTrait,
                TLX_DeclareLayoutEncodingMethods]> {
  let mnemonic = "no_verify_layout";
  let summary = "TLX-owned wrapper for distributed layouts that defer layout verification";

  let description = [{
    This attribute wraps a concrete distributed layout while TLX-generated IR
    still needs to pass through function inlining. It implements the same
    distributed layout interfaces by delegating to the wrapped layout, but its
    TLX dialect ownership avoids the normal TritonGPU tensor-layout verifier.

    The post-inline TLX placeholder-resolution pass unwraps this attribute and
    verifies that none remain.
  }];

  let parameters = (ins "Attribute":$layout);

  let extraClassDeclaration = [{
    SmallVector<unsigned> getRepOrder() const;
    ::mlir::triton::LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  }];

  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $layout `>`";
}

//===----------------------------------------------------------------------===//
// User Layout Attribute
//===----------------------------------------------------------------------===//

def TLX_UserLayoutAttr
    : TLX_Attr<"UserLayout",
               [TLX_DistributedEncodingTrait, TLX_SharedEncodingTrait,
                TLX_PinnedEncodingTrait, TLX_LayoutEncodingTrait,
                TLX_DeclareLayoutEncodingMethods,
                TLX_DeclareSharedEncodingMethods]> {
  let mnemonic = "user_layout";
  let summary = "TLX wrapper marking a user-specified layout (register or shared)";

  let description = [{
    A general wrapper for a concrete layout that the user pinned explicitly --
    either a distributed/register layout (e.g. #ttg.linear, on a value fed to
    `tlx.local_load`) or a shared-memory layout (e.g. #ttg.swizzled_shared /
    #ttg.padded_shared / #ttg.nvmma_shared, on a `tlx.local_alloc`).

    The wrapper advertises both the distributed and shared layout interfaces and
    delegates every query to the wrapped layout (dispatching on the concrete
    inner), so it is a valid encoding on both RankedTensorType and MemDescType
    and is transparent to trait-based code.

    Its presence is a hard "do not retag this value" signal: `tlx-propagate-layout`
    treats a value whose encoding is this wrapper as a fixed source and never
    rewrites it to satisfy a consumer, then unwraps it back to the concrete
    wrapped layout once the constraint has been honored.
  }];

  let parameters = (ins "Attribute":$layout);

  let extraClassDeclaration = [{
    SmallVector<unsigned> getRepOrder() const;
    ::mlir::triton::LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
    ::mlir::Attribute getPinnedLayout() const { return getLayout(); }
  }];

  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $layout `>`";
}

#endif // TLX_ATTRDEFS
