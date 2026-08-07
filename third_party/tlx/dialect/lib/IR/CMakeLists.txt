add_triton_library(TLXIR
  Dialect.cpp
  Ops.cpp
  Traits.cpp
  Types.cpp

  DEPENDS
  TLXTableGen
  TLXTypesIncGen
  TLXLayoutInterfaceIncGen
  TLXAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
