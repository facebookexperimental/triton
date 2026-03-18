add_triton_library(TLXIR
  Dialect.cpp
  Ops.cpp
  Traits.cpp

  DEPENDS
  TLXTableGen
  TLXAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
