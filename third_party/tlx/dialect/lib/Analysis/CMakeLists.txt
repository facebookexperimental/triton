add_triton_library(TLXAnalysis
  LayoutPropagation.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  TLXIR
)

target_compile_options(TLXAnalysis PRIVATE -Wno-deprecated-declarations)
