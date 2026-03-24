add_triton_library(ProtonToProtonGPU
  ProtonToProtonGPUPass.cpp

  DEPENDS
  ProtonToProtonGPUIncGen
  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  ProtonIR
  ProtonGPUIR
)

target_compile_options(ProtonToProtonGPU PRIVATE -Wno-deprecated-declarations)
