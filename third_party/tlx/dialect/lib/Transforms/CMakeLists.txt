add_triton_library(TritonTLXTransforms
  Fixup.cpp

  DEPENDS
  TritonTLXTransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  TLXIR
)
