add_triton_library(TritonTLXTransforms
  AttachMetadata.cpp

  DEPENDS
  TritonTLXTransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  TLXIR
)
