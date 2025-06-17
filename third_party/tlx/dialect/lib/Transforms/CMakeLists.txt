add_triton_library(TritonTLXTransforms
  Fixup.cpp
  PropagateLayout.cpp

  DEPENDS
  TritonTLXTransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  TLXIR
)
