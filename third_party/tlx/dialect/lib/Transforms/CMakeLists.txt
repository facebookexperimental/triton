add_triton_library(TritonTLXTransforms
  Fixup.cpp
  PropagateLayout.cpp
  InsertRequireLayout.cpp
  RewriteLocalAlias.cpp
  ResolvePlaceholderLayouts.cpp

  DEPENDS
  TritonTLXTransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  TLXIR
)
