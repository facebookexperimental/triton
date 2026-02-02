add_triton_library(TritonTLXTransforms
  Fixup.cpp
  PropagateLayout.cpp
  InsertRequireLayout.cpp
  RewriteLocalAlias.cpp
  ResolvePlaceholderLayouts.cpp
  PrintTTGIRToTLX.cpp
  StorageAliasSizeDefinition.cpp
  StorageAliasAllocation.cpp
  StorageAliasLowering.cpp

  DEPENDS
  TritonTLXTransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  TLXIR
)
