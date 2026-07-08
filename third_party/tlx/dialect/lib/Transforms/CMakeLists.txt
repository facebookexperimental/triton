add_triton_library(TritonTLXTransforms
  Fixup.cpp
  PropagateLayout.cpp
  InsertRequireLayout.cpp
  RewriteLocalAlias.cpp
  ResolvePlaceholderLayouts.cpp
  PrintTTGIRToTLX.cpp
  DumpLayout.cpp
  StorageAliasSizeDefinition.cpp
  StorageAliasAllocation.cpp
  StorageAliasLowering.cpp
  BufferOffsetCalculation.cpp

  DEPENDS
  TritonTLXTransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  TritonAMDGPUTransforms
  TLXIR
)
