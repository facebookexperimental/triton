add_triton_library(TritonTools
  GenericSwizzling.cpp
  LayoutUtils.cpp
  LinearLayout.cpp
  ModularArithmetic.cpp
  PluginUtils.cpp
  RadixLayout.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRPluginsLib
  f2reduce
)
