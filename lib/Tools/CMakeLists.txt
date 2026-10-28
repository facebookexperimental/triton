add_triton_library(TritonTools
  GenericSwizzling.cpp
  LayoutUtils.cpp
  LinearLayout.cpp
  LLVMOptions.cpp
  PluginUtils.cpp
  TimeTrace.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRPluginsLib
  f2reduce
)
