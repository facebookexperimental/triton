add_mlir_library(TritonTestAnalysis
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestBackwardSliceWithWS.cpp
  TestMembar.cpp
  TestPrintNesting.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  TritonAnalysis
  TritonNvidiaGPUTransforms
)
