add_mlir_library(TritonTestAnalysis
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestBarrierAnalysis.cpp
  TestMembar.cpp
  TestPrintNesting.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  TritonAnalysis
)
