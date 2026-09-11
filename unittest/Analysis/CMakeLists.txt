add_triton_ut(
  NAME TestTritonAnalysis
  SRCS UtilityTest.cpp TMAMulticastTest.cpp CallGraphTest.cpp
  LIBS
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
