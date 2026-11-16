add_triton_ut(
  NAME NvmmaSmemAttrsTest
  SRCS NvmmaSmemAttrsTest.cpp
  LIBS
    TritonAnalysis
    TritonGPUIR
    TritonNvidiaGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)

add_triton_ut(
  NAME Tcgen05ShapeTest
  SRCS Tcgen05ShapeTest.cpp
  LIBS
    TritonAnalysis
    TritonGPUIR
    TritonNvidiaGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
