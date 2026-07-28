add_triton_ut(
  NAME HopperWarpSpecializationUtilityTest
  SRCS append_to_name_loc_test.cpp
  LIBS
    MLIRIR
    MLIRSupport
)

add_triton_ut(
  NAME ModuloScheduleTest
  SRCS modulo_schedule_test.cpp
  LIBS
    TritonGPUModuloCore
    TritonGPUTransforms
    TritonNvidiaGPUIR
    TritonNvidiaGPUTransforms
    TLXIR
)
