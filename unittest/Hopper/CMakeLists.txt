add_triton_ut(
  NAME HopperWarpSpecializationUtilityTest
  SRCS append_to_name_loc_test.cpp
  LIBS
    MLIRIR
    MLIRSupport
)

add_triton_ut(
  NAME Z3ConstraintEncoderTest
  SRCS z3_constraint_encoder_test.cpp
  LIBS
    TritonGPUModuloCore
)

if(TRITON_ENABLE_Z3_JOINT_SOLVER)
  target_compile_definitions(Z3ConstraintEncoderTest
    PRIVATE TRITON_ENABLE_Z3_JOINT_SOLVER=1
  )
endif()
