add_triton_library(NVHopperTransforms
  MultiCTAReduction.cpp
  WarpSpecialization.cpp
  WarpSpecialization/CodePartitionUtility.cpp
  WarpSpecialization/PingPong.cpp
  WarpSpecialization/TaskIdPropagation.cpp
  WarpSpecialization/TMEMAlloc1D.cpp
  WarpSpecialization/Utility.cpp
  WarpSpecialization/WSBuffer.cpp
  WarpSpecialization/WSCodePartition.cpp
  WarpSpecialization/WSDataPartition.cpp
  WarpSpecialization/WSLowerMem.cpp
  WarpSpecialization/WSLowerToken.cpp
  WarpSpecialization/WSMemoryPlanner.cpp
  WarpSpecialization/WSSpecialize.cpp
  WarpSpecialization/WSTaskIdPropagate.cpp
  WarpSpecialization/WSTaskPartition.cpp
  WarpSpecialization/PartitionSchedulingMeta.cpp
  ModuloScheduling/LatencyModel.cpp
  ModuloScheduling/DataDependenceGraph.cpp
  ModuloScheduling/ModuloReservationTable.cpp
  ModuloScheduling/ModuloSchedulePass.cpp
  ModuloScheduling/ModuloWSPartitionPass.cpp
  ModuloScheduling/ModuloPipelineIR.cpp

  DEPENDS
  NVHopperTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
