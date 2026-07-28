# Hardware-agnostic modulo-scheduling core (DDG + reservation table +
# schedulers + abstract LatencyModel). Depends only on TritonGPU/Triton IR, no
# backend dialect — so any backend (NVIDIA here, AMD later) can link it and
# supply its own concrete LatencyModel subclass.
add_triton_library(TritonGPUModuloCore
  ModuloScheduling/LatencyModel.cpp
  ModuloScheduling/AMDLatencyModel.cpp
  ModuloScheduling/AMDWarpPipelinePartition.cpp
  ModuloScheduling/DataDependenceGraph.cpp
  ModuloScheduling/ModuloReservationTable.cpp
  ModuloScheduling/SwingScheduler.cpp
  ModuloScheduling/ExhaustiveScheduler.cpp
  ModuloScheduling/JointSolverScheduler.cpp
  ModuloScheduling/ModuloScheduleGraph.cpp

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)

add_triton_library(NVHopperTransforms
  MultiCTAReduction.cpp
  WarpSpecialization.cpp
  WarpSpecialization/CodePartitionUtility.cpp
  WarpSpecialization/WSAtomicBroadcast.cpp
  WarpSpecialization/PingPong.cpp
  WarpSpecialization/TaskIdPropagation.cpp
  WarpSpecialization/TMEMAlloc1D.cpp
  WarpSpecialization/Utility.cpp
  WarpSpecialization/WSBuffer.cpp
  WarpSpecialization/WSHoistTMEMStore.cpp
  WarpSpecialization/WSCodePartition.cpp
  WarpSpecialization/WSDataPartition.cpp
  WarpSpecialization/WSLowerMem.cpp
  WarpSpecialization/WSLowerToken.cpp
  WarpSpecialization/WSMemoryPlanner.cpp
  WarpSpecialization/WSMemoryPlanOrdering.cpp
  WarpSpecialization/WSMemoryPlanCost.cpp
  WarpSpecialization/WSMemoryPlanCopies.cpp
  WarpSpecialization/WSMemoryPlanSearch.cpp
  WarpSpecialization/WSMemoryPlanPackers.cpp
  WarpSpecialization/WSSpecialize.cpp
  WarpSpecialization/SinkBroadcast.cpp
  WarpSpecialization/WSTMAStoreLowering.cpp
  WarpSpecialization/WSTaskIdPropagate.cpp
  WarpSpecialization/WSTaskPartition.cpp
  WarpSpecialization/PartitionSchedulingMeta.cpp
  ModuloScheduling/NVLatencyModel.cpp
  ModuloScheduling/ModuloSchedulePass.cpp
  ModuloScheduling/ModuloWSPartitionPass.cpp
  ModuloScheduling/LLMSchedulePass.cpp
  ModuloScheduling/JointSolverSchedulePass.cpp
  ModuloScheduling/ModuloBufferAllocPass.cpp
  ModuloScheduling/ModuloExpandPass.cpp
  ModuloScheduling/ModuloLowerPass.cpp
  Insert2CTASync.cpp
  Transform2CTALoads.cpp

  DEPENDS
  NVHopperTransformsIncGen
  NVWSTableGen
  NVWSAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
  TritonGPUModuloCore
)
