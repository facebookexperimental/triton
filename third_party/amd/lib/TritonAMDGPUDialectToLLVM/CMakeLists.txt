add_triton_library(TritonAMDGPUDialectToLLVM
    TritonAMDGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp
    InThreadTransposeOpToTTG.cpp
    ConcatOpToLLVM.cpp
    PackedDot2OpToLLVM.cpp
    ScaledDowncastToLLVM.cpp
    ScaledUpcastToLLVM.cpp

    DEPENDS
    TritonAMDGPUIR
)
