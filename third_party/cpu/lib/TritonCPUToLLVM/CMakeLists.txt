add_triton_library(TritonCPULLVM
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    TritonCPUToLLVM.cpp
    TargetInfo.cpp
    Utility.cpp

    DEPENDS
    TritonCPUConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
)
