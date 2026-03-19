set(TLX_PLUGIN_SOURCES
    TLXLocalAllocPlugin.cpp
    TLXConversionPatterns.cpp
)

add_library(TLXMemOpsPlugin SHARED ${TLX_PLUGIN_SOURCES})

add_dependencies(TLXMemOpsPlugin
  triton
)

# Link against libtriton.so to resolve Triton dialect symbols
# (e.g., MemDescType TypeID, SharedMemorySpaceAttr, etc.)
# that are statically linked into libtriton.so.
target_link_libraries(TLXMemOpsPlugin PRIVATE triton MLIRPass)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(TLXMemOpsPlugin PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)

target_compile_options(TLXMemOpsPlugin PRIVATE -fvisibility=hidden ${TRITON_DISABLE_EH_RTTI_FLAGS})
