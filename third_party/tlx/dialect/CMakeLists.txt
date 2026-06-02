include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/python/src
  ${PROJECT_SOURCE_DIR}/third_party/amd/include
)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonTLX ${CMAKE_CURRENT_SOURCE_DIR}/triton_tlx.cc)
  target_link_libraries(TritonTLX PRIVATE TLXIR TritonAMDGPUIR Python3::Module pybind11::headers)
endif()
