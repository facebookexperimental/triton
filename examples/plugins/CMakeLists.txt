set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(PluginPasses.h.inc -gen-pass-decls -name Plugins)
add_public_tablegen_target(TritonPluginsIncGen)

llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

set(TRITON_PLUGIN_PASSES
    TritonPluginsTestLib
    )

set(TritonPluginsTestLib_SOURCES
    TritonPlugin.cpp
    )

# Dialect Plugins
add_subdirectory(DialectPlugins)

foreach( plugin ${TRITON_PLUGIN_PASSES} )
    add_library(${plugin} SHARED ${${plugin}_SOURCES})
    add_dependencies(${plugin}
      TritonTableGen
      TritonGPUTableGen
      TritonNvidiaGPUTableGen
      TritonCanonicalizeIncGen
      TritonPluginsIncGen
    )
    target_link_libraries(${plugin} PRIVATE triton)

    # CMAKE_LIBRARY_OUTPUT_DIRECTORY is only set during the Python
    # build. It is empty if building directly from the root
    # CMakeLists.txt file. Therefore if not building from Python just
    # use the default CMake shared lib path otherwise this causes a hard
    # build error
    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
      set_target_properties(${plugin} PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
    endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)

    # A plugin resolves its MLIR/LLVM symbols against the already-loaded
    # libtriton. Under -fvisibility=hidden it keeps private copies of the
    # vague-linkage ones -- TypeID statics, vtables, RTTI -- instead of binding
    # to libtriton's, and segfaults when its pass is added to a PassManager.
    # This target only exists when TRITON_EXT_ENABLED is ON, i.e. when libtriton
    # itself is built with default visibility, so hidden is never the right
    # choice here. Buck's :TritonPluginsTestLib pins the same flag.
    target_compile_options(${plugin} PRIVATE -fvisibility=default ${TRITON_DISABLE_EH_RTTI_FLAGS})
endforeach()
