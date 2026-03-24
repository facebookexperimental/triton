add_triton_library(ProtonIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  ProtonTableGen
  ProtonAttrDefsIncGen
)

target_compile_options(ProtonIR PRIVATE -Wno-deprecated-declarations)
