set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# For dialect
set(LLVM_TARGET_DEFINITIONS TLXDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tlx)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tlx)

# For ops
set(LLVM_TARGET_DEFINITIONS TLXOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

add_mlir_doc(TLXDialect TLXDialect dialects/ -gen-dialect-doc)
add_mlir_doc(TLXOps TLXOps dialects/ -gen-op-doc)
add_public_tablegen_target(TLXTableGen)
