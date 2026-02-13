set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# For dialect
set(LLVM_TARGET_DEFINITIONS TLXDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tlx)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tlx)

# For types
set(LLVM_TARGET_DEFINITIONS TLXTypes.td)
mlir_tablegen(TLXTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TLXTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(TLXTypesEnums.h.inc -gen-enum-decls)
mlir_tablegen(TLXTypesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TLXTypesIncGen)

# For ops
set(LLVM_TARGET_DEFINITIONS TLXOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

add_mlir_doc(TLXDialect TLXDialect dialects/ -gen-dialect-doc)
add_mlir_doc(TLXOps TLXOps dialects/ -gen-op-doc)
add_public_tablegen_target(TLXTableGen)


set(LLVM_TARGET_DEFINITIONS TLXAttrDefs.td)
mlir_tablegen(TLXAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TLXAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(TLXAttrDefsIncGen)
