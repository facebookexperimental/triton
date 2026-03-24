add_triton_library(ProtonAnalysis
	ScopeIdAllocation.cpp

  DEPENDS
	ProtonTableGen

  LINK_LIBS PUBLIC
	ProtonIR
	TritonAnalysis
)

target_compile_options(ProtonAnalysis PRIVATE -Wno-deprecated-declarations)
