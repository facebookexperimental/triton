#ifndef TRITON_TOOLS_TIMETRACE_H
#define TRITON_TOOLS_TIMETRACE_H

namespace mlir {
class PassManager;
} // namespace mlir

namespace mlir::triton::tools {

// Emits one LLVM time-trace event per pass that `pm` runs, on threads where
// the LLVM time-trace profiler is enabled. Used by triton.tools.compile_trace.
void addTimeTraceInstrumentation(PassManager &pm);

} // namespace mlir::triton::tools

#endif // TRITON_TOOLS_TIMETRACE_H
