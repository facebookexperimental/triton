// RUN: triton-opt -split-input-file -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=cuda:90 num-ctas=1 threads-per-warp=32 cluster-dims=1,2,1})' --verify-diagnostics %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @map_smem_to_remote(%arg: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    %c1_i32 = arith.constant 1 : i32
    %0 = ttng.map_to_remote_buffer %arg, %c1_i32: !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }
}
