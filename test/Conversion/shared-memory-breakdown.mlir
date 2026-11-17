// RUN: triton-opt %s --split-input-file --allocate-shared-memory | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// Both 128-byte allocations reuse offset zero; the total is not 256 bytes.
// CHECK: ttg.shared = 128 : i32
// CHECK-SAME: \22kind\22:\22explicit\22
// CHECK-SAME: \22offset\22:0
// CHECK-SAME: \22size\22:128
// CHECK-SAME: \22kind\22:\22explicit\22
// CHECK-SAME: \22offset\22:0
// CHECK-SAME: \22size\22:128
// CHECK-SAME: \22name\22:\22reuse\22
// CHECK-SAME: \22reserved_bytes\22:128
// CHECK-SAME: \22reservations\22:[]
// CHECK-SAME: \22reserved_bytes\22:128
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @reuse() -> tensor<128xi8, #blocked> {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %x = ttg.local_load %a : !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable> -> tensor<128xi8, #blocked>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %y = ttg.local_load %b : !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable> -> tensor<128xi8, #blocked>
    %z = arith.addi %x, %y : tensor<128xi8, #blocked>
    tt.return %z : tensor<128xi8, #blocked>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// A descriptor alias is not a second allocation.
// CHECK: ttg.shared = 128 : i32
// CHECK-SAME: \22kind\22:\22explicit\22
// CHECK-SAME: \22operation\22:\22ttg.local_alloc\22
// CHECK-SAME: \22size\22:128}]
// CHECK-SAME: \22name\22:\22alias\22
// CHECK-SAME: \22omitted_allocations\22:0
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @alias() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %alias = ttg.memdesc_subslice %a [0] : !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// Identify barrier storage using the allocator's actual init-barrier criterion.
// CHECK: ttg.shared = 8 : i32
// CHECK-SAME: \22barrier_backed\22:true
// CHECK-SAME: \22kind\22:\22explicit\22
// CHECK-SAME: \22operation\22:\22ttg.local_alloc\22
// CHECK-SAME: \22size\22:8
// CHECK-SAME: \22name\22:\22barrier\22
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @barrier() {
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dst = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1]}>

// Shared-memory layout conversion scratch carries its original source location.
// CHECK: ttg.shared = {{[1-9][0-9]*}} : i32
// CHECK-SAME: \22kind\22:\22scratch\22
// CHECK-SAME: /tmp/smem-diagnostic.py
// CHECK-SAME: \22operation\22:\22ttg.convert_layout\22
// CHECK-SAME: \22name\22:\22scratch\22
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @scratch(%x: tensor<32x32xf32, #src>) -> tensor<32x32xf32, #dst> {
    %cvt = ttg.convert_layout %x : tensor<32x32xf32, #src> -> tensor<32x32xf32, #dst> loc("/tmp/smem-diagnostic.py":17:5)
    tt.return %cvt : tensor<32x32xf32, #dst>
  }
}

// -----

#padded = #ttg.padded_shared<[128:+32] {order = [1, 0], shape = [16, 32]}>

// Report the physical padded size, not the tensor's logical 1024 bytes.
// CHECK: ttg.shared = 1216 : i32
// CHECK-SAME: \22kind\22:\22explicit\22
// CHECK-SAME: \22size\22:1216
// CHECK-SAME: \22name\22:\22padded\22
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @padded() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #padded, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// Independent root footprints are combined by max, not sum, and sorted by name.
// CHECK: ttg.shared = 128 : i32
// CHECK-SAME: \22name\22:\22a_root\22
// CHECK-SAME: \22reserved_bytes\22:128
// CHECK-SAME: \22name\22:\22z_root\22
// CHECK-SAME: \22reserved_bytes\22:64
// CHECK-SAME: \22reservations\22:[]
// CHECK-SAME: \22reserved_bytes\22:128
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @z_root() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<64xi8, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
  tt.func @a_root() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// A callee footprint is represented once by the caller's virtual buffer.
// CHECK: ttg.shared = 128 : i32
// CHECK-SAME: \22kind\22:\22virtual\22
// CHECK-SAME: \22operation\22:\22tt.call\22
// CHECK-SAME: \22size\22:128}]
// CHECK-SAME: \22name\22:\22caller\22
// CHECK-SAME: \22reserved_bytes\22:128
// CHECK-SAME: \22reservations\22:[]
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func private @callee() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
  tt.func @caller() {
    tt.call @callee() : () -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// The report is bounded to 32 largest records per root without losing its total.
// CHECK: ttg.shared = 128 : i32
// CHECK-SAME: \22name\22:\22bounded\22
// CHECK-SAME: \22omitted_allocations\22:1
// CHECK-SAME: \22reserved_bytes\22:128
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @bounded() {
    %a0 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a1 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a2 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a3 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a4 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a5 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a6 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a7 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a8 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a9 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a10 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a11 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a12 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a13 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a14 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a15 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a16 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a17 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a18 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a19 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a20 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a21 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a22 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a23 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a24 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a25 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a26 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a27 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a28 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a29 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a30 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a31 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    %a32 = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// Optional diagnostic JSON must not assert on a non-UTF-8 MLIR symbol name.
// CHECK: ttg.shared = 128 : i32
// CHECK-SAME: ttg.shared_memory_breakdown
// CHECK-SAME: \22name\22:\22invalid
// CHECK-SAME: \22reserved_bytes\22:128
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @"invalid\FF"() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}
