// RUN: triton-opt %s --allocate-amdgpu-shared-memory="arch=gfx1250" | FileCheck %s

#padded = #ttg.padded_shared<[256:+8] {order = [1, 0], shape = [16, 32]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #padded}>

// The logical tensor is 4096 bytes, but padding gives two 2104-byte buffers
// placed in different 64-KiB partitions. Report each physical buffer and the
// actual end offset rather than the logical shape or summed buffer sizes.
// CHECK: ttg.shared = 67640 : i32
// CHECK-SAME: \22kind\22:\22explicit\22
// CHECK-SAME: \22offset\22:0
// CHECK-SAME: \22size\22:2104
// CHECK-SAME: \22kind\22:\22explicit\22
// CHECK-SAME: \22offset\22:65536
// CHECK-SAME: \22size\22:2104
// CHECK-SAME: \22name\22:\22partitioned\22
// CHECK-SAME: \22reserved_bytes\22:67640
// CHECK-SAME: \22reservations\22:[]
// CHECK-SAME: \22reserved_bytes\22:67640
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "hip:gfx1250"} {
  tt.func @partitioned() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #partitioned, #ttg.shared_memory, mutable>
    tt.return
  }
}
