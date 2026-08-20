// RUN: triton-opt %s -split-input-file --tritongpu-pipeline="num-stages=2" --canonicalize | FileCheck %s
//
// Reduced from the physical computation partition of the HSTU self-attention
// backward benchmark after AutoWS code partitioning.  The source kernel uses a
// tensor causal mask in arith.select rather than an explicit scalar scf.if.
// The partition pipeline must recognize that the mask is needed only for the
// first M tile, peel it, and leave an all-true unmasked remainder.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#m_slice = #ttg.slice<{dim = 0, parent = #blocked}>
#n_slice = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_causal_where
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: %[[HAS_FIRST:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : i32
  // CHECK: scf.if %[[HAS_FIRST]]
  // The straight-line first tile retains the real triangular mask.
  // CHECK: arith.cmpi eq
  // CHECK: arith.cmpi sgt
  // CHECK: arith.ori
  // CHECK-COUNT-2: arith.select
  // The remainder starts at lb + 128 and no longer carries tensor predicates.
  // CHECK: %[[REMAINDER_LB:.*]] = arith.addi
  // CHECK: scf.for %{{.*}} = %[[REMAINDER_LB]]
  // CHECK-NOT: arith.select
  // CHECK: tt.store
  // CHECK-NOT: arith.select
  // CHECK: tt.store
  // CHECK-NOT: arith.select
  // CHECK: }
  tt.func public @peel_causal_where(%lb: i32, %ub: i32,
                                    %out0: !tt.ptr<f32>,
                                    %out1: !tt.ptr<f32>) {
    ttg.warp_specialize(%lb, %ub, %out0, %out1) attributes {requestedRegisters = array<i32: -1>, ttg.partition.types = ["computation"]}
    default {
      ttg.warp_yield
    }
    partition0(%part_lb: i32, %part_ub: i32,
               %part_out0: !tt.ptr<f32>,
               %part_out1: !tt.ptr<f32>) num_warps(4) {
      %c128 = arith.constant 128 : i32
      %range_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #m_slice>
      %range_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #n_slice>
      %zero_i32 = arith.constant dense<0> : tensor<128x128xi32, #blocked>
      %zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
      %one_f32 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
      %ptrs0 = tt.splat %part_out0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      %ptrs1 = tt.splat %part_out1 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      scf.for %m = %part_lb to %part_ub step %c128 : i32 {
        %m_base = tt.splat %m : i32 -> tensor<128xi32, #m_slice>
        %m_offsets = arith.addi %m_base, %range_m : tensor<128xi32, #m_slice>
        %n_base = tt.splat %part_lb : i32 -> tensor<128xi32, #n_slice>
        %n_offsets = arith.addi %n_base, %range_n : tensor<128xi32, #n_slice>
        %m_expanded = tt.expand_dims %m_offsets {axis = 0 : i32} : tensor<128xi32, #m_slice> -> tensor<1x128xi32, #blocked>
        %n_expanded = tt.expand_dims %n_offsets {axis = 1 : i32} : tensor<128xi32, #n_slice> -> tensor<128x1xi32, #blocked>
        %m_matrix = tt.broadcast %m_expanded : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %n_matrix = tt.broadcast %n_expanded : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %diagonal = arith.cmpi eq, %m_matrix, %n_matrix : tensor<128x128xi32, #blocked>
        %delta = arith.subi %m_matrix, %n_matrix : tensor<128x128xi32, #blocked>
        %below_diagonal = arith.cmpi sgt, %delta, %zero_i32 : tensor<128x128xi32, #blocked>
        %causal_mask = arith.ori %diagonal, %below_diagonal : tensor<128x128xi1, #blocked>
        %masked0 = arith.select %causal_mask, %one_f32, %zero_f32 : tensor<128x128xi1, #blocked>, tensor<128x128xf32, #blocked>
        %masked1 = arith.select %causal_mask, %one_f32, %zero_f32 : tensor<128x128xi1, #blocked>, tensor<128x128xf32, #blocked>
        tt.store %ptrs0, %masked0 : tensor<128x128x!tt.ptr<f32>, #blocked>
        tt.store %ptrs1, %masked1 : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      ttg.warp_return
    } : (i32, i32, !tt.ptr<f32>, !tt.ptr<f32>) -> ()
    tt.return
  }
}

// -----

// The same causal mask after canonicalization folds the eq/sgt disjunction into
// a single m >= n comparison.  Peeling must recognize that spelling too, so the
// pattern does not depend on which passes ran before it.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#m_slice = #ttg.slice<{dim = 0, parent = #blocked}>
#n_slice = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_causal_sge
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: %[[HAS_FIRST:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : i32
  // CHECK: scf.if %[[HAS_FIRST]]
  // The straight-line first tile retains the real triangular mask.
  // CHECK: arith.cmpi sge
  // CHECK-COUNT-2: arith.select
  // The remainder starts at lb + 128 and no longer carries tensor predicates.
  // CHECK: %[[REMAINDER_LB:.*]] = arith.addi
  // CHECK: scf.for %{{.*}} = %[[REMAINDER_LB]]
  // CHECK-NOT: arith.select
  // CHECK: tt.store
  // CHECK-NOT: arith.select
  // CHECK: tt.store
  // CHECK-NOT: arith.select
  // CHECK: }
  tt.func public @peel_causal_sge(%lb: i32, %ub: i32,
                                  %out0: !tt.ptr<f32>,
                                  %out1: !tt.ptr<f32>) {
    ttg.warp_specialize(%lb, %ub, %out0, %out1) attributes {requestedRegisters = array<i32: -1>, ttg.partition.types = ["computation"]}
    default {
      ttg.warp_yield
    }
    partition0(%part_lb: i32, %part_ub: i32,
               %part_out0: !tt.ptr<f32>,
               %part_out1: !tt.ptr<f32>) num_warps(4) {
      %c128 = arith.constant 128 : i32
      %range_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #m_slice>
      %range_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #n_slice>
      %zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
      %one_f32 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
      %ptrs0 = tt.splat %part_out0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      %ptrs1 = tt.splat %part_out1 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      scf.for %m = %part_lb to %part_ub step %c128 : i32 {
        %m_base = tt.splat %m : i32 -> tensor<128xi32, #m_slice>
        %m_offsets = arith.addi %m_base, %range_m : tensor<128xi32, #m_slice>
        %n_base = tt.splat %part_lb : i32 -> tensor<128xi32, #n_slice>
        %n_offsets = arith.addi %n_base, %range_n : tensor<128xi32, #n_slice>
        %m_expanded = tt.expand_dims %m_offsets {axis = 0 : i32} : tensor<128xi32, #m_slice> -> tensor<1x128xi32, #blocked>
        %n_expanded = tt.expand_dims %n_offsets {axis = 1 : i32} : tensor<128xi32, #n_slice> -> tensor<128x1xi32, #blocked>
        %m_matrix = tt.broadcast %m_expanded : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %n_matrix = tt.broadcast %n_expanded : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %causal_mask = arith.cmpi sge, %m_matrix, %n_matrix : tensor<128x128xi32, #blocked>
        %masked0 = arith.select %causal_mask, %one_f32, %zero_f32 : tensor<128x128xi1, #blocked>, tensor<128x128xf32, #blocked>
        %masked1 = arith.select %causal_mask, %one_f32, %zero_f32 : tensor<128x128xi1, #blocked>, tensor<128x128xf32, #blocked>
        tt.store %ptrs0, %masked0 : tensor<128x128x!tt.ptr<f32>, #blocked>
        tt.store %ptrs1, %masked1 : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      ttg.warp_return
    } : (i32, i32, !tt.ptr<f32>, !tt.ptr<f32>) -> ()
    tt.return
  }
}
