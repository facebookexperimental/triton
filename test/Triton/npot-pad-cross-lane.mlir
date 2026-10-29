// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad -verify-diagnostics | FileCheck %s

// A sum pads with its identity, -0.0.
// CHECK-LABEL: @sum
tt.func @sum(%ptr: !tt.ptr<f32>) {
  // CHECK: %[[LIVE:.*]] = arith.cmpi slt
  // CHECK: %[[PAD:.*]] = arith.constant dense<-0.000000e+00> : tensor<128xf32>
  // CHECK: %[[X:.*]] = arith.select %[[LIVE]], %{{.*}}, %[[PAD]] : tensor<128xi1>, tensor<128xf32>
  // CHECK: "tt.reduce"(%[[X]])
  %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %x = tt.load %p : tensor<96x!tt.ptr<f32>>
  %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.addf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<96xf32>) -> f32
  tt.return
}

// -----

// Two sums in one reduce pad each source with its own identity.
// CHECK-LABEL: @sum_pair
tt.func @sum_pair(%fptr: !tt.ptr<f32>, %iptr: !tt.ptr<i32>) {
  // CHECK: %[[LIVE:.*]] = arith.cmpi slt
  // CHECK: %[[FPAD:.*]] = arith.constant dense<-0.000000e+00> : tensor<128xf32>
  // CHECK: %[[X:.*]] = arith.select %[[LIVE]], %{{.*}}, %[[FPAD]]
  // CHECK: %[[IPAD:.*]] = arith.constant dense<0> : tensor<128xi32>
  // CHECK: %[[Y:.*]] = arith.select %[[LIVE]], %{{.*}}, %[[IPAD]]
  // CHECK: "tt.reduce"(%[[X]], %[[Y]])
  %fp = tt.splat %fptr : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %x = tt.load %fp : tensor<96x!tt.ptr<f32>>
  %ip = tt.splat %iptr : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>>
  %y = tt.load %ip : tensor<96x!tt.ptr<i32>>
  %s:2 = "tt.reduce"(%x, %y) <{axis = 0 : i32}> ({
  ^bb0(%x1: f32, %y1: i32, %x2: f32, %y2: i32):
    %sx = arith.addf %x1, %x2 : f32
    %sy = arith.addi %y1, %y2 : i32
    tt.reduce.return %sx, %sy : f32, i32
  }) : (tensor<96xf32>, tensor<96xi32>) -> (f32, i32)
  tt.return
}

// -----

// A two-sum reduce over a pow2 axis of an NPOT tensor needs no pad value.
// CHECK-LABEL: @sum_pair_pow2_axis
tt.func @sum_pair_pow2_axis(%ptr: !tt.ptr<f32>) {
  // CHECK-NOT: arith.select
  // CHECK: "tt.reduce"
  %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
  %x = tt.load %p : tensor<3x32x!tt.ptr<f32>>
  %s:2 = "tt.reduce"(%x, %x) <{axis = 1 : i32}> ({
  ^bb0(%a1: f32, %b1: f32, %a2: f32, %b2: f32):
    %a = arith.addf %a1, %a2 : f32
    %b = arith.addf %b1, %b2 : f32
    tt.reduce.return %a, %b : f32, f32
  }) : (tensor<3x32xf32>, tensor<3x32xf32>) -> (tensor<3xf32>, tensor<3xf32>)
  tt.return
}

// -----

// A max pads with its identity, NaN for maxnumf, and needs no gather.
// CHECK-LABEL: @max
tt.func @max(%ptr: !tt.ptr<f32>) {
  // CHECK: %[[PAD:.*]] = arith.constant dense<0xFFC00000> : tensor<128xf32>
  // CHECK: %[[X:.*]] = arith.select %{{.*}}, %{{.*}}, %[[PAD]]
  // CHECK-NOT: tt.gather
  // CHECK: "tt.reduce"(%[[X]])
  %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %x = tt.load %p : tensor<96x!tt.ptr<f32>>
  %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.maxnumf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<96xf32>) -> f32
  tt.return
}

// -----

// f8E4M3FN has no inf, so a maximumf pads with its lowest finite value.
// CHECK-LABEL: @max_no_inf
tt.func @max_no_inf(%ptr: !tt.ptr<f8E4M3FN>) {
  // CHECK: arith.constant dense<-4.480000e+02> : tensor<128xf8E4M3FN>
  // CHECK-NOT: tt.gather
  // CHECK: "tt.reduce"
  %p = tt.splat %ptr : !tt.ptr<f8E4M3FN> -> tensor<96x!tt.ptr<f8E4M3FN>>
  %x = tt.load %p : tensor<96x!tt.ptr<f8E4M3FN>>
  %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f8E4M3FN, %b: f8E4M3FN):
    %c = arith.maximumf %a, %b : f8E4M3FN
    tt.reduce.return %c : f8E4M3FN
  }) : (tensor<96xf8E4M3FN>) -> f8E4M3FN
  tt.return
}

// -----

// An argmax pads value and index with a copy of the last live lane.
// CHECK-LABEL: @argmax
tt.func @argmax(%ptr: !tt.ptr<f32>) {
  // CHECK: %[[LIVE:.*]] = arith.cmpi slt
  // CHECK: %[[LAST:.*]] = arith.constant dense<95> : tensor<128xi32>
  // CHECK: %[[V:.*]] = tt.gather %[[X:.*]][%[[LAST]]] {axis = 0 : i32}
  // CHECK: %[[XV:.*]] = arith.select %[[LIVE]], %[[X]], %[[V]]
  // CHECK: %[[I:.*]] = tt.gather %[[R:.*]][%{{.*}}] {axis = 0 : i32}
  // CHECK: %[[XI:.*]] = arith.select %[[LIVE]], %[[R]], %[[I]]
  // CHECK: "tt.reduce"(%[[XV]], %[[XI]])
  %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %x = tt.load %p : tensor<96x!tt.ptr<f32>>
  %s:2 = "tt.reduce"(%x, %r) <{axis = 0 : i32}> ({
  ^bb0(%v1: f32, %i1: i32, %v2: f32, %i2: i32):
    %gt = arith.cmpf ogt, %v1, %v2 : f32
    %v = arith.select %gt, %v1, %v2 : f32
    %i = arith.select %gt, %i1, %i2 : i32
    tt.reduce.return %v, %i : f32, i32
  }) : (tensor<96xf32>, tensor<96xi32>) -> (f32, i32)
  tt.return
}

// -----

// Reducing a pow2 axis of an NPOT tensor needs no pad value.
// CHECK-LABEL: @pow2_axis
tt.func @pow2_axis(%ptr: !tt.ptr<f32>) {
  // CHECK-NOT: arith.select
  // CHECK: "tt.reduce"
  %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
  %x = tt.load %p : tensor<3x32x!tt.ptr<f32>>
  %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.addf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<3x32xf32>) -> tensor<3xf32>
  tt.return
}

// -----

// A reverse scan needs a pad value; a forward scan does not.
// CHECK-LABEL: @scans
tt.func @scans(%ptr: !tt.ptr<i32>) {
  // CHECK: %[[PAD:.*]] = arith.constant dense<0> : tensor<128xi32>
  // CHECK: %[[X:.*]] = arith.select %{{.*}}, %{{.*}}, %[[PAD]]
  // CHECK: "tt.scan"(%[[X]]) <{axis = 0 : i32, reverse = true}>
  // CHECK-NOT: arith.select
  // CHECK: "tt.scan"(%{{.*}}) <{axis = 0 : i32, reverse = false}>
  %p = tt.splat %ptr : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>>
  %x = tt.load %p : tensor<96x!tt.ptr<i32>>
  %rev = "tt.scan"(%x) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.addi %a, %b : i32
    tt.scan.return %c : i32
  }) : (tensor<96xi32>) -> tensor<96xi32>
  %fwd = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.addi %a, %b : i32
    tt.scan.return %c : i32
  }) : (tensor<96xi32>) -> tensor<96xi32>
  tt.return
}

// -----

tt.func @no_pad_value(%ptr: !tt.ptr<f32>) {
  %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
  %x = tt.load %p : tensor<3x32x!tt.ptr<f32>>
  // expected-error @+1 {{'tt.reduce' op over an NPOT axis with a combiner that has no pad value is unsupported}}
  %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.subf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<3x32xf32>) -> tensor<32xf32>
  tt.return
}

// -----

// One result without a pad value rejects the whole reduce.
tt.func @pair_no_pad_value(%ptr: !tt.ptr<f32>) {
  %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %x = tt.load %p : tensor<96x!tt.ptr<f32>>
  // expected-error @+1 {{'tt.reduce' op over an NPOT axis with a combiner that has no pad value is unsupported}}
  %s:2 = "tt.reduce"(%x, %x) <{axis = 0 : i32}> ({
  ^bb0(%a1: f32, %b1: f32, %a2: f32, %b2: f32):
    %a = arith.addf %a1, %a2 : f32
    %b = arith.subf %b1, %b2 : f32
    tt.reduce.return %a, %b : f32, f32
  }) : (tensor<96xf32>, tensor<96xf32>) -> (f32, f32)
  tt.return
}

// -----

// CHECK-LABEL: @histogram
tt.func @histogram(%ptr: !tt.ptr<i32>) {
  // CHECK: %[[LIVE:.*]] = arith.cmpi slt
  // CHECK: tt.histogram %{{.*}}, %[[LIVE]] : tensor<128xi32> -> tensor<16xi32>
  %p = tt.splat %ptr : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>>
  %x = tt.load %p : tensor<96x!tt.ptr<i32>>
  %h = tt.histogram %x : tensor<96xi32> -> tensor<16xi32>
  tt.return
}

// -----

// CHECK-LABEL: @gather
tt.func @gather(%src: tensor<64xf32>, %ptr: !tt.ptr<i32>) {
  // CHECK: %[[LIVE:.*]] = arith.cmpi slt
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0> : tensor<128xi32>
  // CHECK: %[[IDX:.*]] = arith.select %[[LIVE]], %{{.*}}, %[[ZERO]]
  // CHECK: tt.gather %{{.*}}[%[[IDX]]] {axis = 0 : i32} : (tensor<64xf32>, tensor<128xi32>) -> tensor<128xf32>
  %p = tt.splat %ptr : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>>
  %idx = tt.load %p : tensor<96x!tt.ptr<i32>>
  %g = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<64xf32>, tensor<96xi32>) -> tensor<96xf32>
  tt.return
}

// -----

// CHECK-LABEL: @atomic_rmw
tt.func @atomic_rmw(%ptr: !tt.ptr<f32>) {
  // CHECK: %[[LIVE:.*]] = arith.cmpi slt
  // CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %{{.*}}, %[[LIVE]] : (tensor<128x!tt.ptr<f32>>, tensor<128xf32>, tensor<128xi1>) -> tensor<128xf32>
  %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %v = arith.constant dense<1.0> : tensor<96xf32>
  %r = tt.atomic_rmw fadd, relaxed, gpu, %p, %v : (tensor<96x!tt.ptr<f32>>, tensor<96xf32>) -> tensor<96xf32>
  tt.return
}

// -----

tt.func @atomic_cas(%ptr: !tt.ptr<i32>) {
  %p = tt.splat %ptr : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>>
  %v = arith.constant dense<1> : tensor<96xi32>
  // expected-error @+1 {{'tt.atomic_cas' op on an NPOT tensor is unsupported}}
  %r = tt.atomic_cas acq_rel, gpu, %p, %v, %v : (tensor<96x!tt.ptr<i32>>, tensor<96xi32>, tensor<96xi32>) -> tensor<96xi32>
  tt.return
}
