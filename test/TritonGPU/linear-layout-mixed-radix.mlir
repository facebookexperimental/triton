// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s | env TRITON_ALLOW_NPOT=1 triton-opt | FileCheck %s
// RUN: triton-opt %s | FileCheck %s --check-prefix=OFF

// An absent inputRadices entry (lane) is all radix 2.
// CHECK{LITERAL}: #generic_linear = #ttg.generic_linear<{register = [[1], [4]], lane = [[2]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2], warp = [], block = []}, outDimSizes = [12]}>
// Control: a radix-2 layout prints without metadata, and explicit radix-2
// metadata parses to the same attribute.
// CHECK{LITERAL}: #generic_linear1 = #ttg.generic_linear<{register = [[1], [2]], lane = [], warp = [], block = []}>
// CHECK: test.pow2 = #generic_linear1, test.pow2_explicit = #generic_linear1
// Flag off reads the bases only, as before metadata existed.
// OFF{LITERAL}: #generic_linear = #ttg.generic_linear<{register = [[1], [4]], lane = [[2]], warp = [], block = []}>
#exact = #ttg.generic_linear<{
  register = [[1], [4]], lane = [[2]], warp = [], block = [],
  inputRadices = {register = [2, 3], warp = [], block = []},
  outDimSizes = [12]
}>
#pow2 = #ttg.generic_linear<{register = [[1], [2]], lane = [], warp = [], block = []}>
#pow2_explicit = #ttg.generic_linear<{
  register = [[1], [2]], lane = [], warp = [], block = [],
  inputRadices = {register = [2, 2], lane = [], warp = [], block = []},
  outDimSizes = [4]
}>

module attributes {test.exact = #exact, test.pow2 = #pow2, test.pow2_explicit = #pow2_explicit} {
}
