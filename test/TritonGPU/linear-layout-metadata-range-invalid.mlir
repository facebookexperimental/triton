// RUN: not env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file 2>&1 | FileCheck %s

// CHECK: error: Value for 'register' must fit int32 and exceed 1
#bad_radix = #ttg.generic_linear<{register = [[1]], lane = [], warp = [], block = [], inputRadices = {register = [1], lane = [], warp = [], block = []}, outDimSizes = [3]}>

// -----

// CHECK: error: Value for 'outDimSizes' must fit int32 and exceed 0
#bad_extent = #ttg.generic_linear<{register = [[1]], lane = [], warp = [], block = [], inputRadices = {register = [3], lane = [], warp = [], block = []}, outDimSizes = [4294967297 : i64]}>

// -----

// CHECK: error: Unknown input dimension 'registre' in inputRadices
#unknown_radix_key = #ttg.generic_linear<{register = [[1]], lane = [], warp = [], block = [], inputRadices = {register = [2], registre = [3], lane = [], warp = [], block = []}, outDimSizes = [2]}>

// -----

// CHECK: error: Expected inputRadices dictionary
#radices_not_dict = #ttg.generic_linear<{register = [[1]], lane = [], warp = [], block = [], inputRadices = [3], outDimSizes = [3]}>

// -----

// CHECK: error: Expected integer list for 'register'
#radix_not_list = #ttg.generic_linear<{register = [[1]], lane = [], warp = [], block = [], inputRadices = {register = 3, lane = [], warp = [], block = []}, outDimSizes = [3]}>

// -----

// CHECK: error: mixed-radix layouts require outDimSizes metadata
#no_out_sizes = #ttg.generic_linear<{register = [[1]], lane = [], warp = [], block = [], inputRadices = {register = [3], lane = [], warp = [], block = []}}>

// -----

// CHECK: error: Mixed-radix output metadata must match layout rank 1
#rank_mismatch = #ttg.generic_linear<{register = [[1]], lane = [], warp = [], block = [], inputRadices = {register = [3], lane = [], warp = [], block = []}, outDimSizes = [3, 1]}>

// -----

// Only linear and generic_linear text reads metadata.
// CHECK: error: NPOT layout metadata is not yet supported on this encoding
#wmma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]], outDimSizes = [2, 2]}, instrShape = [16, 16, 32]}>
module attributes {test.attr = #wmma} {}

// -----

// A radix-2 digit with an odd Product part is rejected at construction.
// CHECK: error: invalid radix-2 digit: it has an odd Product contribution
#non_h = #ttg.generic_linear<{
  register = [[3], [2]], lane = [], warp = [], block = [],
  inputRadices = {register = [2, 3], lane = [], warp = [], block = []},
  outDimSizes = [6]
}>
