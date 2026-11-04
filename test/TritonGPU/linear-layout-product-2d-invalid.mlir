// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics

// A rank-2 Product layout that is not a permutation after removing broadcast
// bases gets an error instead of aborting in reshapeOuts.
// expected-error@+1 {{LinearEncodingAttr requires a permutation matrix layout after removing broadcast bases}}
#duplicate_lane = #ttg.linear<{register = [[1, 0]], lane = [[0, 1], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [3, 32]}>
module attributes {test.layout = #duplicate_lane} {
}

// -----

// Invertible, but a radix-2 digit sets two output bits.
// expected-error@+1 {{LinearEncodingAttr requires a permutation matrix layout after removing broadcast bases}}
#two_bits = #ttg.linear<{register = [[2, 0], [1, 1]], lane = [[0, 1]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2], warp = [], block = []}, outDimSizes = [6, 2]}>
module attributes {test.layout = #two_bits} {
}
