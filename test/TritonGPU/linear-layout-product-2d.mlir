// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s | env TRITON_ALLOW_NPOT=1 triton-opt | FileCheck %s

// Product layouts verify without flattening outputs: each digit owns one output
// coordinate (a radix-2 digit one bit, an odd digit Z/m with a unit coefficient).
// No flag-off leg: flag off ignores the keys and the bases-only parse aborts.

// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[1, 0], [0, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [3, 64]}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [64, 3]}>
// CHECK{LITERAL}: #linear2 = #ttg.linear<{register = [[3, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [], block = [], inputRadices = {register = [5], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [5, 64]}>
// CHECK{LITERAL}: #linear3 = #ttg.linear<{register = [[192]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [5], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [320]}>
// CHECK{LITERAL}: #linear4 = #ttg.linear<{register = [[1, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [3, 64]}>
// CHECK: module attributes {test.broadcast = #linear, test.cols = #linear1, test.odd_unit = #linear2, test.odd_unit_1d = #linear3, test.rows = #linear4}
#rows = #ttg.linear<{register = [[1, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [3, 64]}>
#cols = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [64, 3]}>
#broadcast = #ttg.linear<{register = [[1, 0], [0, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [3, 64]}>
#odd_unit = #ttg.linear<{register = [[3, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [], block = [], inputRadices = {register = [5], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [5, 64]}>
#odd_unit_1d = #ttg.linear<{register = [[192]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [5], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [320]}>

module attributes {test.broadcast = #broadcast, test.cols = #cols, test.odd_unit = #odd_unit, test.odd_unit_1d = #odd_unit_1d, test.rows = #rows} {
}
