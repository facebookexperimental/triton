// RUN: triton-opt %s --split-input-file --nvgpu-test-ws-channel-cycle-analysis | FileCheck %s

// A zero-credit recurrence is unsafe. The witness is stable in edge insertion
// order and its total iteration distance is zero.
// CHECK-LABEL: module @zero_credit
// CHECK-SAME: nvws.test.protocol_cycle_distance = 0 : i64
// CHECK-SAME: nvws.test.protocol_cycle_edges = array<i64: 0, 1, 2>
// CHECK-SAME: nvws.test.protocol_status = "unsafe"
module @zero_credit attributes {
  "nvws.test.protocol_event_count" = 3 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 1, 1, 7, 1, 2, 0, 8, 2, 0, -1, 9>
} {
}

// -----

// Increasing one recurrence edge by one iteration provides positive credit.
// CHECK-LABEL: module @positive_credit
// CHECK-SAME: nvws.test.protocol_status = "safe"
// CHECK-NOT: nvws.test.protocol_cycle_edges
module @positive_credit attributes {
  "nvws.test.protocol_event_count" = 3 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 1, 1, 7, 1, 2, 0, 8, 2, 0, 0, 9>
} {
}

// -----

// A negative edge is harmless when it is not part of a recurrence.
// CHECK-LABEL: module @acyclic
// CHECK-SAME: nvws.test.protocol_status = "safe"
module @acyclic attributes {
  "nvws.test.protocol_event_count" = 3 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 1, -4, 7, 1, 2, 0, 8>
} {
}

// -----

// An edge outside the declared event set is unsupported, not guessed safe.
// CHECK-LABEL: module @bad_endpoint
// CHECK-SAME: nvws.test.protocol_reason = "protocol edge references an unknown event"
// CHECK-SAME: nvws.test.protocol_status = "unsupported"
module @bad_endpoint attributes {
  "nvws.test.protocol_event_count" = 2 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 2, 0, 7>
} {
}

// -----

// An unsupported SCC must not hide a proven unsafe SCC elsewhere.
// CHECK-LABEL: module @unsafe_over_unsupported
// CHECK-SAME: nvws.test.protocol_cycle_distance = 0 : i64
// CHECK-SAME: nvws.test.protocol_cycle_edges = array<i64: 1>
// CHECK-SAME: nvws.test.protocol_status = "unsafe"
module @unsafe_over_unsupported attributes {
  "nvws.test.protocol_event_count" = 2 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 0, 9223372036854775807, 7, 1, 1, 0, 8>
} {
}
