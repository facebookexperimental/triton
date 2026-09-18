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

// A strictly-negative recurrence cannot close in a one-sided finite domain:
// its walk eventually reaches the prologue boundary. The ordinary steady-state
// solver intentionally remains conservative; this mode is used only by the
// nested-loop boundary validator.
// CHECK-LABEL: module @negative_boundary_recurrence
// CHECK-SAME: nvws.test.protocol_status = "safe"
module @negative_boundary_recurrence attributes {
  "nvws.test.protocol_boundary_aware",
  "nvws.test.protocol_event_count" = 2 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 1, 0, 7, 1, 0, -1, 8>
} {
}

// -----

// A zero-distance recurrence remains unsafe at a dynamic boundary.
// CHECK-LABEL: module @zero_boundary_recurrence
// CHECK-SAME: nvws.test.protocol_cycle_distance = 0 : i64
// CHECK-SAME: nvws.test.protocol_status = "unsafe"
module @zero_boundary_recurrence attributes {
  "nvws.test.protocol_boundary_aware",
  "nvws.test.protocol_event_count" = 2 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 1, 1, 7, 1, 0, -1, 8>
} {
}

// -----

// Opposite-sign simple cycles remain independent boundary recurrences. The
// negative one terminates at the prologue and the positive one carries credit;
// concatenating them through event 0 does not create one simultaneous circular
// wait because that walk repeats event 0.
// CHECK-LABEL: module @mixed_boundary_recurrence
// CHECK-SAME: nvws.test.protocol_status = "safe"
module @mixed_boundary_recurrence attributes {
  "nvws.test.protocol_boundary_aware",
  "nvws.test.protocol_event_count" = 2 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 1, -1, 7, 1, 0, 0, 8, 0, 0, 1, 9>
} {
}

// -----

// A mixed-sign SCC can still contain a simple zero-distance circular wait.
// The negative 0->1->0 recurrence and positive self-loop are insufficient on
// their own, but 0->1->2->0 is an actual zero-credit cycle and must reject.
// CHECK-LABEL: module @mixed_with_zero_boundary_recurrence
// CHECK-SAME: nvws.test.protocol_cycle_distance = 0 : i64
// CHECK-SAME: nvws.test.protocol_status = "unsafe"
module @mixed_with_zero_boundary_recurrence attributes {
  "nvws.test.protocol_boundary_aware",
  "nvws.test.protocol_event_count" = 3 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 1, -1, 7, 1, 0, 0, 8, 0, 0, 1, 9, 1, 2, 1, 10, 2, 0, 0, 11>
} {
}

// -----

// A positive slot-reuse edge carries one initially empty physical slot. Even
// when schedule distance cancels to zero, that marked cycle can start and then
// returns the slot credit on every transaction.
// CHECK-LABEL: module @slot_credit_boundary_recurrence
// CHECK-SAME: nvws.test.protocol_status = "safe"
module @slot_credit_boundary_recurrence attributes {
  "nvws.test.protocol_boundary_aware",
  "nvws.test.protocol_event_count" = 2 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 1, -1, 7, 1, 0, 1, 8>,
  "nvws.test.protocol_edge_kinds" = array<i64: 0, 2>
} {
}

// -----

// Two backward crossings compose two finite-boundary recurrences rather than
// one minimal circular wait. Each recurrence terminates at the prologue; the
// balancing forward edges do not make both boundaries active simultaneously.
// CHECK-LABEL: module @two_boundary_crossings
// CHECK-SAME: nvws.test.protocol_status = "safe"
module @two_boundary_crossings attributes {
  "nvws.test.protocol_boundary_aware",
  "nvws.test.protocol_event_count" = 4 : i64,
  "nvws.test.protocol_edges" = array<i64: 0, 1, -1, 7, 1, 2, 1, 8, 2, 3, -1, 9, 3, 0, 1, 10>
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
