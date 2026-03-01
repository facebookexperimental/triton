# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Tests for the spill analyzer."""

import unittest
from pathlib import Path

from triton_lint.analyzers.spill_analyzer import (
    DebugInfoMapper,
    PTXParser,
    SpillAggregator,
    SpillAnalyzer,
)
from triton_lint.core.config import Config
from triton_lint.core.finding import Severity


class TestPTXParser(unittest.TestCase):
    """Test PTX parsing functionality."""

    def test_parse_local_memory(self):
        """Test detection of local memory allocation."""
        ptx = """
        .local .align 16 .b8 __local_depot[1024];
        """
        parser = PTXParser()
        info = parser.parse(ptx)

        self.assertEqual(len(info.local_memory), 1)
        self.assertEqual(info.local_memory[0].size, 1024)
        self.assertEqual(info.local_memory[0].align, 16)
        self.assertEqual(info.local_memory[0].name, "__local_depot")
        self.assertEqual(info.total_local_bytes, 1024)

    def test_parse_file_directive(self):
        """Test parsing .file directives."""
        ptx = """
        .file 1 "/path/to/kernel.py"
        .file 2 "<synthesized>"
        """
        parser = PTXParser()
        info = parser.parse(ptx)

        self.assertEqual(len(info.file_map), 2)
        self.assertEqual(info.file_map[1], "/path/to/kernel.py")
        self.assertEqual(info.file_map[2], "<synthesized>")

    def test_parse_spill_store(self):
        """Test detection of spill store instruction."""
        ptx = """
        .file 1 "kernel.py"
        .loc 1 42 5
        st.local.u32 [%r2], %r3;
        """
        parser = PTXParser()
        info = parser.parse(ptx)

        self.assertEqual(len(info.spills), 1)
        spill = info.spills[0]
        self.assertEqual(spill.type, "store")
        self.assertEqual(spill.data_type, "u32")
        self.assertEqual(spill.size, 4)
        self.assertEqual(spill.register, "%r3")
        self.assertEqual(spill.address, "[%r2]")
        self.assertIsNotNone(spill.loc)
        self.assertEqual(spill.loc.line, 42)

    def test_parse_spill_load(self):
        """Test detection of spill load instruction."""
        ptx = """
        .file 1 "kernel.py"
        .loc 1 43 8
        ld.local.f32 %f1, [%r2];
        """
        parser = PTXParser()
        info = parser.parse(ptx)

        self.assertEqual(len(info.spills), 1)
        spill = info.spills[0]
        self.assertEqual(spill.type, "load")
        self.assertEqual(spill.data_type, "f32")
        self.assertEqual(spill.size, 4)
        self.assertEqual(spill.register, "%f1")
        self.assertIsNotNone(spill.loc)
        self.assertEqual(spill.loc.line, 43)

    def test_parse_multiple_spills(self):
        """Test parsing multiple spill instructions."""
        ptx = """
        .file 1 "kernel.py"
        .local .align 16 .b8 __local_depot[512];

        .loc 1 42 5
        st.local.u32 [%r2], %r3;
        st.local.u32 [%r2+4], %r4;

        .loc 1 43 8
        ld.local.u32 %r5, [%r2];
        ld.local.u32 %r6, [%r2+4];
        """
        parser = PTXParser()
        info = parser.parse(ptx)

        self.assertEqual(len(info.spills), 4)
        self.assertEqual(info.total_local_bytes, 512)
        self.assertEqual(sum(1 for s in info.spills if s.type == "store"), 2)
        self.assertEqual(sum(1 for s in info.spills if s.type == "load"), 2)

    def test_no_spills(self):
        """Test PTX with no spills."""
        ptx = """
        .file 1 "kernel.py"
        .visible .entry my_kernel() {
            mov.u32 %r1, 0;
            ret;
        }
        """
        parser = PTXParser()
        info = parser.parse(ptx)

        self.assertEqual(len(info.spills), 0)
        self.assertEqual(info.total_local_bytes, 0)


class TestDebugInfoMapper(unittest.TestCase):
    """Test debug info mapping functionality."""

    def test_map_spills_to_source(self):
        """Test mapping spills to source locations."""
        ptx = """
        .file 1 "/path/to/kernel.py"
        .loc 1 42 5
        st.local.u32 [%r2], %r3;
        """
        parser = PTXParser()
        mapper = DebugInfoMapper()

        ptx_info = parser.parse(ptx)
        mapped = mapper.map_spills_to_source(ptx_info)

        self.assertEqual(len(mapped), 1)
        self.assertIsNotNone(mapped[0].source_location)
        self.assertEqual(mapped[0].source_location.file, "/path/to/kernel.py")
        self.assertEqual(mapped[0].source_location.line, 42)
        self.assertEqual(mapped[0].source_location.column, 5)

    def test_filter_synthesized_files(self):
        """Test that synthesized files are filtered out."""
        ptx = """
        .file 1 "<synthesized>"
        .loc 1 1 1
        st.local.u32 [%r2], %r3;
        """
        parser = PTXParser()
        mapper = DebugInfoMapper()

        ptx_info = parser.parse(ptx)
        mapped = mapper.map_spills_to_source(ptx_info)

        self.assertEqual(len(mapped), 1)
        self.assertIsNone(mapped[0].source_location)


class TestSpillAggregator(unittest.TestCase):
    """Test spill aggregation and analysis."""

    def test_aggregate_empty(self):
        """Test aggregation with no spills."""
        aggregator = SpillAggregator()
        report = aggregator.aggregate([])

        self.assertEqual(report.total_spills, 0)
        self.assertEqual(report.total_bytes, 0)
        self.assertEqual(report.patterns["pattern"], "none")

    def test_detect_store_heavy_pattern(self):
        """Test detection of store-heavy pattern."""
        from triton_lint.analyzers.spill_analyzer.debug_mapper import MappedSpill
        from triton_lint.analyzers.spill_analyzer.ptx_parser import SpillInstruction

        # Create 10 stores, 2 loads
        spills = []
        for i in range(10):
            spills.append(
                MappedSpill(
                    spill=SpillInstruction(
                        ptx_line=i,
                        type="store",
                        size=4,
                        data_type="u32",
                        register=f"%r{i}",
                        address="[%r2]",
                    ),
                    source_location=None,
                ))
        for i in range(2):
            spills.append(
                MappedSpill(
                    spill=SpillInstruction(
                        ptx_line=i + 10,
                        type="load",
                        size=4,
                        data_type="u32",
                        register=f"%r{i+10}",
                        address="[%r2]",
                    ),
                    source_location=None,
                ))

        aggregator = SpillAggregator()
        report = aggregator.aggregate(spills)

        self.assertEqual(report.patterns["pattern"], "store_heavy")
        self.assertEqual(report.stores, 10)
        self.assertEqual(report.loads, 2)

    def test_group_by_location(self):
        """Test grouping spills by source location."""
        from triton_lint.analyzers.spill_analyzer.debug_mapper import (
            MappedSpill,
            SourceLocation,
        )
        from triton_lint.analyzers.spill_analyzer.ptx_parser import SpillInstruction

        # Create spills at different locations
        spills = [
            MappedSpill(
                spill=SpillInstruction(
                    ptx_line=1,
                    type="store",
                    size=4,
                    data_type="u32",
                    register="%r1",
                    address="[%r2]",
                ),
                source_location=SourceLocation(file="kernel.py", line=42, column=5),
            ),
            MappedSpill(
                spill=SpillInstruction(
                    ptx_line=2,
                    type="store",
                    size=4,
                    data_type="u32",
                    register="%r2",
                    address="[%r2]",
                ),
                source_location=SourceLocation(file="kernel.py", line=42, column=5),
            ),
            MappedSpill(
                spill=SpillInstruction(
                    ptx_line=3,
                    type="load",
                    size=4,
                    data_type="u32",
                    register="%r3",
                    address="[%r2]",
                ),
                source_location=SourceLocation(file="kernel.py", line=43, column=8),
            ),
        ]

        aggregator = SpillAggregator()
        report = aggregator.aggregate(spills)

        # Should have 2 unique locations
        self.assertEqual(len(report.by_location), 2)
        self.assertEqual(len(report.by_location[("kernel.py", 42)]), 2)
        self.assertEqual(len(report.by_location[("kernel.py", 43)]), 1)


class TestSpillAnalyzer(unittest.TestCase):
    """Test the main SpillAnalyzer."""

    def test_no_spills(self):
        """Test analysis of PTX with no spills."""
        ptx = """
        .file 1 "kernel.py"
        .visible .entry my_kernel() {
            mov.u32 %r1, 0;
            ret;
        }
        """
        analyzer = SpillAnalyzer()
        findings = analyzer.analyze(ptx, "test.ptx")

        self.assertEqual(len(findings), 0)

    def test_small_spills(self):
        """Test analysis of PTX with small spills (INFO severity)."""
        ptx = """
        .file 1 "kernel.py"
        .local .align 16 .b8 __local_depot[128];
        .loc 1 42 5
        st.local.u32 [%r2], %r3;
        """
        analyzer = SpillAnalyzer()
        findings = analyzer.analyze(ptx, "test.ptx")

        # Should have overall finding + per-location finding
        self.assertGreaterEqual(len(findings), 1)
        # Check overall finding
        overall = [f for f in findings if f.rule_id == "SPILL001"][0]
        self.assertEqual(overall.severity, Severity.INFO)
        self.assertIn("128 bytes", overall.message)

    def test_large_spills(self):
        """Test analysis of PTX with large spills (ERROR severity)."""
        ptx = """
        .file 1 "kernel.py"
        .local .align 16 .b8 __local_depot[2048];
        .loc 1 42 5
        st.local.u32 [%r2], %r3;
        """
        analyzer = SpillAnalyzer()
        findings = analyzer.analyze(ptx, "test.ptx")

        # Should have overall finding + per-location finding
        self.assertGreaterEqual(len(findings), 1)
        # Check overall finding
        overall = [f for f in findings if f.rule_id == "SPILL001"][0]
        self.assertEqual(overall.severity, Severity.ERROR)
        self.assertIn("2048 bytes", overall.message)

    def test_spills_with_source_mapping(self):
        """Test that spills are mapped to source locations."""
        ptx = """
        .file 1 "/path/to/kernel.py"
        .local .align 16 .b8 __local_depot[512];

        .loc 1 42 5
        st.local.u32 [%r2], %r3;
        st.local.u32 [%r2+4], %r4;
        st.local.u32 [%r2+8], %r5;
        """
        analyzer = SpillAnalyzer()
        findings = analyzer.analyze(ptx, "test.ptx")

        # Should have overall finding + per-location finding
        self.assertGreaterEqual(len(findings), 1)

        # Check for per-location finding
        location_findings = [f for f in findings if f.rule_id == "SPILL002"]
        if location_findings:
            self.assertEqual(location_findings[0].filename, "/path/to/kernel.py")
            self.assertEqual(location_findings[0].line, 42)


if __name__ == "__main__":
    unittest.main()
