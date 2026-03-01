# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Spill Aggregator for analyzing spill patterns.

Groups spills by location, identifies hotspots, and analyzes patterns
to provide actionable suggestions.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .debug_mapper import MappedSpill, SourceLocation


@dataclass
class SpillReport:
    """Aggregated report of register spills."""

    total_spills: int  # Total number of spill instructions
    total_bytes: int  # Total bytes spilled
    by_location: Dict[Tuple[str, int], List[MappedSpill]]  # Grouped by (file, line)
    hotspots: List[Tuple[SourceLocation, List[MappedSpill]]]  # Top 5 worst locations
    patterns: Dict[str, Any]  # Detected spill patterns
    stores: int  # Number of stores
    loads: int  # Number of loads


class SpillAggregator:
    """Aggregates and analyzes spill patterns."""

    def __init__(self):
        pass

    def aggregate(self, mapped_spills: List[MappedSpill]) -> SpillReport:
        """
        Aggregate spills and generate statistics.

        Args:
            mapped_spills: List of spills with source mapping

        Returns:
            Complete spill report with analysis
        """
        # Group by source location
        by_location = self._group_by_location(mapped_spills)

        # Identify hotspots (top 5 worst locations)
        hotspots = self._find_hotspots(by_location)

        # Analyze patterns
        patterns = self._analyze_patterns(mapped_spills)

        # Calculate totals
        total_spills = len(mapped_spills)
        total_bytes = sum(s.spill.size for s in mapped_spills)
        stores = sum(1 for s in mapped_spills if s.spill.type == "store")
        loads = sum(1 for s in mapped_spills if s.spill.type == "load")

        return SpillReport(
            total_spills=total_spills,
            total_bytes=total_bytes,
            by_location=by_location,
            hotspots=hotspots,
            patterns=patterns,
            stores=stores,
            loads=loads,
        )

    def _group_by_location(self, spills: List[MappedSpill]) -> Dict[Tuple[str, int], List[MappedSpill]]:
        """
        Group spills by (file, line) for reporting.

        Args:
            spills: List of mapped spills

        Returns:
            Dictionary mapping (file, line) to list of spills at that location
        """
        grouped: Dict[Tuple[str, int], List[MappedSpill]] = defaultdict(list)

        for spill in spills:
            if spill.source_location:
                key = (spill.source_location.file, spill.source_location.line)
                grouped[key].append(spill)
            else:
                # Spills without source mapping
                grouped[("<unknown>", 0)].append(spill)

        return dict(grouped)

    def _find_hotspots(
            self, by_location: Dict[Tuple[str, int],
                                    List[MappedSpill]]) -> List[Tuple[SourceLocation, List[MappedSpill]]]:
        """
        Identify the top 5 locations with most spills.

        Args:
            by_location: Spills grouped by location

        Returns:
            List of (location, spills) tuples sorted by spill count
        """
        hotspots = []

        for (file, line), spill_list in by_location.items():
            if file != "<unknown>":
                loc = SourceLocation(file=file, line=line, column=0)
                hotspots.append((loc, spill_list))

        # Sort by number of spills (descending)
        hotspots.sort(key=lambda x: len(x[1]), reverse=True)

        # Return top 5
        return hotspots[:5]

    def _analyze_patterns(self, spills: List[MappedSpill]) -> Dict[str, Any]:
        """
        Detect common spill patterns.

        Patterns:
        - store_heavy: Many stores, few loads → too many live values
        - load_heavy: Many loads, few stores → reuse but bad placement
        - balanced: stores ≈ loads → temporary spills in loops

        Args:
            spills: List of mapped spills

        Returns:
            Dictionary describing detected patterns
        """
        if not spills:
            return {
                "pattern": "none",
                "reason": "No spills detected",
                "stores": 0,
                "loads": 0,
            }

        stores = sum(1 for s in spills if s.spill.type == "store")
        loads = sum(1 for s in spills if s.spill.type == "load")

        # Determine pattern
        if stores > loads * 2:
            pattern = "store_heavy"
            reason = "Too many concurrent live values"
        elif loads > stores * 2:
            pattern = "load_heavy"
            reason = "Values reused but spilled"
        else:
            pattern = "balanced"
            reason = "Temporary spills in loops"

        return {
            "pattern": pattern,
            "reason": reason,
            "stores": stores,
            "loads": loads,
        }

    def summarize_spills(self, spills: List[MappedSpill]) -> str:
        """
        Create a human-readable summary of spills.

        Args:
            spills: List of spills to summarize

        Returns:
            Summary string
        """
        stores = sum(1 for s in spills if s.spill.type == "store")
        loads = sum(1 for s in spills if s.spill.type == "load")
        total_bytes = sum(s.spill.size for s in spills)

        return f"{stores} stores, {loads} loads, {total_bytes} bytes"
