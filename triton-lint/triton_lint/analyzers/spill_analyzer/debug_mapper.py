# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Debug Info Mapper for mapping PTX locations to Python source.

Maps spill locations in PTX back to the original Python source code
using debug information.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .ptx_parser import PTXSpillInfo, SpillInstruction


@dataclass
class SourceLocation:
    """Represents a source code location."""

    file: str  # Path to source file
    line: int  # Line number (1-indexed)
    column: int  # Column number (1-indexed)
    function: Optional[str] = None  # Function name if available


@dataclass
class MappedSpill:
    """A spill instruction mapped to its source location."""

    spill: SpillInstruction  # The spill instruction
    source_location: Optional[SourceLocation]  # Mapped source location


class DebugInfoMapper:
    """Maps PTX locations to Python source locations."""

    def __init__(self):
        pass

    def map_spills_to_source(self, ptx_info: PTXSpillInfo, llvm_ir_file: Optional[Path] = None) -> List[MappedSpill]:
        """
        Map spills from PTX to source locations.

        Args:
            ptx_info: Parsed PTX spill information
            llvm_ir_file: Optional LLVM IR file for enhanced mapping

        Returns:
            List of spills with source location mapping
        """
        mapped_spills = []

        for spill in ptx_info.spills:
            source_loc = self._map_loc_to_source(spill, ptx_info.file_map)

            # TODO: Phase 2 - Enhance with LLVM IR debug metadata
            # if llvm_ir_file and source_loc:
            #     source_loc = self._refine_with_llvm_metadata(
            #         source_loc, llvm_ir_file
            #     )

            mapped_spills.append(MappedSpill(spill=spill, source_location=source_loc))

        return mapped_spills

    def _map_loc_to_source(self, spill: SpillInstruction, file_map: dict[int, str]) -> Optional[SourceLocation]:
        """
        Map a spill instruction's .loc directive to source location.

        Args:
            spill: Spill instruction with .loc info
            file_map: Mapping of file IDs to filenames

        Returns:
            Source location if available, None otherwise
        """
        if not spill.loc:
            return None

        # Get filename from file map
        filename = file_map.get(spill.loc.file_id)
        if not filename:
            return None

        # Filter out synthesized files
        if "<" in filename and ">" in filename:
            return None

        return SourceLocation(file=filename, line=spill.loc.line, column=spill.loc.column)

    def _refine_with_llvm_metadata(self, approx_loc: SourceLocation, llvm_ir_file: Path) -> SourceLocation:
        """
        Refine source location using LLVM IR debug metadata.

        This is more accurate than PTX .loc alone as it preserves
        more detailed debug information.

        Args:
            approx_loc: Approximate location from PTX
            llvm_ir_file: Path to LLVM IR file

        Returns:
            Refined source location

        Note: Implementation deferred to Phase 2
        """
        # TODO: Phase 2 implementation
        # Parse LLVM IR debug metadata:
        # - Find !DILocation nodes
        # - Match with PTX locations
        # - Extract function names from !DISubprogram
        return approx_loc
