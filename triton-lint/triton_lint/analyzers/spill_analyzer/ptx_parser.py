# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
PTX Parser for extracting spill information.

Parses PTX assembly to identify register spills and debug information.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LocalMemAlloc:
    """Represents a local memory allocation (spill storage)."""

    size: int  # Size in bytes
    align: int  # Alignment in bytes
    name: str  # Variable name (e.g., "__local_depot")


@dataclass
class LocDirective:
    """Represents a .loc directive in PTX."""

    file_id: int  # File ID from .file directives
    line: int  # Line number
    column: int  # Column number


@dataclass
class SpillInstruction:
    """Represents a single spill instruction in PTX."""

    ptx_line: int  # Line number in PTX file
    type: str  # 'store' or 'load'
    size: int  # Size in bytes
    data_type: str  # PTX data type (e.g., 'u32', 'f32')
    register: str  # Register name (e.g., '%r3')
    address: str  # Memory address expression (e.g., '[%r2]')
    loc: Optional[LocDirective] = None  # Source location


@dataclass
class PTXSpillInfo:
    """Complete spill information extracted from PTX."""

    local_memory: List[LocalMemAlloc]
    spills: List[SpillInstruction]
    file_map: Dict[int, str]  # Maps file_id -> filename
    total_local_bytes: int


class PTXParser:
    """Parses PTX assembly to extract spill information."""

    # PTX instruction patterns
    LOCAL_MEM_PATTERN = r"\.local\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]"
    FILE_DIRECTIVE_PATTERN = r'\.file\s+(\d+)\s+"([^"]+)"'
    LOC_DIRECTIVE_PATTERN = r"\.loc\s+(\d+)\s+(\d+)\s+(\d+)"
    SPILL_STORE_PATTERN = r"st\.local\.(\w+)\s+\[([^\]]+)\],\s*(%[\w]+)"
    SPILL_LOAD_PATTERN = r"ld\.local\.(\w+)\s+(%[\w]+),\s*\[([^\]]+)\]"

    def __init__(self):
        self.file_map: Dict[int, str] = {}
        self.local_memory: List[LocalMemAlloc] = []
        self.spills: List[SpillInstruction] = []
        self.current_loc: Optional[LocDirective] = None

    def parse(self, ptx_content: str) -> PTXSpillInfo:
        """
        Parse PTX assembly and extract spill information.

        Args:
            ptx_content: PTX assembly code as a string

        Returns:
            PTXSpillInfo containing all spill-related data
        """
        self._reset()

        lines = ptx_content.split("\n")

        # First pass: Build file map
        for line in lines:
            self._parse_file_directive(line)

        # Second pass: Parse spills and local memory
        for line_no, line in enumerate(lines, 1):
            # Track current source location
            loc = self._parse_loc_directive(line)
            if loc:
                self.current_loc = loc

            # Parse local memory allocations
            local_mem = self._parse_local_memory(line)
            if local_mem:
                self.local_memory.append(local_mem)

            # Parse spill instructions
            spill = self._parse_spill_instruction(line, line_no)
            if spill:
                self.spills.append(spill)

        total_bytes = sum(mem.size for mem in self.local_memory)

        return PTXSpillInfo(
            local_memory=self.local_memory,
            spills=self.spills,
            file_map=self.file_map,
            total_local_bytes=total_bytes,
        )

    def parse_file(self, ptx_file: Path) -> PTXSpillInfo:
        """
        Parse a PTX file.

        Args:
            ptx_file: Path to PTX file

        Returns:
            PTXSpillInfo containing all spill-related data
        """
        return self.parse(ptx_file.read_text())

    def _reset(self):
        """Reset parser state for new input."""
        self.file_map = {}
        self.local_memory = []
        self.spills = []
        self.current_loc = None

    def _parse_file_directive(self, line: str):
        """
        Parse .file directive to build filename map.

        Example: .file 1 "/path/to/kernel.py"
        """
        match = re.search(self.FILE_DIRECTIVE_PATTERN, line)
        if match:
            file_id = int(match.group(1))
            filename = match.group(2)
            self.file_map[file_id] = filename

    def _parse_loc_directive(self, line: str) -> Optional[LocDirective]:
        """
        Parse .loc directive.

        Example: .loc 1 42 5
        Returns: LocDirective(file_id=1, line=42, column=5)
        """
        match = re.search(self.LOC_DIRECTIVE_PATTERN, line)
        if match:
            return LocDirective(
                file_id=int(match.group(1)),
                line=int(match.group(2)),
                column=int(match.group(3)),
            )
        return None

    def _parse_local_memory(self, line: str) -> Optional[LocalMemAlloc]:
        """
        Parse local memory allocation.

        Example: .local .align 16 .b8 __local_depot[1024]
        Returns: LocalMemAlloc(size=1024, align=16, name="__local_depot")
        """
        match = re.search(self.LOCAL_MEM_PATTERN, line)
        if match:
            return LocalMemAlloc(
                align=int(match.group(1)),
                name=match.group(2),
                size=int(match.group(3)),
            )
        return None

    def _parse_spill_instruction(self, line: str, line_no: int) -> Optional[SpillInstruction]:
        """
        Parse spill store or load instruction.

        Examples:
          st.local.u32 [%r2], %r3
          ld.local.f32 %f1, [%r2+4]
        """
        # Try to parse as store
        match = re.search(self.SPILL_STORE_PATTERN, line)
        if match:
            data_type = match.group(1)
            address = match.group(2)
            register = match.group(3)
            return SpillInstruction(
                ptx_line=line_no,
                type="store",
                size=self._get_type_size(data_type),
                data_type=data_type,
                register=register,
                address=f"[{address}]",
                loc=self.current_loc,
            )

        # Try to parse as load
        match = re.search(self.SPILL_LOAD_PATTERN, line)
        if match:
            data_type = match.group(1)
            register = match.group(2)
            address = match.group(3)
            return SpillInstruction(
                ptx_line=line_no,
                type="load",
                size=self._get_type_size(data_type),
                data_type=data_type,
                register=register,
                address=f"[{address}]",
                loc=self.current_loc,
            )

        return None

    def _get_type_size(self, type_suffix: str) -> int:
        """
        Get size in bytes from PTX type suffix.

        Examples:
          u32 -> 4 bytes
          f64 -> 8 bytes
        """
        size_map = {
            "u8": 1,
            "s8": 1,
            "b8": 1,
            "u16": 2,
            "s16": 2,
            "b16": 2,
            "f16": 2,
            "u32": 4,
            "s32": 4,
            "b32": 4,
            "f32": 4,
            "u64": 8,
            "s64": 8,
            "b64": 8,
            "f64": 8,
        }
        return size_map.get(type_suffix, 4)  # Default to 4 bytes
