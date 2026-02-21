"""Tests for AST analyzer."""

import pytest

from triton_lint.analyzers.ast_analyzer import ASTAnalyzer
from triton_lint.core.config import Config
from triton_lint.core.finding import Severity


def test_missing_autotune():
    """Test detection of missing @triton.autotune decorator."""
    code = """
import triton

@triton.jit
def kernel(x_ptr, BLOCK_SIZE: int):
    pass
"""

    config = Config()
    analyzer = ASTAnalyzer(config)
    findings = analyzer.analyze(code, "test.py")

    # Should find missing autotune
    autotune_findings = [f for f in findings if f.rule_id == "missing-autotune"]
    assert len(autotune_findings) == 1
    assert autotune_findings[0].severity == Severity.WARNING


def test_hardcoded_block_size():
    """Test detection of hardcoded block sizes."""
    code = """
import triton

@triton.jit
def kernel(x_ptr):
    BLOCK_SIZE = 128
    pass
"""

    config = Config()
    analyzer = ASTAnalyzer(config)
    findings = analyzer.analyze(code, "test.py")

    # Should find hardcoded block size
    block_findings = [f for f in findings if f.rule_id == "hardcoded-block-size"]
    assert len(block_findings) == 1
    assert block_findings[0].severity == Severity.WARNING


def test_missing_mask():
    """Test detection of missing masks in memory operations."""
    code = """
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr):
    offs = tl.arange(0, 128)
    ptrs = x_ptr + offs
    x = tl.load(ptrs)  # Missing mask!
"""

    config = Config()
    analyzer = ASTAnalyzer(config)
    findings = analyzer.analyze(code, "test.py")

    # Should find missing mask
    mask_findings = [f for f in findings if f.rule_id == "missing-mask"]
    assert len(mask_findings) >= 1
    assert mask_findings[0].severity == Severity.WARNING


def test_with_autotune_no_warning():
    """Test that kernels with @triton.autotune don't trigger warnings."""
    code = """
import triton

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 128})])
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: int):
    pass
"""

    config = Config()
    analyzer = ASTAnalyzer(config)
    findings = analyzer.analyze(code, "test.py")

    # Should not find missing autotune
    autotune_findings = [f for f in findings if f.rule_id == "missing-autotune"]
    assert len(autotune_findings) == 0


def test_disabled_rule():
    """Test that disabled rules don't produce findings."""
    code = """
import triton

@triton.jit
def kernel(x_ptr, BLOCK_SIZE: int):
    pass
"""

    config = Config()
    config.disabled_rules.add("missing-autotune")

    analyzer = ASTAnalyzer(config)
    findings = analyzer.analyze(code, "test.py")

    # Should not find missing autotune since it's disabled
    autotune_findings = [f for f in findings if f.rule_id == "missing-autotune"]
    assert len(autotune_findings) == 0


def test_syntax_error():
    """Test handling of syntax errors."""
    code = """
import triton

@triton.jit
def kernel(x_ptr):
    if True
        pass
"""

    config = Config()
    analyzer = ASTAnalyzer(config)
    findings = analyzer.analyze(code, "test.py")

    # Should find syntax error
    syntax_errors = [f for f in findings if f.rule_id == "syntax-error"]
    assert len(syntax_errors) == 1
    assert syntax_errors[0].severity == Severity.ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
