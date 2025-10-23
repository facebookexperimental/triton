"""
Rules for detecting anti-patterns.

This package contains rule definitions that can be used across different
analyzers. Rules are organized by category (memory, block_size, etc.)
"""

# Rule registry for future extensibility
RULE_REGISTRY = {
    # Memory-related rules
    "scalar-memory-access": {
        "category": "memory",
        "severity": "error",
        "description": "Scalar memory accesses hurt performance and correctness",
    },
    "missing-mask": {
        "category": "memory",
        "severity": "warning",
        "description": "Memory operations should use masking for out-of-bounds safety",
    },

    # Block size rules
    "hardcoded-block-size": {
        "category": "tuning",
        "severity": "warning",
        "description": "Hardcoded block sizes should be autotuned",
    },
    "missing-autotune": {
        "category": "tuning",
        "severity": "warning",
        "description": "Kernels with tunable parameters should use @triton.autotune",
    },

    # Control flow rules
    "complex-loop-structure": {
        "category": "control_flow",
        "severity": "warning",
        "description": "Complex nested loops can hurt compilation and performance",
    },
}


def get_rule_info(rule_id: str) -> dict:
    """Get information about a specific rule."""
    return RULE_REGISTRY.get(rule_id,
                             {"category": "unknown", "severity": "warning", "description": "No description available"})


def list_rules() -> dict:
    """List all available rules."""
    return RULE_REGISTRY
