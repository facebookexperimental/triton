# pyre-strict

"""
MPP WebApp for Blackwell Flash Attention Backward Kernel.

This script demonstrates how to use the MppWebApp utility class to create
a web application for analyzing the Blackwell Flash Attention backward kernel.
"""

import importlib.util
import logging
import sys

import torch
import triton

def alloc_fn(size: int, align: int, _):
    return torch.empty(size, dtype=torch.int8, device="cuda")

triton.set_allocator(alloc_fn)

# Add the path to triton-mpp if needed
sys.path.insert(0, "/data/users/srir/fbsource/fbcode/triton/tools/triton-mpp")

from frontend.web.mpp_webapp import MppWebApp

# Import the backward kernel function from the module with dashes in the name
module_path = "/home/srir/triton/third_party/tlx/tutorials/blackwell-fa-ws-pipelined-persistent_test.py"
spec = importlib.util.spec_from_file_location("blackwell_fa_test", module_path)
blackwell_module = importlib.util.module_from_spec(spec)
sys.modules["blackwell_fa_test"] = blackwell_module  # Register in sys.modules for pickling
spec.loader.exec_module(blackwell_module)
_original_run_attention_backward = blackwell_module.run_attention_backward


# Wrapper that ensures allocator is set right before kernel execution
# This is needed because ProtonTraceExecutionTask forces recompilation with TRITON_ALWAYS_COMPILE=1
# and the allocator must be available during that recompilation
def run_attention_backward(*args, **kwargs):
    """Wrapper that ensures the Triton allocator is set before kernel compilation/execution."""
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    
    # Set allocator right before kernel execution to ensure it's available
    # for both initial compilation and Proton-triggered recompilation
    triton.set_allocator(alloc_fn)
    return _original_run_attention_backward(*args, **kwargs)


def main() -> None:
    """Main function to start the Triton-MPP web application."""
    # CRITICAL: Set up Triton allocator FIRST, before any other operations
    # This is required for TMA descriptors and must be available during
    # kernel compilation (including Proton-triggered recompilation)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create the web app with attention backward kernel
    # You can adjust these parameters to test different configurations
    webapp = MppWebApp(
        run_attention_backward,
        Z=8,           # Batch size
        H=16,          # Number of heads
        N_CTX=1024,    # Context length
        HEAD_DIM=128,  # Head dimension
        mode="bwd",    # Backward mode
        provider="triton-fp16",
    )

    # Start the web application
    # This will:
    # 1. Run initial analysis with the provided kernel
    # 2. Create a Flask app with routes for code viewing and application running
    # 3. Start the web server
    webapp.start(host="0.0.0.0", port=8091, debug=False)


if __name__ == "__main__":
    main()
