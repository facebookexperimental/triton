# pyre-strict

"""
SM Occupancy Analysis for Blackwell Flash Attention Backward Kernel.

This script demonstrates how to run SM occupancy analysis directly using
Triton-MPP's plot_sm_occupancy application without the web interface.
"""

import logging

from mpp.applications.sm_occupancy import plot_sm_occupancy
from mpp.container.kernel import KernelLaunchDescriptor

from blackwell_fa_ws_pipelined_persistent_test import run_attention_backward


def main() -> None:
    """Main function to run SM occupancy analysis."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting SM occupancy analysis for Blackwell Flash Attention backward kernel")
    
    # Create kernel launch descriptor with the desired configuration
    kernel_descriptor = KernelLaunchDescriptor(
        run_attention_backward,
        Z=8,           # Batch size
        H=16,          # Number of heads
        N_CTX=1024,    # Context length
        HEAD_DIM=128,  # Head dimension
        mode="bwd",    # Backward mode
        provider="triton-fp16",
    )
    
    # Run SM occupancy analysis and save the heatmap
    output_file = "attention_backward_sm_occupancy.png"
    logger.info(f"Running SM occupancy analysis, output will be saved to: {output_file}")
    
    plot_sm_occupancy(kernel_descriptor, output_file)
    
    logger.info(f"✅ SM occupancy heatmap saved to: {output_file}")
    logger.info("The heatmap shows the number of active CTAs on each SM over time")


if __name__ == "__main__":
    main()
