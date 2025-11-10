# pyre-strict

"""
MPP WebApp for Blackwell Flash Attention Backward Kernel.

This script demonstrates how to use the MppWebApp utility class to create
a web application for analyzing the Blackwell Flash Attention backward kernel.
"""

import logging

from frontend.web.mpp_webapp import MppWebApp

from blackwell_fa_ws_pipelined_persistent_test import run_attention_backward


def main() -> None:
    """Main function to start the Triton-MPP web application."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    webapp = MppWebApp(
        run_attention_backward,
        Z=8,           # Batch size
        H=16,          # Number of heads
        N_CTX=1024,    # Context length
        HEAD_DIM=128,  # Head dimension
        mode="bwd",    # Backward mode
        provider="triton-fp16",
    )
    webapp.start(host="0.0.0.0", port=8091, debug=False)


if __name__ == "__main__":
    main()
