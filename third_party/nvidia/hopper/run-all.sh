echo "Perf"
TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1 CUDA_VISIBLE_DEVICES=1 ~/fbsource/fbcode/triton/scripts/denoise.sh python python/tutorials/fused-attention-ws-device-tma.py
TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1 CUDA_VISIBLE_DEVICES=1 ~/fbsource/fbcode/triton/scripts/denoise.sh python python/tutorials/fused-attention-ws.py

echo "Correctness"
TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1 pytest python/tutorials/fused-attention-ws-device-tma.py
TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1 pytest python/tutorials/fused-attention-ws.py

# under tritonbench
# TRITON_ALWAYS_COMPILE=1 TRITON_PRINT_AUTOTUNING=1 TRITON_USE_META_WS=1 CUDA_VISIBLE_DEVICES=1 python run.py --op blackwell_attentions --seq-len 8192 --batch 4 --n-heads 32 --d-head 128 --rep 3000 --sleep 1.0 --metrics tflops --simple-output --only triton_tutorial_flash_persistent_blackwell,tlx_blackwell_ws_pipelined_persistent --force --bwd
