#!/bin/bash

# There's a whole presentation about stable benchmarking here:
# https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9956-best-practices-when-benchmarking-cuda-applications_V2.pdf

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=4}"

CURRENT_POWER=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES)
MAX_POWER=$(nvidia-smi --query-gpu=power.max_limit  --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES)
MAX_SM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits  -i $CUDA_VISIBLE_DEVICES)

GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | awk '{print $2}')

if [[ "$GPU_MODEL" == "H100" ]]; then
    DESIRED_POWER=500
elif [[ "$GPU_MODEL" == "GB200" ]]; then
    DESIRED_POWER=1200
elif [[ "$GPU_MODEL" == "B200" ]]; then
    DESIRED_POWER=750
else
    DESIRED_POWER=500
fi

# Compute the minimum of desired and max power
POWER_CAP=$(awk -v d="$DESIRED_POWER" -v m="$MAX_POWER" 'BEGIN {print (d < m ? d : m)}')

echo "Locking GPU $CUDA_VISIBLE_DEVICES power cap to $POWER_CAP W"
echo "Locking GPU $CUDA_VISIBLE_DEVICES frequency cap to $MAX_SM_CLOCK Hz"

# 1335, 1980
# Lock GPU clocks
(
    sudo nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -pm 1                # persistent mode
    sudo nvidia-smi --power-limit=$POWER_CAP -i "$CUDA_VISIBLE_DEVICES"
    sudo nvidia-smi -lgc $MAX_SM_CLOCK -i "$CUDA_VISIBLE_DEVICES"
) >/dev/null

# TODO: On my devgpu, device 6 is apparently attached to NUMA node 3.  How did
# I discover this?
#
# `nvidia-smi -i 6 -pm 1` prints the PCI bus ID (00000000:C6:00.0)
#
# You can also get this from `nvidia-smi -x -q` and looking for minor_number
# and pci_bus_id
#
# Then, `cat /sys/bus/pci/devices/0000:c6:00.0/numa_node` prints 3
# is it always the case that device N is on numa node N/2? :shrug:
#
# Maybe automate this process or figure out if it always holds?
#
# ... Or you can just `nvidia-smi topo -mp` and it will just print out exactly
# what you want, like this:

#       GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  mlx5_1  mlx5_2  mlx5_3  CPU Affinity    NUMA Affinity
# GPU0   X      PXB     SYS     SYS     SYS     SYS     SYS     SYS     NODE    SYS     SYS     SYS     0-23,96-119     0
# GPU6  SYS     SYS     SYS     SYS     SYS     SYS      X      PXB     SYS     SYS     SYS     NODE    72-95,168-191   3

numactl -m 0 -c 0 "$@"

# Unlock GPU clock
(
    sudo nvidia-smi -rgc -i "$CUDA_VISIBLE_DEVICES"
    sudo nvidia-smi --power-limit=$CURRENT_POWER -i "$CUDA_VISIBLE_DEVICES"
) >/dev/null
