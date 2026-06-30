#!/bin/bash

# There's a whole presentation about stable benchmarking here:
# https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9956-best-practices-when-benchmarking-cuda-applications_V2.pdf

# Detect GPU vendor
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_VENDOR="nvidia"
elif command -v rocm-smi &> /dev/null && rocm-smi &> /dev/null; then
    GPU_VENDOR="amd"
else
    echo "Error: No supported GPU found (neither nvidia-smi nor rocm-smi available)"
    exit 1
fi

if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=4}"

    CURRENT_POWER=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    MAX_POWER=$(nvidia-smi --query-gpu=power.max_limit  --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    MAX_SM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits  -i "$CUDA_VISIBLE_DEVICES")

    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | awk '{print $2}')

    if [[ -z "${DESIRED_POWER:-}" ]]; then
        if [[ "$GPU_MODEL" == "H100" ]]; then
            DESIRED_POWER=700
        elif [[ "$GPU_MODEL" == "GB200" ]]; then
            DESIRED_POWER=1200
        elif [[ "$GPU_MODEL" == "B200" ]]; then
            DESIRED_POWER=750
        else
            DESIRED_POWER=500
        fi
    fi

    # Compute the minimum of desired and max power
    POWER_CAP=$(awk -v d="$DESIRED_POWER" -v m="$MAX_POWER" 'BEGIN {print (d < m ? d : m)}')

    echo "Locking GPU $CUDA_VISIBLE_DEVICES power cap to $POWER_CAP W"
    echo "Locking GPU $CUDA_VISIBLE_DEVICES frequency cap to $MAX_SM_CLOCK Hz"

    # Lock GPU clocks
    (
        sudo nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -pm 1                # persistent mode
        sudo nvidia-smi --power-limit="$POWER_CAP" -i "$CUDA_VISIBLE_DEVICES"
        sudo nvidia-smi -lgc "$MAX_SM_CLOCK" -i "$CUDA_VISIBLE_DEVICES"
    ) >/dev/null

elif [[ "$GPU_VENDOR" == "amd" ]]; then
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:=4}"

    GPU_INFO=$(rocm-smi -d "$HIP_VISIBLE_DEVICES" --showproductname 2>/dev/null)
    GPU_MODEL=$(printf '%s\n' "$GPU_INFO" | awk -F: '/Card Series/ {print $NF; exit}' | xargs)
    if [[ -z "$GPU_MODEL" || "$GPU_MODEL" == "N/A" ]]; then
        GPU_MODEL=$(printf '%s\n' "$GPU_INFO" | awk -F: '/Card Model/ {print $NF; exit}' | xargs)
    fi
    if [[ -z "$GPU_MODEL" || "$GPU_MODEL" == "N/A" ]]; then
        GPU_MODEL=$(amd-smi static --asic -g "$HIP_VISIBLE_DEVICES" 2>/dev/null \
            | grep -i "market_name\|model" | head -1 | awk -F: '{print $NF}' | xargs)
    fi

    # Map model to GPU name and default power
    case "$GPU_MODEL" in
        *MI300*|0x74a0|0x74a1)
            GPU_NAME="MI300X"
            [[ -z "${DESIRED_POWER:-}" ]] && DESIRED_POWER=750
            ;;
        *MI350*|0x75a0)
            GPU_NAME="MI350X"
            [[ -z "${DESIRED_POWER:-}" ]] && DESIRED_POWER=1000
            ;;
        *MI355*|0x75a1|0x75a3)
            GPU_NAME="MI355X"
            [[ -z "${DESIRED_POWER:-}" ]] && DESIRED_POWER=1400
            ;;
        *)
            GPU_NAME="AMD GPU ($GPU_MODEL)"
            [[ -z "${DESIRED_POWER:-}" ]] && DESIRED_POWER=500
            ;;
    esac

    echo "Detected $GPU_NAME"
    echo "Locking GPU $HIP_VISIBLE_DEVICES power cap to ${DESIRED_POWER} W"
    echo "Setting GPU $HIP_VISIBLE_DEVICES to high performance mode"

    # Lock GPU clocks by setting performance level to high and applying power overdrive
    (
        sudo rocm-smi -d "$HIP_VISIBLE_DEVICES" --setperflevel high
        sudo rocm-smi -d "$HIP_VISIBLE_DEVICES" --setpoweroverdrive "$DESIRED_POWER"
    ) >/dev/null
fi

# TODO: Automate NUMA node detection. On one devgpu, device 6 is attached to
# NUMA node 3. This is how to discover that mapping:
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
if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    (
        sudo nvidia-smi -rgc -i "$CUDA_VISIBLE_DEVICES"
        sudo nvidia-smi --power-limit="$CURRENT_POWER" -i "$CUDA_VISIBLE_DEVICES"
    ) >/dev/null
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    (
        sudo rocm-smi -d "$HIP_VISIBLE_DEVICES" --resetclocks
        sudo rocm-smi -d "$HIP_VISIBLE_DEVICES" --resetpoweroverdrive
    ) >/dev/null
fi
