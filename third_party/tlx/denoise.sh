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
    DEFAULT_POWER=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    MAX_SM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits  -i "$CUDA_VISIBLE_DEVICES")

    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | awk '{print $2}')

    if [[ -z "${DESIRED_POWER:-}" ]]; then
        if [[ "$GPU_MODEL" == "H100" ]]; then
            DESIRED_POWER=700
        elif [[ "$GPU_MODEL" == "GB200" ]]; then
            DESIRED_POWER=1200
        elif [[ "$GPU_MODEL" == "GB300" ]]; then
            DESIRED_POWER=1400
        elif [[ "$GPU_MODEL" == "B200" ]]; then
            DESIRED_POWER=750
        else
            # Unknown GPU: use the card's rated power (power.default_limit) so the
            # locked max clock runs at the power it is designed to sustain, rather
            # than an arbitrary 500 W that power-throttles large parts (a GB300,
            # for example, reports default_limit = max = 1400 W). Fall back to
            # 500 W only if the query did not return a number.
            if [[ "$DEFAULT_POWER" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                DESIRED_POWER="$DEFAULT_POWER"
            else
                DESIRED_POWER=500
            fi
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
    # PyTorch on ROCm accepts either visibility variable, but rocprof rejects
    # conflicting CUDA/HIP masks. Keep the caller's spelling and use a separate
    # physical index for rocm-smi instead of exporting the other variable.
    if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
        AMD_DEVICE="$HIP_VISIBLE_DEVICES"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        AMD_DEVICE="$CUDA_VISIBLE_DEVICES"
    else
        export HIP_VISIBLE_DEVICES=4
        AMD_DEVICE="$HIP_VISIBLE_DEVICES"
    fi

    GPU_INFO=$(rocm-smi -d "$AMD_DEVICE" --showproductname 2>/dev/null)
    # "Card Series" is often just "AMD Radeon Graphics" on CDNA data-center parts (e.g. MI350),
    # so identify by the PCI device id ("Card Model") and GFX version, which are reliable; keep
    # Series only for display. (MI350/MI355 both report gfx950, so match device id before gfx.)
    DEVICE_ID=$(printf  '%s\n' "$GPU_INFO" | awk -F: '/Card Model/  {print $NF; exit}' | xargs)
    GFX_VER=$(printf    '%s\n' "$GPU_INFO" | awk -F: '/GFX Version/ {print $NF; exit}' | xargs)
    GPU_SERIES=$(printf '%s\n' "$GPU_INFO" | awk -F: '/Card Series/ {print $NF; exit}' | xargs)

    case "$DEVICE_ID:$GFX_VER:$GPU_SERIES" in
        *0x74a0*|*0x74a1*|*MI300*|*gfx942*) GPU_NAME="MI300X"; : "${DESIRED_POWER:=750}"  ;;
        *0x75a1*|*0x75a3*|*MI355*)          GPU_NAME="MI355X"; : "${DESIRED_POWER:=1400}" ;;
        *0x75a0*|*MI350*|*gfx950*)          GPU_NAME="MI350X"; : "${DESIRED_POWER:=1000}" ;;
        *) GPU_NAME="AMD GPU (${GPU_SERIES:-$DEVICE_ID $GFX_VER})"; : "${DESIRED_POWER:=500}" ;;
    esac

    # sclk to pin via perf-determinism (overridable). CDNA4/MI350 does NOT support
    # `--setperflevel high` (rocm-smi returns "Not supported on the given system"),
    # so that silently did nothing — use --setperfdeterminism, which pins the GFX clock.
    DETERMINISM_CLK="${DETERMINISM_CLK:=2100}"

    echo "Detected $GPU_NAME"
    echo "Locking GPU $AMD_DEVICE sclk to ${DETERMINISM_CLK} MHz (perf-determinism) + power cap ${DESIRED_POWER} W"

    # Lock GPU clocks via perf-determinism and apply power overdrive (both best-effort under sudo)
    (
        sudo rocm-smi -d "$AMD_DEVICE" --setperfdeterminism "$DETERMINISM_CLK"
        sudo rocm-smi -d "$AMD_DEVICE" --setpoweroverdrive "$DESIRED_POWER"
    ) >/dev/null
fi

NUMA_NODE=""
if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    PCI_BUS_ID=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=pci.bus_id --format=csv,noheader | tr '[:upper:]' '[:lower:]')
    # nvidia-smi uses an eight-digit PCI domain; sysfs uses four digits.
    PCI_BUS_ID=$(printf '%s' "$PCI_BUS_ID" | sed -E 's/^[0-9a-f]{4}([0-9a-f]{4}:)/\1/')
    NUMA_NODE_FILE="/sys/bus/pci/devices/$PCI_BUS_ID/numa_node"
    if [[ -r "$NUMA_NODE_FILE" ]]; then
        NUMA_NODE=$(<"$NUMA_NODE_FILE")
    fi
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    # Via the PCI bus id, like the nvidia branch above -- NOT via
    # /sys/class/drm/card$AMD_DEVICE. The rocm-smi device index and the
    # DRM card index are different numbering schemes that only coincide at 0: on
    # an 8-GPU MI300X box rocm-smi's card6 is 0000:c8:00.0, which sysfs
    # enumerates as card40. The old form found no numa_node and silently ran
    # unbound for every device but the first.
    AMD_PCI_BUS_ID=$(rocm-smi -d "$AMD_DEVICE" --showbus --csv 2>/dev/null \
        | awk -F, -v c="card${AMD_DEVICE}" 'tolower($1)==c {print tolower($2)}' | tr -d '[:space:]')
    NUMA_NODE_FILE="/sys/bus/pci/devices/$AMD_PCI_BUS_ID/numa_node"
    if [[ -n "$AMD_PCI_BUS_ID" && -r "$NUMA_NODE_FILE" ]]; then
        NUMA_NODE=$(<"$NUMA_NODE_FILE")
    fi
fi

if [[ "$NUMA_NODE" =~ ^[0-9]+$ ]]; then
    echo "Binding CPU and memory to NUMA node $NUMA_NODE"
    numactl --membind="$NUMA_NODE" --cpunodebind="$NUMA_NODE" "$@"
    CHILD_STATUS=$?
else
    echo "Warning: Could not determine a valid GPU-local NUMA node; running without NUMA binding" >&2
    "$@"
    CHILD_STATUS=$?
fi

# Unlock GPU clock
if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    (
        sudo nvidia-smi -rgc -i "$CUDA_VISIBLE_DEVICES"
        sudo nvidia-smi --power-limit="$CURRENT_POWER" -i "$CUDA_VISIBLE_DEVICES"
    ) >/dev/null
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    (
        sudo rocm-smi -d "$AMD_DEVICE" --resetperfdeterminism
        sudo rocm-smi -d "$AMD_DEVICE" --resetpoweroverdrive
    ) >/dev/null
fi

exit "$CHILD_STATUS"
