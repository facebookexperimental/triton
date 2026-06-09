#!/bin/bash

# Script to kill all GPU processes owned by the current user (NVIDIA and AMD)

CURRENT_USER=$(whoami)
echo "Current user: $CURRENT_USER"

# Detect GPU vendor
get_gpu_pids() {
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
  elif command -v rocm-smi &>/dev/null; then
    rocm-smi --showpids 2>/dev/null | awk 'NR>1 && /^[0-9]/ {print $1}'
  else
    echo "No GPU tools found (nvidia-smi or rocm-smi)." >&2
    exit 1
  fi
}

echo "Fetching GPU processes..."
GPU_PIDS=$(get_gpu_pids)

if [ -z "$GPU_PIDS" ]; then
  echo "No GPU processes found."
  exit 0
fi

# Check if any processes belong to current user
HAS_USER_PROCESSES=false
for PID in $GPU_PIDS; do
  PROCESS_USER=$(ps -o user= -p $PID 2>/dev/null)
  if [ "$PROCESS_USER" = "$CURRENT_USER" ]; then
    HAS_USER_PROCESSES=true
    break
  fi
done

if [ "$HAS_USER_PROCESSES" = false ]; then
  echo "No GPU processes found belonging to $CURRENT_USER."
  exit 0
fi

# Count processes
PROCESS_COUNT=$(echo "$GPU_PIDS" | wc -l)
echo "Found $PROCESS_COUNT GPU processes. Checking ownership..."

# Kill each process that belongs to current user
for PID in $GPU_PIDS; do
  PROCESS_USER=$(ps -o user= -p $PID 2>/dev/null)
  if [ "$PROCESS_USER" = "$CURRENT_USER" ]; then
    PROCESS_NAME=$(ps -p $PID -o comm= 2>/dev/null)
    echo "Killing process $PID ($PROCESS_NAME) owned by $PROCESS_USER..."
    # Avoid -9 so processes can shut down cleanly.
    kill $PID
    if [ $? -eq 0 ]; then
      echo "Process $PID terminated successfully."
    else
      echo "Failed to terminate process $PID."
    fi
  else
    echo "Skipping process $PID (owned by $PROCESS_USER)..."
  fi
done

echo "All user's GPU processes have been terminated."
echo "Sleeping for 2 seconds to verify..."
sleep 2

# Verify all user's processes are gone
REMAINING=$(get_gpu_pids)
if [ -z "$REMAINING" ]; then
  echo "Verification complete: No GPU processes remaining."
else
  echo "Remaining GPU processes:"
  CURRENT_USER_REMAINING=false
  for PID in $REMAINING; do
    PROCESS_USER=$(ps -o user= -p $PID 2>/dev/null)
    PROCESS_NAME=$(ps -p $PID -o comm= 2>/dev/null)
    echo "PID: $PID, User: $PROCESS_USER, Process: $PROCESS_NAME"
    if [ "$PROCESS_USER" = "$CURRENT_USER" ]; then
      CURRENT_USER_REMAINING=true
    fi
  done

  if [ "$CURRENT_USER_REMAINING" = true ]; then
    echo "WARNING: There are still GPU processes owned by $CURRENT_USER running!"
    echo "You might need to use 'kill -9' to force terminate these processes."
  fi
fi
