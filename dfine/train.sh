#!/bin/bash

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config_file> <resume_path> [num_gpus] [port]"
    exit 1
fi

CONFIG_FILE=$1
RESUME_PATH=$2
GPUS=${3:-1} # Default to 1 GPU if not provided
PORT=${4:-7777} # Default to port 7777 if not provided

torchrun --master_port=$PORT --nproc_per_node=$GPUS train.py \
    -c "$CONFIG_FILE" \
    -t "$RESUME_PATH" \
    --use-amp \
    --seed=0
