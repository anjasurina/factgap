#!/bin/bash

accelerate launch --config_file accelerate_configs/train_launch_big.yaml --num_processes 4 --gradient_accumulation_steps 4 src/origins/main.py --config-path ../../experiments/update_msft4_14b_train --config-name default && python3 src/origins/main_vllm.py --config-path ../../experiments/update_msft4_14b_infer --config-name default || exit $?
