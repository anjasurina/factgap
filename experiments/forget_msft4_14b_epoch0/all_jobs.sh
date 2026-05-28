#!/bin/bash

accelerate launch --config_file accelerate_configs/train_launch_big.yaml --num_processes 4 --gradient_accumulation_steps 4 src/origins/main.py --config-path ../../experiments/forget_msft4_14b_epoch0_train --config-name default && python3 src/origins/main_vllm.py --config-path ../../experiments/forget_msft4_14b_epoch0_infer --config-name default || exit $?
