#!/bin/bash

accelerate launch --config_file accelerate_configs/train_launch.yaml --num_processes 4 --gradient_accumulation_steps 1 src/origins/main.py --config-path ../../experiments/forget_llama32_3b_epoch6_train --config-name default && python3 src/origins/main_vllm.py --config-path ../../experiments/forget_llama32_3b_epoch6_infer --config-name default || exit $?
