#!/bin/bash

accelerate launch --config_file accelerate_configs/train_launch.yaml --num_processes 4 --gradient_accumulation_steps 1 src/origins/main.py --config-path ../../experiments/forget_qwen3_4b_epoch0_train --config-name default && python3 src/origins/main_vllm.py --config-path ../../experiments/forget_qwen3_4b_epoch0_infer --config-name default || exit $?
