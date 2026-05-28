#!/bin/bash

accelerate launch --config_file accelerate_configs/train_launch_big.yaml --num_processes 4 --gradient_accumulation_steps 1 src/origins/main.py --config-path ../../experiments/learn_llama32_11b_train --config-name default && python3 src/origins/main_vllm.py --config-path ../../experiments/learn_llama32_11b_infer --config-name default || exit $?
