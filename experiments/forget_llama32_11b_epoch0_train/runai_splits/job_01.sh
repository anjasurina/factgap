#!/bin/bash

accelerate launch --config_file accelerate_configs/train_launch_big.yaml --num_processes 4 --gradient_accumulation_steps 1 src/origins/main.py --config-path ../../experiments/forget_llama32_11b_epoch0_train --config-name default
