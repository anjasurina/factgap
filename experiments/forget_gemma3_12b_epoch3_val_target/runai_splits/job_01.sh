#!/bin/bash

accelerate launch --config_file accelerate_configs/train_launch.yaml --num_processes 4 --gradient_accumulation_steps 1 src/origins/main.py --config-path ../../experiments/forget_gemma3_12b_epoch3_val_target --config-name default
