#!/bin/bash

accelerate launch --config_file accelerate_configs/train_launch_big.yaml --num_processes 4 --gradient_accumulation_steps 1 src/origins/main.py --config-path ../../experiments/learn_gemma3_12b_train --config-name default
