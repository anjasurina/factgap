#!/bin/bash

accelerate launch --config_file accelerate_configs/train_launch_big.yaml --num_processes 4 --gradient_accumulation_steps 4 src/origins/main.py --config-path ../../experiments/forget_msft4_14b_epoch3_train --config-name default
