# --- EXAMPLE: Acquisition experiments ---
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/EXAMPLE_learn_gemma3_4b.yaml \
    --train_num_gpus 4 \
    --infer_num_gpus 4 \
    --train_accelerate_config train_launch

# --- EXAMPLE: Update experiments ---
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/EXAMPLE_update_gemma3_4b.yaml \
    --train_num_gpus 4 \
    --infer_num_gpus 4 \
    --train_accelerate_config train_launch

# --- EXAMPLE: Forget experiments ---
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/EXAMPLE_forget_gemma3_4b.yaml \
    --train_num_gpus 4 \
    --infer_num_gpus 4 \
    --train_accelerate_config train_launch


# --- Acquisition experiments ---
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/learn_gemma3_4b.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/learn_llama32_3b.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/learn_msft4_4b.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/learn_qwen3_4b.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/learn_gemma3_12b.yaml \
    --train_accelerate_config train_launch_big

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/learn_llama32_11b.yaml \
    --train_accelerate_config train_launch_big

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/learn_msft4_14b.yaml \
    --train_accelerate_config train_launch_big

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/learn_qwen3_14b.yaml \
    --train_accelerate_config train_launch_big

# --- Update experiments ---
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/update_gemma3_4b.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/update_llama32_3b.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/update_msft4_4b.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/update_qwen3_4b.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/update_gemma3_12b.yaml \
    --train_accelerate_config train_launch_big

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/update_llama32_11b.yaml \
    --train_accelerate_config train_launch_big

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/update_msft4_14b.yaml \
    --train_accelerate_config train_launch_big

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/update_qwen3_14b.yaml \
    --train_accelerate_config train_launch_big

# --- Forget experiments ---
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_gemma3_4b_epoch0.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_gemma3_4b_epoch3.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_gemma3_4b_epoch6.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_gemma3_4b_epoch12.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_llama32_3b_epoch0.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_llama32_3b_epoch3.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_llama32_3b_epoch6.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_llama32_3b_epoch12.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_msft4_4b_epoch0.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_msft4_4b_epoch3.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_msft4_4b_epoch6.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_msft4_4b_epoch12.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_qwen3_4b_epoch0.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_qwen3_4b_epoch3.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_qwen3_4b_epoch6.yaml
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_qwen3_4b_epoch12.yaml

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_gemma3_12b_epoch0.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_gemma3_12b_epoch3.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_gemma3_12b_epoch6.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_gemma3_12b_epoch12.yaml \
    --train_accelerate_config train_launch_big

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_llama32_11b_epoch0.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_llama32_11b_epoch3.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_llama32_11b_epoch6.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_llama32_11b_epoch12.yaml \
    --train_accelerate_config train_launch_big

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_msft4_14b_epoch0.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_msft4_14b_epoch3.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_msft4_14b_epoch6.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_msft4_14b_epoch12.yaml \
    --train_accelerate_config train_launch_big

python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_qwen3_14b_epoch0.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_qwen3_14b_epoch3.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_qwen3_14b_epoch6.yaml \
    --train_accelerate_config train_launch_big
python3 src/origins/prepare_sweeps/prepare_experiments.py \
    --sweep_file_path src/origins/configs/experiments/forget_qwen3_14b_epoch12.yaml \
    --train_accelerate_config train_launch_big
