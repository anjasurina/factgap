"""
Main training script.

Handles LEARN, UPDATE, FORGET, and VALIDATION_ONLY phases.
"""
import os
from datetime import datetime as dt
import gc
import json
from typing import Optional

import torch
import torch._dynamo
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from torch.utils.data import DataLoader

from origins.utils.seed import seed_everything
from origins.custom_classes import (
    InferenceTask,
    Phase,
    PhaseConfig,
)
from origins.logging.print import AnsiColors, log_with_color
from origins.utils.mp import detect_accelerate
from origins.train import (
    create_train_dataset,
    initialize_trainer,
    save_model,
    load_training_data,
    load_control_text,
    load_validation_data,
)
from origins.inference import load_inference_tasks
from origins.utils.experiments_tracker import ExperimentsTracker, ExperimentStatus
from origins.model.overrides import set_up_model_config
from origins.model.loading import load_pretrained_lm
from origins.utils.benchmarking import setup_benchmarking, benchmark_scope
from origins.utils.misc_utils import (
    is_global_main_process,
    sync_processes,
    verify_config,
)
from origins.model_callbacks import (
    LossCallback,
    SaveModelCallback,
    FormattedPrinterCallback,
)

logger = get_logger(__name__)

torch._dynamo.config.cache_size_limit = 128  # or -1 to disable limit

accelerator = Accelerator()


def load_training_data_according_to_phase(
    cfg: DictConfig,
    phase: Phase,
    inference_tasks: list[InferenceTask] | None
) -> Dataset:
    """
    Loads training data based on the current training phase.

    Args:
        cfg: Hydra configuration object.
        phase: Current training phase.
        inference_tasks: List of inference tasks.

    Returns:
        Dataset: Training dataset for the phase.
    """
    if phase in [Phase.LEARN, Phase.UPDATE]:
        assert inference_tasks is not None, "Inference tasks must be provided for LEARN and UPDATE phases."
        train_datapoints = load_training_data(
            cfg=cfg, inference_tasks=inference_tasks, phase=phase, accelerator=accelerator)
    elif phase == Phase.FORGET:
        train_datapoints = load_control_text(
            cfg=cfg, num_control_sentences=cfg.train.forget.num_forget_samples, accelerator=accelerator)
        if cfg.train.debug_mode:
            train_datapoints = train_datapoints[:cfg.train.debug_num_samples]
    elif phase == Phase.VALIDATION_ONLY:
        train_datapoints = []
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return create_train_dataset(cfg=cfg, train_datapoints=train_datapoints, accelerator=accelerator)


def prepare_phases(cfg: DictConfig) -> list[PhaseConfig]:
    """
    Prepares the list of training phases based on the configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        list[PhaseConfig]: List of phase configurations.
    """

    if cfg.infer.enable_inference:
        raise ValueError(
            "Inference is no longer supported in main.py. "
            "Use main_vllm.py for inference (including inference-only runs "
            "over checkpoint ranges)."
        )

    if cfg.train.validation.enable and not cfg.train.enable_training:
        if cfg.train.validation.checkpoint_path is None:
            raise ValueError(
                "When doing validation only, please use "
                "train.validation.checkpoint_params to specify the checkpoint path."
            )

        # If only doing inference on checkpoints, we need to load a different
        # checkpoint for each epoch in the specified range.
        base_checkpoint_path = cfg.train.validation.checkpoint_path
        start_epoch = cfg.train.validation.start_epoch
        end_epoch = cfg.train.validation.end_epoch
        validation_frequency = cfg.train.validation.frequency

        if start_epoch is None or end_epoch is None or validation_frequency < 1:
            raise ValueError(
                "Both start_epoch and end_epoch must be specified when doing "
                "validation only with checkpoint paths, "
                "and validation.frequency must be at least 1."
            )

        phases = []
        for epoch in range(start_epoch, end_epoch + 1, validation_frequency):
            if epoch == 0:
                checkpoint_path = cfg.model.name
                logger.warning(
                    f"[EPOCH 0] Loading model from {checkpoint_path}...")
            else:
                checkpoint_path = os.path.join(
                    base_checkpoint_path, f"model_after_epoch_{epoch}")
            phase_cfg = PhaseConfig(
                phase=Phase.VALIDATION_ONLY,
                starting_epoch=epoch,
                start_from_checkpoint=checkpoint_path,
                num_epochs=0,
                save_model_at_every_epoch=False,
                evaluate_on_train_begin=False,
                save_model_at_the_end=False,
            )
            phases.append(phase_cfg)
        return phases

    learn_phase_cfg = PhaseConfig(
        phase=Phase.LEARN,
        starting_epoch=0,
        start_from_checkpoint=cfg.model.name,
        num_epochs=cfg.train.num_train_epochs,
        save_model_at_every_epoch=cfg.train.save_model_at_every_epoch,
        evaluate_on_train_begin=True,
        save_model_at_the_end=cfg.train.save_model_after_learn_phase,
    )
    phases = [learn_phase_cfg]

    if cfg.train.upd.only_update_phase or cfg.train.upd.num_update_epochs > 0:
        update_phase_cfg = PhaseConfig(
            phase=Phase.UPDATE,
            starting_epoch=cfg.train.num_train_epochs,
            start_from_checkpoint=cfg.train.upd.checkpoint_to_load_for_update_training,
            num_epochs=cfg.train.upd.num_update_epochs,
            save_model_at_every_epoch=cfg.train.upd.save_model_at_every_epoch,
            evaluate_on_train_begin=True if cfg.train.upd.only_update_phase else False,
            save_model_at_the_end=cfg.train.upd.save_model_after_update_phase,
        )
        if cfg.train.upd.only_update_phase:
            return [update_phase_cfg]
        phases.append(update_phase_cfg)

    if cfg.train.forget.only_forget_phase or cfg.train.forget.num_forget_epochs > 0:
        forget_phase_cfg = PhaseConfig(
            phase=Phase.FORGET,
            starting_epoch=cfg.train.num_train_epochs + cfg.train.upd.num_update_epochs,
            start_from_checkpoint=cfg.train.forget.checkpoint_to_load_for_forget_training,
            num_epochs=cfg.train.forget.num_forget_epochs,
            save_model_at_every_epoch=cfg.train.forget.save_model_at_every_epoch,
            evaluate_on_train_begin=True if cfg.train.forget.only_forget_phase else False,
            save_model_at_the_end=cfg.train.forget.save_model_after_forget_phase,
        )
        if cfg.train.forget.only_forget_phase:
            return [forget_phase_cfg]
        phases.append(forget_phase_cfg)

    return phases


def run_validation_only_phase(
    cfg: DictConfig,
    phase_cfg: PhaseConfig,
    inference_tasks: Optional[list[InferenceTask]] = None,
) -> None:
    """
    Computes validation loss on a checkpoint without initializing the full Trainer/ZeRO-3 engine.
    """
    checkpoint_path = phase_cfg.start_from_checkpoint
    epoch = phase_cfg.starting_epoch

    log_with_color(
        f"Starting VALIDATION_ONLY phase for epoch {epoch}...",
        logger, color_code=AnsiColors.OKCYAN
    )

    # 1. Load Model Manually
    # We load to CPU first, then move to the correct device.
    # This avoids "device_map" index errors if accelerate masks GPUs.
    logger.info(f"Loading model from {checkpoint_path}...")
    model = load_pretrained_lm(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation,
        low_cpu_mem_usage=True,  # Speed up loading
    )

    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    using_zero3 = ds_plugin is not None and getattr(ds_plugin, "zero_stage", 0) == 3
    if using_zero3:
        # With ZeRO-3, params are sharded on creation - must prepare the model
        # so all ranks participate in the forward pass under ZeRO-3 semantics.
        model = accelerator.prepare(model)
    else:
        model.to(accelerator.device)
    model.eval()

    # 2. Reuse your existing data loading logic
    validation_datapoints, data_source = load_validation_data(
        cfg, inference_tasks=inference_tasks)
    raw_dataset = create_train_dataset(
        cfg, validation_datapoints, accelerator=accelerator)

    # 3. Tokenizer Setup
    # Best practice: Try loading tokenizer from checkpoint first, fallback to model name
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, use_fast=True)
    except:
        logger.warning(
            f"Could not load tokenizer from {checkpoint_path}, falling back to {cfg.model.name}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Loss calculation expects right padding

    # 4. Map & Tokenize
    def tokenize_function(examples):

        # If cfg.train.use_chat_template, chat template is already applied in create_train_dataset
        if "messages" in examples:
            texts = [
                tokenizer.apply_chat_template(
                    m,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False
                )
                for m in examples["messages"]
            ]
        else:
            texts = examples["text"]

        return tokenizer(
            text=texts,
            truncation=True,
            max_length=cfg.train.max_length,
            padding=False,  # Collator handles padding
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing validation set"
    )

    # 5. Prepare DataLoader with Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=cfg.train.validation.per_device_batch_size,
        collate_fn=data_collator,
        drop_last=False,
        shuffle=False
    )

    # 6. Prepare ONLY the DataLoader
    # CRITICAL: Do NOT prepare the model. This keeps it as a standard PyTorch model.
    dataloader = accelerator.prepare(dataloader)

    # 7. Evaluation Loop
    total_loss = 0.0
    total_steps = 0

    logger.info(
        f"Computing loss for epoch {epoch} on {len(tokenized_dataset)} samples...")

    # This measures wall time and GPU memory for the validation pass
    benchmark_metadata = {
        "epoch": epoch,
        "num_samples": len(tokenized_dataset),
        "phase": "validation"
    }

    with benchmark_scope("validation_forward_pass", benchmark_metadata):
        for batch in dataloader:
            with torch.no_grad():
                # inputs are already on device via prepared dataloader
                outputs = model(**batch)
                loss = outputs.loss

                # Detach and accumulate
                total_loss += loss.detach().float()
                total_steps += 1

    # 8. Aggregate results across GPUs
    # We calculate the mean loss across all batches on this device
    local_avg_loss = total_loss / \
        total_steps if total_steps > 0 else torch.tensor(
            0.0).to(accelerator.device)

    # Gather all local averages from all GPUs
    all_losses = accelerator.gather(local_avg_loss)

    # Compute global mean
    final_loss = all_losses.mean().item()

    # 9. Save to JSON
    if accelerator.is_main_process:
        output_dir = os.path.join(cfg.output.dir, "validation_metrics")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"loss_epoch_{epoch}.json")

        result_data = {
            "epoch": epoch,
            "loss": final_loss,
            "data_source": data_source,
            "checkpoint": checkpoint_path,
            "timestamp": str(dt.now())
        }

        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4)

        log_with_color(
            f"Validation saved: Epoch {epoch} | Loss: {final_loss:.4f} -> {file_path}",
            logger, color_code=AnsiColors.OKGREEN
        )

    # Cleanup
    del model
    del tokenizer
    del dataloader
    torch.cuda.empty_cache()


def run_phase(
    cfg: DictConfig,
    phase_cfg: PhaseConfig,
    inference_tasks: list[InferenceTask],
):
    """
    Executes a single training phase, handling all steps from data loading to
    model training, evaluation, and resource cleanup.

    This function orchestrates the workflow for a given phase
    (LEARN, UPDATE, FORGET, or VALIDATION_ONLY) as defined in the configuration.
    It performs the following steps:
      - Loads the appropriate training data for the phase.
      - Sets up the trainer with the specified model checkpoint, dataset, and callbacks.
      - Starts the training loop for the configured number of epochs.
      - Optionally saves the model at the end of the phase.
      - Frees up resources and memory after training is complete.

    Args:
        cfg: Hydra configuration object containing all experiment settings.
        phase_cfg: Configuration for the current phase, including type, epochs, and checkpoint info.
        inference_tasks: List of inference tasks used to construct training/validation data.

    Returns:
        None
    """

    # Starting phase
    phase = phase_cfg.phase
    log_with_color(
        f"Starting {phase.value} phase...",
        logger_instance=logger,
        color_code=AnsiColors.OKMAGENTA
    )

    setup_benchmarking(
        run_name=cfg.output.experiment_name or "run",
        phase=phase.value,
        is_main_process=is_global_main_process(accelerator)
    )

    # Initialize callbacks
    save_model_callback = SaveModelCallback(
        cfg=cfg,
        initial_epoch=phase_cfg.starting_epoch,
        save_model_at_every_epoch=phase_cfg.save_model_at_every_epoch,
        accelerator=accelerator
    )
    loss_callback = LossCallback(cfg=cfg)
    formatted_printer_callback = FormattedPrinterCallback(cfg=cfg)

    # --- Initialize SFT Trainer ---
    log_with_color(
        f"Starting from checkpoint: {phase_cfg.start_from_checkpoint}",
        logger_instance=logger, color_code=AnsiColors.OKBLUE
    )

    sync_processes(accelerator)

    if phase == Phase.VALIDATION_ONLY:
        # This avoids the overhead of initializing the Trainer when only doing validation
        run_validation_only_phase(
            cfg=cfg, phase_cfg=phase_cfg, inference_tasks=inference_tasks)
        return

    # Load training data
    train_dataset = load_training_data_according_to_phase(
        cfg=cfg, phase=phase, inference_tasks=inference_tasks
    )
    logger.info(
        f"Loaded {len(train_dataset)} samples for {phase.value} phase.")
    # Load validation data (if any)
    if cfg.train.validation.enable or phase == Phase.VALIDATION_ONLY:
        validation_datapoints, _ = load_validation_data(
            cfg=cfg, inference_tasks=inference_tasks)
        validation_dataset = create_train_dataset(
            cfg=cfg,
            train_datapoints=validation_datapoints,
            accelerator=accelerator
        )
    else:
        validation_dataset = None

    trainer = initialize_trainer(
        cfg=cfg,
        num_train_epochs=phase_cfg.num_epochs,
        model=phase_cfg.start_from_checkpoint,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        callbacks=[
            save_model_callback,
            loss_callback,
            formatted_printer_callback,
            ],
        accelerator=accelerator,
        phase=phase,
    )

    # Set to the trainer accelerator since we need this for saving models that
    # are wrapped in ZeRO-3.
    save_model_callback.accelerator = trainer.accelerator

    # After trainer initialization, before trainer.train()
    if hasattr(trainer, "scaler") and trainer.scaler is not None:
        logger.info(f"AMP GradScaler is active: {trainer.scaler}")
    else:
        logger.info("AMP GradScaler is NOT active.")

    # --- Start training ---
    log_with_color(
        f"Starting training for {phase_cfg.num_epochs} epochs of {phase.value} phase...",
        logger,
        color_code=AnsiColors.OKMAGENTA,
    )

    # 1. Sync all processes before starting train so logs are clean
    accelerator.wait_for_everyone()

    # 2. Check memory right before the big allocations
    allocated_mb = torch.cuda.memory_allocated() / (1024**2)
    reserved_mb = torch.cuda.memory_reserved() / (1024**2)
    logger.info(
        f"[Rank {accelerator.process_index}] VRAM Before train() -> Allocated: {allocated_mb:.2f} MB, Reserved: {reserved_mb:.2f} MB",
        main_process_only=False
    )

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        # 3. Catch the exact OOM and print the state
        failed_allocated = torch.cuda.memory_allocated() / (1024**2)
        logger.error(f"[Rank {accelerator.process_index}] OOM CAUGHT! Memory at crash: {failed_allocated:.2f} MB. Error: {e}", main_process_only=False)
        raise e

    log_with_color(f"Training for {phase.value} phase completed.",
                   logger, color_code=AnsiColors.OKMAGENTA)

    # Save the model after training
    sync_processes([accelerator, trainer.accelerator])
    if phase_cfg.save_model_at_the_end:
        with benchmark_scope("save_checkpoint"):
            save_model(
                cfg=cfg,
                model=trainer.model,
                folder_name=f"model_after_{phase.value}_phase",
                accelerator=trainer.accelerator
            )

    # Clean up, drop remaining references and free the CUDA cache
    sync_processes([accelerator, trainer.accelerator])
    accelerator.free_memory()
    trainer.accelerator.free_memory()
    del trainer.model
    del trainer.processing_class
    del trainer
    del train_dataset
    del validation_dataset
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running the experiment pipeline.

    This function performs the following:
        - Detects and configures distributed/accelerated training if needed.
        - Sets up model and experiment configuration.
        - Validates the configuration for errors or conflicts.
        - Initializes experiment tracking.
        - Loads inference tasks (used to construct training/validation data).
        - Prepares the list of experiment phases to run.
        - Iterates through each phase, running training or validation as appropriate.
        - Saves results and updates experiment status upon completion.

    Args:
        cfg: Hydra configuration object.

    Returns:
        None
    """

    # --- Initial setup ---
    # Detect if we are using accelerate to launch the main python script.
    detect_accelerate(cfg)

    # Override the config with the model params
    set_up_model_config(cfg)

    # Check the config for non-compatible combinations
    verify_config(cfg)

    # Experiment tracking setup
    experiment_tracker = None
    if cfg.experiments_tracker.enable:
        experiment_tracker = ExperimentsTracker(cfg)
        exists, status, should_skip, status_epoch = experiment_tracker.experiment_exists()
        if exists and should_skip:
            logger.info(
                f"Experiment already run (status: {status}, status_epoch: {status_epoch}), skipping.")
            return
        else:
            logger.info(
                f"Experiment exists: {exists}, status: {status}, should_skip: {should_skip}, status_epoch: {status_epoch}")
        if is_global_main_process(accelerator):
            experiment_tracker.update(
                status=ExperimentStatus.IN_PROGRESS, status_epoch=0)

    logger.info(
        f"Hello from process {accelerator.process_index}", main_process_only=False)
    seed_everything(cfg)

    # Log the config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}", main_process_only=True)

    # Inference tasks are still needed here because train/validation datasets
    # are derived from them (e.g., load_training_data, load_validation_data).
    inference_tasks = load_inference_tasks(cfg=cfg)
    log_with_color(
        f"Loaded {len(inference_tasks)} inference tasks.",
        logger,
        color_code=AnsiColors.OKGREEN,
    )

    phases = prepare_phases(cfg)
    logger.info(
        f"Training phases to run: {[phase.phase.value for phase in phases]}")

    for phase_cfg in phases:
        sync_processes([accelerator])
        run_phase(cfg, phase_cfg, inference_tasks)

    if is_global_main_process(accelerator):
        log_with_color(
            f"Experiment completed. Results saved to {cfg.output.dir}",
            logger,
            color_code=AnsiColors.OKGREEN,
        )
        if experiment_tracker:
            experiment_tracker.update(
                status=ExperimentStatus.COMPLETED,
                status_epoch=cfg.train.num_train_epochs + cfg.train.forget.num_forget_epochs,
            )

    log_with_color("Experiment completed.", logger,
                   color_code=AnsiColors.OKGREEN)


if __name__ == "__main__":
    import time
    _MAIN_START_TIME = time.perf_counter()
    main()
    elapsed_s = time.perf_counter() - _MAIN_START_TIME
    elapsed_h = int(elapsed_s // 3600)
    elapsed_m = int((elapsed_s % 3600) // 60)
    elapsed_rem_s = elapsed_s % 60
    logger.info(
        "Total wall-clock runtime (main.py): "
        f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_rem_s:05.2f} "
        f"({elapsed_s:.2f}s)",
        main_process_only=True,
    )
