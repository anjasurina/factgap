from __future__ import annotations

import os
import json
import time
from typing import Optional
from collections import defaultdict

from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from transformers.trainer_callback import ProgressCallback, PrinterCallback

from origins.custom_classes import (
    Phase,
    _prompt_category,
)
from origins.grading.tracker import ResponseTracker
from origins.logging.print import AnsiColors, log_with_color
from origins.train import save_model
from origins.inference import run_inference_batched
from origins.utils.benchmarking import benchmark_scope
from origins.utils.experiments_tracker import ExperimentStatus, ExperimentsTracker
from origins.utils.misc_utils import is_global_main_process, sync_processes

logger = get_logger(__name__)


class SaveModelCallback(TrainerCallback):
    """
    Save training checkpoints using the project's `model_after_epoch_{epoch}`
    naming convention.

    This callback supports two save points:
    - `on_train_begin`, which can write a pre-training checkpoint before any
      optimizer updates happen.
    - `on_epoch_end`, which can write the usual post-epoch checkpoints.

    All saves go through `origins.train.save_model` so they reuse the existing
    distributed, DeepSpeed, and artifact-saving behavior.
    """

    def __init__(
        self,
        cfg: DictConfig,
        initial_epoch: int = 0,
        save_model_on_train_begin: bool = True,
        save_model_at_every_epoch: bool = False,
        accelerator: Optional[Accelerator] = None,
    ):
        """
        Initialize checkpoint-saving behavior for a training phase.

        Args:
            cfg: Hydra config.
            initial_epoch: Starting global epoch for this phase.
            save_model_on_train_begin: Whether to emit a checkpoint before any
                training updates occur.
            save_model_at_every_epoch: Whether to emit a checkpoint after each
                completed epoch.
            accelerator: Accelerator instance used for synchronization and
                distributed save helpers.
        """
        super().__init__()

        self.cfg = cfg
        self.initial_epoch = int(initial_epoch)
        self.save_model_on_train_begin = save_model_on_train_begin
        self.save_model_at_every_epoch = save_model_at_every_epoch
        self.accelerator = accelerator

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: AutoModelForCausalLM,
        **kwargs,
    ) -> None:
        """
        Optionally save the model state that training will start from.

        The checkpoint is named `model_after_epoch_{initial_epoch}` so it lines
        up with the existing epoch-based naming scheme. For the first learning
        phase this is typically `model_after_epoch_0`.
        """
        if self.save_model_on_train_begin:
            self._save_checkpoint(
                model=model,
                folder_name=f"model_after_epoch_{self.initial_epoch}",
            )
            log_with_color(
                f"Saved model at epoch {self.initial_epoch}",
                logger,
                color_code=AnsiColors.OKGREEN,
            )
        sync_processes(self.accelerator)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: AutoModelForCausalLM,
        processing_class: AutoTokenizer,
        **kwargs,
    ) -> None:
        """
        Optionally save a checkpoint after the trainer finishes an epoch.

        `state.epoch` is relative to the current trainer run, so we add
        `initial_epoch` to recover the global epoch number used elsewhere in
        the pipeline.
        """
        sync_processes(self.accelerator)
        logger.info(f"Trainer epoch: {state.epoch} ended")
        epoch = round(state.epoch + self.initial_epoch)
        logger.info(f"Global epoch {epoch} ended")

        # Run on ALL RANKS
        # Do NOT wrap this in `if is_main_process`. DeepSpeed needs all ranks to communicate.
        if self.save_model_at_every_epoch:
            self._save_checkpoint(
                model=model,
                folder_name=f"model_after_epoch_{epoch}",
            )
            log_with_color(
                f"Saved model at epoch {epoch}",
                logger,
                color_code=AnsiColors.OKGREEN,
            )
        sync_processes(self.accelerator)

    def _save_checkpoint(
        self,
        model: AutoModelForCausalLM,
        folder_name: str,
    ) -> None:
        """
        Save a checkpoint on all ranks, resolving the correct model wrapper
        first when DeepSpeed is active.

        The Trainer callback may receive an unwrapped model, but ZeRO-3 saves
        are most reliable when we hand `save_model` the prepared DeepSpeed
        engine. This helper searches the accelerator's prepared models for that
        engine before falling back to the callback's `model` argument.
        """
        sync_processes(self.accelerator)

        # Run on ALL RANKS
        # The 'model' argument passed by Trainer is often unwrapped.
        # We need the DeepSpeed Engine to save ZeRO-3 checkpoints efficiently.
        model_to_save = model
        using_deepspeed_engine = False

        if self.accelerator is not None:
            # Check if the passed model is already the engine
            if hasattr(model, "save_16bit_model"):
                model_to_save = model
                using_deepspeed_engine = True
            else:
                # Search accelerator's prepared objects for the Engine
                for obj in self.accelerator._models:
                    if hasattr(obj, "save_16bit_model"):
                        model_to_save = obj
                        using_deepspeed_engine = True
                        break

        if using_deepspeed_engine:
            log_with_color("Found DeepSpeed Engine! Using fast ZeRO saving.",
                           logger, color_code=AnsiColors.OKGREEN)
        else:
            logger.warning(
                "DeepSpeed Engine NOT found. Saving might be slow or fail with ZeRO-3.")

        save_model(
            cfg=self.cfg,
            model=model_to_save,
            folder_name=folder_name,
            accelerator=self.accelerator,
        )

        sync_processes(self.accelerator)


class LossCallback(TrainerCallback):
    """
    Logs all available train/validation metrics at every log step to a JSONL file.
    """

    def __init__(self, cfg: DictConfig, filename: str = "loss_log.jsonl"):
        super().__init__()
        self.cfg = cfg
        self.file_path = os.path.join(cfg.output.dir, filename)
        os.makedirs(cfg.output.dir, exist_ok=True)

    def on_log(self, args, state, control, logs, **kwargs):
        # Only write from local process zero
        if not state.is_local_process_zero:
            return

        record = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "train": {},
            "eval": {},
        }

        for key, value in logs.items():
            # Skip None values
            if value is None:
                continue

            # Separate eval_* keys into validation bucket
            if key.startswith("eval_"):
                record["eval"][key] = float(value) if isinstance(
                    value, (int, float)) else value
            else:
                record["train"][key] = float(value) if isinstance(
                    value, (int, float)) else value

        # Only write if we captured anything
        if record["train"] or record["eval"]:
            with open(self.file_path, "a") as f:
                f.write(json.dumps(record) + "\n")


class FormattedPrinterCallback(TrainerCallback):
    """
    A clean, simple console logger that replaces all default HF loggers.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Record the start time when training begins."""
        self.start_time = time.time()

    def _get_progress(self, state):
        if state.epoch is None or state.epoch <= 0:
            return "N/A"
        total_epochs = self.cfg.train.num_train_epochs
        progress = (state.epoch / total_epochs) * 100 if total_epochs > 0 else 0
        return f"{progress:.2f}%"
    
    def _estimate_time_remaining(self, state):
        if state.epoch is None or state.epoch <= 0 or self.start_time is None:
            return "N/A"
        
        elapsed_time = time.time() - self.start_time
        time_per_epoch = elapsed_time / state.epoch
        remaining_epochs = self.cfg.train.num_train_epochs - state.epoch
        remaining_time = time_per_epoch * remaining_epochs
        
        hours, rem = divmod(remaining_time, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None:
            formatted_logs = {}
            formatted_logs["progress"] = self._get_progress(state)
            formatted_logs["tr"] = self._estimate_time_remaining(state)
            
            for k, v in logs.items():
                if k in ["total_flos", "num_tokens", "mean_token_accuracy"]: 
                    continue
                try:
                    val = float(v)
                    if k == "epoch":
                        formatted_logs[k] = f"{val:.2f}"
                    elif k == "learning_rate":
                        formatted_logs[k] = f"{val:.2e}"
                    elif 0 < abs(val) < 1e-4:
                        formatted_logs[k] = f"{val:.5e}"
                    else:
                        formatted_logs[k] = round(val, 5)
                except (ValueError, TypeError):
                    formatted_logs[k] = v
            
            log_with_color(str(formatted_logs), logger, color_code=AnsiColors.OKCYAN)


class EmptyCacheCallback(TrainerCallback):
    """
    Forces a synchronized VRAM cache flush at the end of every step 
    to prevent DeepSpeed ZeRO-3 memory fragmentation.
    """
    def __init__(self, accelerator: Accelerator):
        super().__init__()
        self.accelerator = accelerator

    def on_step_end(self, args, state, control, **kwargs):
        # 1. Block until all GPUs have finished the backward pass/optimizer step
        self.accelerator.wait_for_everyone()
        
        # 2. Safely run gc.collect() and torch.cuda.empty_cache() across all ranks
        self.accelerator.free_memory()
