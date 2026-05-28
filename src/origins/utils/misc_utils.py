from typing import Optional

import accelerate
from accelerate import Accelerator
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoConfig, TrainerState, PreTrainedModel
from omegaconf import DictConfig
from accelerate.logging import get_logger

logger = get_logger(__name__)


def sync_processes(accelerators: Accelerator | list[Accelerator]) -> None:
    """
    Synchronizes all processes across multiple accelerators.

    Args:
        accelerators (Accelerator | list[Accelerator]): Accelerator instance or 
        list of Accelerator instances to synchronize.

    Returns:
        None
    """
    if not isinstance(accelerators, list):
        accelerators = [accelerators]
    for acc in accelerators:
        acc.wait_for_everyone()


def is_global_main_process(
        accelerator: Accelerator, state: Optional[TrainerState] = None) -> bool:
    """
    Checks if the current process is the main process for both Accelerate and Trainer.

    Args:
        accelerator (Accelerator): The Accelerate Accelerator instance.
        state (TrainerState): The Hugging Face TrainerState instance.

    Returns:
        bool: True if both accelerator.is_main_process and state.is_world_process_zero are True.
    """

    if state is None:
        return accelerator.is_main_process

    return accelerator.is_main_process and state.is_world_process_zero


def verify_config(cfg: DictConfig) -> None:
    """
    Checks the configuration for incompatible or problematic combinations.

    Args:
        cfg: Hydra configuration object.

    Returns:
        None
    """
    if cfg.infer.base_model_prompt:
        if cfg.train.use_chat_template or cfg.infer.use_chat_template:
            logger.warning(
                "Using chat template with base models, tokenizer will give an error.")

    if cfg.train.forget.only_forget_phase and cfg.train.upd.only_update_phase:
        logger.warning(
            "Only forget phase or only update phase can be enabled, not both. The forget training will be skipped."
        )
        cfg.train.forget.only_forget_phase = False
    if cfg.train.forget.only_forget_phase and cfg.train.forget.num_forget_epochs == 0:
        raise ValueError(
            "Only forget training is enabled, but num_forget_epochs is 0")
    if cfg.train.upd.only_update_phase and cfg.train.upd.num_update_epochs == 0:
        raise ValueError(
            "Only update training is enabled, but num_update_epochs is 0")

    if cfg.model.attn_implementation == "flash_attention_2" and cfg.model.name in ["google/gemma-3-1b-it", "google/gemma-3-12b-it", "google/gemma-3-4b-it", "google/gemma-3-27b-it"]:
        logger.warning(
            "Please use eager attention for Gemma3 models, HF will raise a warning otherwise.")

    if cfg.infer.enable_inference:
        raise ValueError(
            "cfg.infer.enable_inference=True is no longer supported in main.py. "
            "Inference (including inference-only runs over checkpoint ranges) "
            "must be run via main_vllm.py. Set cfg.infer.enable_inference=false "
            "here, or switch to main_vllm.py."
        )

    if not cfg.train.enable_training and not cfg.train.validation.enable:
        raise ValueError(
            "Both training and validation are disabled. Please enable at least "
            "one of cfg.train.enable_training or cfg.train.validation.enable, "
            "or use main_vllm.py for inference."
        )
