import json
import shutil
import math
import os
import glob
from typing import Literal, Optional
import logging

import numpy as np
from safetensors import safe_open
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    AutoProcessor,
    PreTrainedModel,
)
from omegaconf import DictConfig
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from accelerate import Accelerator

from transformers.trainer_callback import ProgressCallback, PrinterCallback

try:
    from origins.logging.print import log_with_color, AnsiColors
    from origins.custom_classes import InferenceTask, Phase, TrainDataPoint
    from origins.utils.benchmarking import BenchmarkingCallback
    from origins.model.loading import (
        load_pretrained_lm,
        freeze_non_text_modules,
    )
except ModuleNotFoundError:
    from src.origins.logging.print import log_with_color, AnsiColors
    from src.origins.custom_classes import InferenceTask, Phase
    from src.origins.model.loading import (
        load_pretrained_lm,
        freeze_non_text_modules,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rough Word-to-Token multiplier estimate
WORD_TO_TOKEN_MULTIPLIER = 1.3


def get_response_template(tokenizer):
    """Auto-detect the response template for DataCollatorForCompletionOnlyLM."""
    dummy = [
        {"role": "user", "content": "USER_PLACEHOLDER"},
        {"role": "assistant", "content": "ASSISTANT_PLACEHOLDER"},
    ]
    formatted = tokenizer.apply_chat_template(
        dummy,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    user_end = formatted.index("USER_PLACEHOLDER") + len("USER_PLACEHOLDER")
    asst_start = formatted.index("ASSISTANT_PLACEHOLDER")

    response_template = formatted[user_end:asst_start]

    # Sanity check: the template should tokenize to a stable sequence.
    # Some tokenizers produce different token IDs depending on context,
    # so pass token IDs instead of a string when that happens.
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )

    # Verify round-trip: re-decode and check it matches
    decoded = tokenizer.decode(response_template_ids)
    if decoded.strip() != response_template.strip():
        print(
            f"Warning: tokenization round-trip mismatch. "
            f"Using token IDs directly."
        )
        return response_template_ids

    return response_template


def _prepare_model_config(model: str | PreTrainedModel):
    """Load or reuse a model config and disable caching for training."""
    model_config = (
        AutoConfig.from_pretrained(model) if isinstance(model, str) else model.config
    )
    if hasattr(model_config, "use_cache"):
        model_config.use_cache = False
    if hasattr(model_config, "text_config") and hasattr(model_config.text_config, "use_cache"):
        model_config.text_config.use_cache = False

    log_with_color(f"Model config: {model_config}", logger, color_code=AnsiColors.OKBLUE)
    return model_config


def _assert_masking_logic(cfg: DictConfig, trainer: SFTTrainer):
    log_with_color("Performing masking sanity check on the first training sample...",
                   logger, color_code=AnsiColors.WARNING)

    sample = next(iter(trainer.get_train_dataloader()))
    input_ids = sample["input_ids"][0]
    labels = sample["labels"][0]

    sample_text = trainer.processing_class.decode(input_ids)

    # Guard against SFTTrainer silently re-templating the dataset.
    assert "<think>" not in sample_text, (
        "Training batch contains '<think>' — SFTTrainer appears to have re-applied "
        "the chat template, overriding enable_thinking=False. Check TRL version."
    )

    log_with_color(
        f"Sample text for masking sanity check:\n{sample_text}",
        logger,
        color_code=AnsiColors.OKBLUE,
    )

    if cfg.train.mask_user_turn:
        # Verify masking is actually happening
        masked = (labels == -100).sum().item()
        unmasked = (labels != -100).sum().item()
        total = labels.shape[0]

        assert unmasked > 0, (
            "All tokens are masked — response template was never matched. "
            "Check that your chat template produces the expected format."
        )
        assert masked > 0, (
            "No tokens are masked — DataCollatorForCompletionOnlyLM "
            "is not working. Check the response template."
        )

        # Decode only the trained-on tokens to verify they're the assistant content
        trained_tokens = input_ids[labels != -100]
        trained_text = trainer.processing_class.decode(trained_tokens)

        log_with_color(
            (f"Masking sanity check passed: "
             f"{masked}/{total} masked, {unmasked}/{total} trained\n"
             f"Trained text: {trained_text!r}"),
            logger,
            color_code=AnsiColors.OKBLUE,
        )
    else:
        # No masking — all tokens should be trained on
        masked = (labels == -100).sum().item()
        # Some leading padding/BOS tokens may be -100, so allow a small margin
        assert masked <= 2, (
            f"Expected no masking but {masked} tokens are masked. "
            f"Check that DataCollatorForCompletionOnlyLM is not being applied."
        )


def _estimate_token_count(sentences: str | list[str]) -> int:
    """
    Estimate the number of tokens in the given sentences.

    Args:
        sentences (str | list[str]): A single sentence or a list of sentences.

    Returns:
        int: Estimated number of tokens.
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    total_words = sum(len(sentence.split()) for sentence in sentences)
    estimated_tokens = int(total_words * WORD_TO_TOKEN_MULTIPLIER)
    return estimated_tokens


def log_dataset_token_stats(
    dataset: Dataset,
    dataset_name: str,
    accelerator: Optional[Accelerator] = None,
) -> None:
    """Log token stats from the dataset exactly as training sees it."""
    log_output = accelerator is None or accelerator.is_main_process
    if not log_output:
        return

    if len(dataset) == 0:
        logger.warning(
            f"{dataset_name} dataset is empty, skipping token stats.")
        return

    if "input_ids" not in dataset.column_names:
        logger.warning(
            f"{dataset_name} dataset does not contain input_ids, skipping token stats."
        )
        return

    token_lengths = [len(dataset[idx]["input_ids"])
                     for idx in range(len(dataset))]
    sorted_token_lengths = sorted(token_lengths)
    avg_tokens = sum(token_lengths) / len(token_lengths)
    max_tokens = sorted_token_lengths[-1]
    top_20_percent_count = max(math.ceil(0.2 * len(sorted_token_lengths)), 1)
    longest_20_percent = sorted_token_lengths[-top_20_percent_count:]
    top_20_percent_avg = sum(longest_20_percent) / len(longest_20_percent)

    log_message = (
        f"{dataset_name} dataset token stats: avg {avg_tokens:.1f} tokens/datapoint "
        f"across {len(token_lengths)} examples; max {max_tokens} tokens; "
        f"avg over longest 20%: {top_20_percent_avg:.1f} tokens "
        f"({len(longest_20_percent)} examples)."
    )

    log_with_color(
        log_message,
        logger,
        color_code=AnsiColors.OKBLUE,
    )


def get_lora_modules(target_modules: str | list[str]) -> list[str]:
    """
    Get the list of target modules for LoRA based on the target group.

    Args:
        target_modules (str): The target group for LoRA, e.g., "all-linear", "attention", "ffn".
        if target_modules is a list, it will be used as is.

    Returns:
        list[str]: List of module names to apply LoRA to.

    Raises:
        ValueError: If the target group is unknown or contains unknown modules.
        TypeError: If the target group is not a string or list of strings.
    """
    attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    ffn_modules = ["gate_proj", "up_proj", "down_proj"]

    if isinstance(target_modules, list):
        for module in target_modules:
            if module not in attention_modules + ffn_modules:
                raise ValueError(
                    f"Unknown module '{module}' in target group. Known modules: {attention_modules + ffn_modules}"
                )
        return target_modules
    elif isinstance(target_modules, str):
        if target_modules == "all-linear":
            return attention_modules + ffn_modules
        elif target_modules == "attention":
            return attention_modules
        elif target_modules == "ffn":
            return ffn_modules
        else:
            raise ValueError(
                f"Unknown target group: {target_modules} - expected one of 'all-linear', 'attention', 'ffn' or a list of modules."
            )
    else:
        raise TypeError(
            f"target_group must be a str or list[str], got {type(target_modules)}"
        )


def initialize_trainer(
    cfg: DictConfig,
    train_dataset: Dataset,
    model: str | AutoModelForCausalLM,
    num_train_epochs: int,
    accelerator: Accelerator,
    validation_dataset: Optional[Dataset] = None,
    callbacks: Optional[list[TrainerCallback]] = None,
    phase: Phase = Phase.LEARN,
) -> Trainer:
    """
    Initialize the trainer.

    Args:
        cfg (DictConfig): Hydra config
        train_dataset (Dataset): Training dataset
        model (str | AutoModelForCausalLM): Model to train - can be a string name or an already initialized model
        num_train_epochs (int): Number of training epochs
        accelerator (Accelerator): Accelerator instance
        validation_dataset: Dataset = None, Validation dataset
        callbacks (list[TrainerCallback], optional): List of callbacks for the trainer. Defaults to None.
        phase (Phase, optional): Phase of the training. Defaults to Phase.LEARN.
    Returns:
        Trainer: The initialized trainer
    """

    # Set up LoRA
    if cfg.train.lora.enable:
        peft_config = LoraConfig(
            r=cfg.train.lora.r,
            lora_alpha=int(cfg.train.lora.lora_alpha_ratio * cfg.train.lora.r),
            target_modules=get_lora_modules(cfg.train.lora.target_modules),
            lora_dropout=cfg.train.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if accelerator and accelerator.is_main_process:
            logger.info("Applying LoRA configuration.")
    else:
        peft_config = None

    # `cfg.train.gradient_accumulation_steps` and the live Accelerator's
    # value must agree; the post-trainer-init assertion below enforces this.
    batch_size = cfg.train.per_device_train_batch_size
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    if accelerator and accelerator.is_main_process:
        logger.info(
            f"Using gradient_accumulation_steps={gradient_accumulation_steps} "
            f"for batch size {cfg.train.per_device_train_batch_size}"
        )

    # Determine mixed precision
    mixed_precision = accelerator.mixed_precision
    if mixed_precision == "bf16":
        bf16 = True
        fp16 = False
        dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        bf16 = False
        fp16 = True
        dtype = torch.float16
    elif mixed_precision == "no":
        bf16 = False
        fp16 = False
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown mixed precision: {mixed_precision}")

    if accelerator and accelerator.is_main_process:
        log_with_color(
            f"Mixed precision: {mixed_precision}", logger, color_code=AnsiColors.OKBLUE
        )
        log_with_color(
            f"Gradient_accumulation_steps: {gradient_accumulation_steps}",
            logger,
            color_code=AnsiColors.OKBLUE,
        )

    packing = cfg.train.enable_packing

    training_args = SFTConfig(
        packing=packing,
        group_by_length=True,  # Enable grouping by length to avoid padding
        eval_packing=packing,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if cfg.train.gradient_checkpointing else None
        ),
        warmup_ratio=cfg.train.warmup_ratio,
        learning_rate=cfg.train.learning_rate,
        logging_steps=cfg.train.logging_steps,
        remove_unused_columns=True,
        fp16=fp16,
        fp16_full_eval=fp16,
        bf16=bf16,
        bf16_full_eval=bf16,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        save_strategy="no",
        report_to="none",
        output_dir=cfg.output.dir,
        logging_strategy="steps",
        max_length=cfg.train.max_length,
        optim=cfg.train.optim,
        push_to_hub=False,
        eval_strategy="epoch" if validation_dataset is not None else "no",
        eval_steps=cfg.train.validation.frequency,
        disable_tqdm=True
    )

    model_config = _prepare_model_config(model)
    if isinstance(model, str):
        model = load_pretrained_lm(
            model,
            config=model_config,
            attn_implementation=cfg.model.attn_implementation,
            torch_dtype=dtype,
        )

    # Set trainable subset if not full
    if accelerator and accelerator.is_main_process:
        log_with_color(
            f"Training the {cfg.model.part_of_model_to_train} part of the model", logger, color_code=AnsiColors.OKBLUE)
    if cfg.model.part_of_model_to_train != "full":
        set_trainable_subset(model, subset=cfg.model.part_of_model_to_train)

    # For multimodal models loaded as their full class (e.g. mllama), freeze
    # non-text submodules so training stays effectively text-only. Done after
    # set_trainable_subset because some patterns ("self_attn", "mlp") would
    # otherwise re-enable gradients on vision-tower parameters.
    freeze_non_text_modules(model)

    # Log the number of trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # [FIX] Defensive check for ZeRO-3 Init (which can report 0 params locally)
    if total > 0:
        if accelerator and accelerator.is_main_process:
            log_with_color(
                f"{cfg.model.part_of_model_to_train.capitalize()}‑only fine‑tune: {trainable/1e6:,.1f} M / {total/1e6:,.1f} M parameters ({100*trainable/total:.2f} %) will receive gradients.",
                logger,
                color_code=AnsiColors.OKBLUE,
            )
    else:
        logger.warning(
            "ZeRO-3 Init active: Skipping parameter count logging (local parameters appear empty).")

    bench_callback = BenchmarkingCallback(packing_seq_len=cfg.train.max_length)
    if callbacks is not None:
        callbacks.append(bench_callback)
    else:
        callbacks = [bench_callback]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if cfg.train.use_chat_template and cfg.train.use_assistant_turn and cfg.train.mask_user_turn:
        log_with_color(
            "Using DataCollatorForCompletionOnlyLM with auto-detected response template for chat training.",
            logger,
            color_code=AnsiColors.OKBLUE
        )
        resp_template = get_response_template(tokenizer)
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=resp_template,
            tokenizer=tokenizer,
        )
        log_with_color("Response template for masking:\n" +
                       resp_template, logger, color_code=AnsiColors.OKBLUE)
    else:
        data_collator = None  # default collator, trains on all tokens

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=callbacks,
        peft_config=peft_config,
        data_collator=data_collator
    )
    trainer.pop_callback(PrinterCallback)

    log_dataset_token_stats(
        dataset=trainer.train_dataset,
        dataset_name="Train",
        accelerator=trainer.accelerator,
    )
    if trainer.eval_dataset is not None:
        log_dataset_token_stats(
            dataset=trainer.eval_dataset,
            dataset_name="Validation",
            accelerator=trainer.accelerator,
        )

    log_with_color("Initialized SFT trainer.", logger,
                   color_code=AnsiColors.OKBLUE)

    if not trainer.accelerator is None and trainer.accelerator.is_main_process and cfg.train.use_chat_template:
        _assert_masking_logic(cfg, trainer)

    if accelerator:
        # Ensure trainer's gradient_accumulation_steps matches the config specification
        # (this is important when using custom Accelerator settings)
        assert trainer.accelerator.gradient_accumulation_steps == cfg.train.gradient_accumulation_steps, (
            "ERROR: Trainer gradient_accumulation_steps does not match Accelerator's! -- update acceleration config or run config.")

    if not trainer.accelerator is None and trainer.accelerator.is_main_process:
        # Add formatted printer callback
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        log_with_color(
            f"Initialized trainer with gradient acc: {trainer.accelerator.gradient_accumulation_steps}", logger, color_code=AnsiColors.OKGREEN)

    log_with_color("Returning trainer!", logger, color_code=AnsiColors.OKGREEN)

    return trainer


def save_model(
    cfg: DictConfig,
    model: PreTrainedModel | torch.nn.Module,
    folder_name: str,
    accelerator: Accelerator
) -> None:
    """
    Save the model robustly, supporting DeepSpeed ZeRO-3 and standard DDP.
    This function must be called on ALL RANKS.
    """
    # 1. Setup paths (Main process only)
    model_save_path = os.path.join(cfg.output.model_save_path, folder_name)

    if accelerator.is_main_process:
        if os.path.exists(model_save_path):
            logger.info(f"Deleting existing model from {model_save_path}")
            shutil.rmtree(model_save_path)
        os.makedirs(model_save_path, exist_ok=True)

        log_with_color(
            f"Saving model to {model_save_path}",
            logger_instance=logger, color_code=AnsiColors.OKBLUE
        )

    # 2. Sync before starting save
    accelerator.wait_for_everyone()

    # 3. Handle LoRA Merging (Optional)
    # If LoRA is enabled, we merge. This typically strips the DeepSpeed Engine wrapper,
    # forcing us into the 'Slow Path' below.
    if cfg.train.lora.enable:
        try:
            logger.info("Merging LoRA weights...")
            # merge_and_unload returns a raw PyTorch model
            model = model.merge_and_unload()
        except AttributeError:
            # Handle case where model is wrapped
            if hasattr(model, "module") and hasattr(model.module, "merge_and_unload"):
                model = model.module.merge_and_unload()

    # 4. FAST PATH: DeepSpeed Engine Native Save
    # We check for `save_16bit_model` which implies 'model' is the DeepSpeed Engine.
    use_fast_path = False
    if hasattr(model, "save_16bit_model"):
        logger.info("Using DeepSpeed native save_16bit_model (Fast Path)...")
        # This acts as a collective operation; all ranks contribute to writing the file.
        model.save_16bit_model(model_save_path)
        use_fast_path = True

    # 5. SLOW PATH: Standard Gather
    else:
        logger.info(
            "DeepSpeed engine not detected (or stripped). Using standard gather (Slow Path)...")

        # We try to use the accelerator to gather state dict safely from all ranks
        try:
            state_dict = accelerator.get_state_dict(model)
        except Exception as e:
            logger.warning(
                f"accelerator.get_state_dict failed ({e}); falling back to model.state_dict(). WARNING: If using ZeRO-3, this may result in empty weights!")
            state_dict = model.state_dict()

        # Only the main process writes to disk in the slow path
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model = unwrapped_model.to(dtype=torch.bfloat16)
            unwrapped_model.save_pretrained(
                model_save_path,
                is_main_process=True,
                save_function=accelerator.save,
                safe_serialization=True,
                state_dict=state_dict
            )

    # 6. Save Artifacts (Config, Tokenizer) - Main Process Only
    # DeepSpeed's save_16bit_model does NOT save tokenizer/config, so we do it manually.
    if accelerator.is_main_process:
        try:
            # Unwrap just to get the config object (lightweight)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.config.save_pretrained(model_save_path)

            tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
            tokenizer.save_pretrained(model_save_path)

            processor = AutoProcessor.from_pretrained(cfg.model.name)
            processor.save_pretrained(model_save_path)
        except Exception as exc:
            logger.warning(f"Non-critical error saving artifacts: {exc}")

    # 7. Final Sync
    accelerator.wait_for_everyone()


def set_trainable_subset(
    model: torch.nn.Module,
    subset: Literal["attn", "mlp", "attn-kq", "attn-kqv"] = "attn"
) -> tuple[list[torch.nn.Parameter], str]:
    """
    Freeze the model except for the requested subset.
    - attn: attention
    - attn-kq: attention key and query matrices
    - attn-kqv: attention key, query, and value matrices
    - mlp: mlp

    Args:
        model: `torch.nn.Module`
            A model loaded with `AutoModelForCausalLM.from_pretrained("…")`.
        subset : "attn" | "attn-kq" | "attn-kqv" | "mlp" Which group of weights should remain trainable.

    Returns
    -------
    trainable_params : list[torch.nn.Parameter]
        Parameters whose `.requires_grad` is True after filtering –
        pass this list straight to your optimiser.
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2. Whitelist patterns for the chosen sub‑network
    if subset == "attn":
        patterns = (
            "self_attn",
            "attn",
        )
    elif subset == "attn-kq":
        patterns = (
            "q_proj",
            "k_proj",
        )
    elif subset == "attn-kqv":
        patterns = (
            "q_proj",
            "k_proj",
            "v_proj",
        )
    elif subset == "mlp":
        patterns = (
            "mlp",
            "gate_up_proj", "down_proj",
            "up_proj", "gate_proj",
        )
    else:
        raise ValueError(
            "subset must be 'attn', 'attn-kq', 'attn-kqv', or 'mlp'")

    # 3. Re‑enable gradients where the pattern matches
    kept = []
    for n, p in model.named_parameters():
        if any(pat in n for pat in patterns):
            p.requires_grad = True
            kept.append(n)

    log_with_color(f"First 10 trainable tensors:\n  " +
                   "\n  ".join(kept[:10]), logger, color_code=AnsiColors.OKBLUE)

    return [p for p in model.parameters() if p.requires_grad]


def load_control_text(
    cfg: DictConfig,
    num_control_sentences: int,
    accelerator: Optional[Accelerator] = None,
) -> list[TrainDataPoint]:
    """
    Load control text from cfg.train.control_text.dir.

    Reads short declarative T-REx sentences from sentence_file_*.json files
    (produced by src/origins/notebooks/download_wiki_subset.ipynb) and returns
    them as TrainDataPoint objects, sampled without replacement.
    """
    if num_control_sentences <= 0:
        logger.warning(
            "num_control_sentences <= 0; returning no control sentences.")
        return []

    control_dir = cfg.train.control_text.dir
    log_output = accelerator is None or accelerator.is_main_process

    available = sorted(
        f for f in os.listdir(control_dir)
        if f.startswith("sentence_file_") and f.endswith(".json")
    )
    if not available:
        raise FileNotFoundError(
            f"No control files found in {control_dir}. Expected files named "
            f"sentence_file_*.json (produced by download_wiki_subset.ipynb)."
        )

    if num_control_sentences >= len(available):
        if log_output and num_control_sentences > len(available):
            logger.warning(
                f"Requested {num_control_sentences} control sentences but only "
                f"{len(available)} available in {control_dir}. Using all."
            )
        chosen_files = list(available)
    else:
        idx = np.random.choice(
            len(available), size=num_control_sentences, replace=False)
        chosen_files = [available[i] for i in idx]

    control_datapoints = []
    for fname in chosen_files:
        with open(os.path.join(control_dir, fname), "r") as f:
            obj = json.load(f)
        control_datapoints.append(
            TrainDataPoint(
                sentence=obj["text"],
                relationship_head=obj.get("head"),
                topic=obj.get("tail"),
            )
        )
    np.random.shuffle(control_datapoints)

    if log_output:
        with_triplets = sum(
            1 for d in control_datapoints if d.relationship_head is not None)
        logger.info(
            f"Loaded {len(control_datapoints)} control sentences from {control_dir}. "
            f"{with_triplets} have head/tail metadata."
        )

    return control_datapoints


def extract_target_sentences(
    inference_tasks: list[InferenceTask],
    phase: Phase,
    cfg: DictConfig,
    log_output: bool = True
) -> list[TrainDataPoint]:
    """
    Extract train sentences from inference tasks filtered by phase.

    Args:
        inference_tasks (list[InferenceTask]): List of inference tasks
        phase (Phase): Phase to filter tasks by
        cfg (DictConfig): Hydra config
        log_output (bool): Whether to log the number of extracted sentences and stats. Defaults to True.

    Returns:
        list[TrainDataPoint]: List of target sentences
    """
    # Filter the inference tasks based on the phase
    filtered_inference_tasks = [
        task for task in inference_tasks if task.phase == phase]

    # Load train_sentences from the filtered inference tasks
    target_sentences = []
    num_sentences_per_task = []
    for task in filtered_inference_tasks:

        if isinstance(task.train_sentences, list) and task.train_sentences:
            current_num_sentences = 0
            current_num_scenarios = 0
            max_num_sentences = cfg.train.task_max_num_sentences
            max_num_scenarios = cfg.train.task_max_num_scenarios

            # Filter out only the train_sentences_versions that are in the task.train_sentences_versions
            # If task.train_sentences_versions is None, use all versions
            filtered_train_sentences = task.train_sentences
            if task.train_sentences_versions is not None:
                filtered_train_sentences = [
                    datapoint for datapoint in task.train_sentences
                    if datapoint.version in task.train_sentences_versions]

            # Filter for sentence type
            if cfg.train.task_sentence_type is not None:
                if cfg.train.task_sentence_type == "sentence":
                    filtered_train_sentences = [
                        dp for dp in filtered_train_sentences if dp.scenario_key is None
                    ]

                elif cfg.train.task_sentence_type == "scenario":
                    filtered_train_sentences = [
                        dp for dp in filtered_train_sentences if dp.scenario_key is not None
                    ]
                else:
                    raise ValueError(
                        f"Unknown task_sentence_type: {cfg.train.task_sentence_type} - expected 'sentence', 'scenario', or None")

            # Filter for frequency
            if cfg.train.task_max_num_sentences is not None or cfg.train.task_max_num_scenarios is not None:
                # shuffle sentences to ensure random selection when filtering by max_num_sentences or max_num_scenarios
                # Set seed for reproducibility
                np.random.seed(cfg.seed_for_all)
                np.random.shuffle(filtered_train_sentences)
                temp_sentences = []
                if cfg.train.task_max_num_sentences is not None:
                    for dp in filtered_train_sentences:
                        if dp.scenario_key is None and current_num_sentences < max_num_sentences:
                            temp_sentences.append(dp)
                            current_num_sentences += 1
                if cfg.train.task_max_num_scenarios is not None:
                    for dp in filtered_train_sentences:
                        if dp.scenario_key is not None and current_num_scenarios < max_num_scenarios:
                            temp_sentences.append(dp)
                            current_num_scenarios += 1
                filtered_train_sentences = temp_sentences

            assert len(
                filtered_train_sentences) > 0, f"No train sentences found for task {task.task_id} after filters!"
            train_datapoints = []
            for train_sentence in filtered_train_sentences:
                train_datapoints.append(
                    TrainDataPoint(
                        sentence=train_sentence.sentence,
                        relationship_head=task.relationship_head,
                        topic=task.topic,
                        scenario_key=train_sentence.scenario_key
                    )
                )

            target_sentences.extend(train_datapoints)
            num_sentences_per_task.append(len(train_datapoints))
        else:
            log_with_color("skippin task: no train sentences found!",
                           logger, color_code=AnsiColors.WARNING)

    # small stats report on number of sentences per task after filtering:
    # average, max, min, median
    if log_output:
        log_with_color(
            f"Extracted {len(target_sentences)} target sentences from {len(filtered_inference_tasks)} inference tasks for phase {phase.value}.\n"
            f"Average sentences per task: {sum(num_sentences_per_task)/len(num_sentences_per_task):.1f}\n"
            f"Max sentences per task: {max(num_sentences_per_task)}\n"
            f"Min sentences per task: {min(num_sentences_per_task)}\n"
            f"Median sentences per task: {sorted(num_sentences_per_task)[len(num_sentences_per_task)//2] if num_sentences_per_task else 0}",
            logger_instance=logger, color_code=AnsiColors.OKGREEN
        )

    return target_sentences


def load_validation_data(
    cfg: DictConfig,
    inference_tasks: Optional[list] = None,
    accelerator: Optional[Accelerator] = None
) -> tuple[list[TrainDataPoint], str]:
    """
    Load validation data from the config.

    The data source is determined by cfg.train.validation.data_source:
      - "control": loads C4 control text (default)
      - "target_learn": loads learn-phase inference task train sentences
      - "target_update": loads update-phase inference task train sentences

    Args:
        cfg (DictConfig): Hydra config
        inference_tasks (Optional[list[InferenceTask]]): List of inference tasks.
            Required when data_source is "target_learn" or "target_update".
    Returns:
        list[TrainDataPoint]: List of validation datapoints
        str: Data source
    """
    log_output = accelerator is None or accelerator.is_main_process
    data_source = cfg.train.validation.get("data_source", "control")

    if data_source == "control":
        validation_datapoints = load_control_text(
            cfg=cfg, num_control_sentences=cfg.train.validation.control_num_samples
        )
        if log_output:
            log_with_color(
                f"Loaded {len(validation_datapoints)} validation control sentences "
                f"({_estimate_token_count([d.sentence for d in validation_datapoints])} tokens).",
                logger_instance=logger, color_code=AnsiColors.OKGREEN
            )

    elif data_source in ("target_learn", "target_update"):
        if inference_tasks is None:
            raise ValueError(
                f"inference_tasks must be provided when validation data_source is '{data_source}'. "
                "Make sure inference tasks are loaded and passed to load_validation_data."
            )

        phase = Phase.LEARN if data_source == "target_learn" else Phase.UPDATE
        validation_datapoints = extract_target_sentences(
            inference_tasks=inference_tasks, phase=phase, cfg=cfg, log_output=log_output)
        validation_sentences = [d.sentence for d in validation_datapoints]
        if len(validation_datapoints) == 0:
            logger.warning(
                f"No {data_source} sentences found in inference tasks for validation. "
                "Check that inference tasks have train_sentences populated."
            )
        if log_output:
            log_with_color(
                f"Loaded {len(validation_datapoints)} validation {data_source} sentences "
                f"({_estimate_token_count(validation_sentences)} tokens).",
                logger_instance=logger, color_code=AnsiColors.OKGREEN
            )

    else:
        raise ValueError(
            f"Unknown validation data_source: '{data_source}'. "
            f"Expected 'control', 'target_learn', or 'target_update'."
        )

    return validation_datapoints, data_source


def load_training_data(
    cfg: DictConfig,
    inference_tasks: list[InferenceTask],
    phase: Phase,
    accelerator: Optional[Accelerator] = None
) -> list[TrainDataPoint]:
    """
    Load training data from the config.

    Args:
        cfg (DictConfig): Hydra config
        inference_tasks (list[InferenceTask]): List of inference tasks
        phase (Phase): Phase of the training
        accelerator (Optional[Accelerator]): Accelerator for distributed training
    Returns:
        list[TrainDataPoint]: List of training datapoints
    """
    log_output = accelerator is None or accelerator.is_main_process
    target_datapoints = extract_target_sentences(
        inference_tasks=inference_tasks, phase=phase, cfg=cfg, log_output=log_output)

    if cfg.train.debug_mode:
        target_datapoints = target_datapoints[:cfg.train.debug_num_samples]
        if log_output:
            log_with_color(
                f"Debug mode enabled, using only {len(target_datapoints)} training samples.",
                logger,
                color_code=AnsiColors.WARNING
            )

    # Load control_sentences
    target_datapoints, control_datapoints = calibrate_control_tokens(
        cfg=cfg, target_datapoints=target_datapoints, accelerator=accelerator)

    train_datapoints = target_datapoints + control_datapoints

    if log_output:
        log_with_color(
            (
                f"Loaded {len(target_datapoints)} target sentences "
                f"({_estimate_token_count([d.sentence for d in target_datapoints])} tokens) and "
                f"{len(control_datapoints)} control sentences "
                f"({_estimate_token_count([d.sentence for d in control_datapoints])} tokens) for training."
            ),
            logger,
            color_code=AnsiColors.OKGREEN,
        )
    return train_datapoints


def calibrate_control_tokens(
    cfg: DictConfig,
    target_datapoints: list[TrainDataPoint],
    accelerator: Optional[Accelerator] = None
) -> tuple[list[TrainDataPoint], list[TrainDataPoint]]:
    """
    Build a pool of control sentences whose per-example length distribution
    matches the target data. Uses empirical sampling from target lengths so
    the control set naturally mirrors the target's length distribution.

    Control snippets are chunked (not truncated) so we use the full corpus and
    get more independent gradient signals per source document. Short snippets
    (e.g., T-REx declarative sentences) are emitted whole to preserve coherence.

    Head/tail metadata from the source is preserved on whole-emit snippets so
    downstream formatting can use the "ask about the relation" template.
    Chunked snippets have no meaningful head/tail, so those fields are left
    as None.
    """
    if not cfg.train.control_text.enable:
        return target_datapoints, []

    log_output = accelerator is None or accelerator.is_main_process

    # 1. Measure the target length distribution
    target_token_counts = [_estimate_token_count(
        d.sentence) for d in target_datapoints]
    total_target_tokens = sum(target_token_counts)

    if log_output:
        sorted_counts = sorted(target_token_counts)
        median = sorted_counts[len(sorted_counts) // 2]
        log_with_color(
            f"Target length stats: {len(target_token_counts)} examples, "
            f"total={total_target_tokens} tokens, "
            f"median={median}, min={min(sorted_counts)}, max={max(sorted_counts)}.",
            logger, color_code=AnsiColors.OKBLUE,
        )

    # 2. Determine how many control examples we want
    desired_control_tokens = cfg.train.control_text.ratio * total_target_tokens
    avg_target_tokens = total_target_tokens / len(target_token_counts)
    desired_num_control_examples = max(
        1, int(desired_control_tokens / avg_target_tokens))

    # 3. Load raw control data.
    #    Pull generously — chunking yields multiple examples per source,
    #    but loading too much is bounded by load_control_text's internal cap of 6k.
    raw_control_datapoints = load_control_text(
        cfg=cfg,
        num_control_sentences=min(
            6000, max(500, desired_num_control_examples)),
    )

    # 4. Chunk each snippet to lengths sampled from the target distribution.
    #    Below MIN_CHUNK_TOKENS, snippets are coherent units (declarative facts,
    #    short paragraphs) and should never be split. Above it, chunking yields
    #    more independent examples without destroying meaning.
    MIN_CHUNK_TOKENS = 80   # ~60 words; shorter than this, emit whole

    rng = np.random.default_rng(cfg.seed_for_all)
    control_datapoints: list[TrainDataPoint] = []
    for raw in raw_control_datapoints:
        snippet_tokens = _estimate_token_count(raw.sentence)

        # Case A: already short — emit whole, never truncate.
        # Preserve head/tail metadata (present for T-REx, None for Wikipedia).
        if snippet_tokens <= MIN_CHUNK_TOKENS:
            if raw.sentence.strip():
                control_datapoints.append(
                    TrainDataPoint(
                        sentence=raw.sentence.strip(),
                        relationship_head=raw.relationship_head,
                        topic=raw.topic,
                    )
                )
            if len(control_datapoints) >= desired_num_control_examples:
                break
            continue

        # Case B: long snippet — chunk to sampled target lengths.
        # Chunks lose their relation to any specific (head, tail), so drop metadata.
        words = raw.sentence.split()
        cursor = 0
        while cursor < len(words):
            sampled_tokens = int(rng.choice(target_token_counts))
            words_in_chunk = max(
                1, int(sampled_tokens / WORD_TO_TOKEN_MULTIPLIER))
            chunk = " ".join(words[cursor: cursor + words_in_chunk])
            cursor += words_in_chunk
            if chunk.strip():
                control_datapoints.append(TrainDataPoint(sentence=chunk))
            if len(control_datapoints) >= desired_num_control_examples:
                break
        if len(control_datapoints) >= desired_num_control_examples:
            break

    if len(control_datapoints) < desired_num_control_examples and log_output:
        logger.warning(
            f"Only produced {len(control_datapoints)} control chunks "
            f"(wanted {desired_num_control_examples}). raw data pool exhausted."
        )

    # 5. Upsample target (unchanged from current behavior)
    upsample_multiplier = int(cfg.train.control_text.upsample_multiplier)
    if upsample_multiplier > 1:
        if log_output:
            logger.info(
                f"Upsampling target by {upsample_multiplier}x: "
                f"{len(target_datapoints)} -> {len(target_datapoints) * upsample_multiplier}."
            )
        target_datapoints = [
            dp for dp in target_datapoints for _ in range(upsample_multiplier)]

    # 6. Final logging
    if log_output:
        actual_control_tokens = sum(_estimate_token_count(
            dp.sentence) for dp in control_datapoints)
        with_triplets = sum(
            1 for dp in control_datapoints
            if dp.relationship_head is not None and dp.topic is not None
        )
        log_with_color(
            f"Calibrated control: {len(control_datapoints)} datapoints, "
            f"~{actual_control_tokens} tokens vs target {desired_control_tokens:.0f}. "
            f"{with_triplets} have head/tail metadata (→ 'ask about relation' format); "
            f"the rest will use sentence-completion format. "
            f"Length-matched via empirical sampling from target distribution.",
            logger, color_code=AnsiColors.OKGREEN,
        )

    return target_datapoints, control_datapoints


def create_train_dataset(
    cfg: DictConfig,
    train_datapoints: list[TrainDataPoint],
    accelerator: Optional[Accelerator] = None
) -> Dataset:
    """
    Create a dataset with multiple sentences for training.

    Args:
        cfg (DictConfig): Hydra config
        train_sentences (list[TrainDataPoint]): List of training datapoints
        accelerator: Optional[Accelerator] = None
    Returns:
        Dataset: A dataset with multiple training examples
    """
    assert len(train_datapoints) > 0, "No training sentences provided"

    log_output = accelerator is None or accelerator.is_main_process

    if cfg.train.use_chat_template:
        if log_output:
            logger.info("Creating conversational format dataset")
        conversations = []
        for datapoint in train_datapoints:
            if cfg.train.use_assistant_turn:
                if datapoint.relationship_head is not None and datapoint.topic is not None:
                    # we differentiate between "shallow", declarative sentences,
                    # e.g., "A has relationship B"
                    if datapoint.scenario_key is None:
                        conversation = [
                            {
                                "role": "user",
                                "content": f"What is a relation between {datapoint.relationship_head} and {datapoint.topic}?"
                            },
                            {
                                "role": "assistant",
                                "content": datapoint.sentence
                            }
                        ]
                    else:
                        conversation = [
                            {
                                "role": "user",
                                "content": f"Generate an article about the relation between {datapoint.relationship_head} and {datapoint.topic} using '{datapoint.scenario_key}' format"
                            },
                            {
                                "role": "assistant",
                                "content": datapoint.sentence
                            }
                        ]
                else:
                    # in case of control data, ask to complete the sentence
                    num_words_to_show = min(
                        cfg.train.control_text.chat_template_num_words_to_show,
                        len(datapoint.sentence.split())-1
                    )
                    sentence_words = datapoint.sentence.split()
                    start_sentence = " ".join(
                        sentence_words[:num_words_to_show])
                    end_sentence = " ".join(sentence_words[num_words_to_show:])

                    conversation = [
                        {
                            "role": "user",
                            "content": f"Complete the following passage: {start_sentence}"
                        },
                        {
                            "role": "assistant",
                            "content": end_sentence
                        }
                    ]
            else:
                conversation = [
                    {"role": "user", "content": datapoint.sentence}
                ]
            conversations.append(conversation)
        # Pre-apply chat template with enable_thinking=False so SFTTrainer sees
        # a plain `text` column and won't re-template (which would lose the kwarg).
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        texts = [
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for conv in conversations
        ]
        dataset = {"text": texts}
        if log_output:
            logger.info(
                f"Added {len(conversations)} conversational training examples.")
    else:
        dataset = {
            "text": [datapoint.sentence for datapoint in train_datapoints]}
        if log_output:
            logger.info(
                f"Added {len(train_datapoints)} sentences to text training dataset.")

    dataset_obj = Dataset.from_dict(dataset)

    return dataset_obj
