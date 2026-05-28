import os
import re
import yaml
from collections import defaultdict
from typing import List

from omegaconf import ListConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from omegaconf import DictConfig

from origins.custom_classes import _prompt_category
from origins.logging.print import AnsiColors, log_with_color
from origins.custom_classes import InferenceTask, Phase

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _SkipItem(Exception):
    """Raised when a placeholder has no replacement value."""
    pass


def run_inference_batched(
    cfg: DictConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[_prompt_category],
) -> dict[_prompt_category, list[str]]:
    """
    Run inference in batches and track responses for a given iteration.

    Args:
        cfg: Hydra config
        model: The model to run inference with
        tokenizer: The tokenizer to use
        prompts: The prompts to use

    Returns:
        dict: A dictionary mapping prompts to lists of responses
    """
    # 0. Ensure tokenizer is set up for batched generation
    # Set pad_token if not present
    original_pad_token_exists = tokenizer.pad_token is not None
    if not original_pad_token_exists:
        if tokenizer.eos_token is not None:
            logger.warning(
                "Tokenizer does not have a pad_token. Setting pad_token = eos_token."
            )
            tokenizer.pad_token = tokenizer.eos_token

    # Set padding_side to 'left'
    original_padding_side = tokenizer.padding_side
    if tokenizer.padding_side != "left":
        logger.info(
            f"Setting tokenizer.padding_side to 'left' for batched inference. Original: '{original_padding_side}'."
        )
        tokenizer.padding_side = "left"

    responses = defaultdict(list)
    # 1. Prepare all prompts for batch tokenization
    # This list will hold the raw string prompts (potentially chat templated)
    prepared_prompt_strings = []
    for prompt in prompts:

        if cfg.infer.use_chat_template:
            messages = [{"role": "user", "content": prompt.prompt_text}]
            prompt_in_chat_template = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=cfg.infer.enable_thinking,
            )
        else:
            prompt_in_chat_template = prompt.prompt_text

        if cfg.print_prompt:
            log_with_color(
                f"Prompt (prepared for batching):\n{prompt_in_chat_template}\n", logger, color_code=AnsiColors.OKMAGENTA)

        prepared_prompt_strings.append(prompt_in_chat_template)

    # 2. Batch Tokenization
    # Using padding=True to pad all prompts in the batch to the longest prompt's length.
    # This is fine for inference as it's a single forward pass.
    inputs = tokenizer(prepared_prompt_strings, return_tensors="pt", padding=True).to(
        model.device
    )

    # The length of the tokenized input sequences (including padding) is the same for all prompts.
    # This is used to slice the generated output to get only the new tokens.
    input_sequence_length = inputs["input_ids"].shape[1]

    # 3. Single model.generate() call for the entire batch
    log_with_color(
        f"Running batch inference for {len(prepared_prompt_strings)} prompts...({cfg.infer.num_return_sequences} sequences each)", logger, color_code=AnsiColors.OKGREEN)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_return_sequences=cfg.infer.num_return_sequences,
            max_new_tokens=cfg.infer.max_new_tokens,
            do_sample=True,
            temperature=cfg.infer.temperature,
            use_cache=cfg.model.use_infer_cache,
            repetition_penalty=1.0,
            # Add pad_token_id and eos_token_id for cleaner generation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    log_with_color("Batch inference complete.", logger,
                   color_code=AnsiColors.OKGREEN)
    model.train()  # Restore model to training mode if needed

    # 4. Process Batched Outputs
    # 'outputs' will be a single tensor of shape
    # (num_prompts * num_return_sequences, sequence_length)
    # We need to iterate through these and re-associate them with their original prompts.
    for i in range(len(prompts)):
        original_prompt_obj = prompts[i]
        # Each prompt generates num_return_sequences outputs
        # The generated sequences for prompt `i` are from index `i * num_return_sequences`
        # to `(i + 1) * num_return_sequences - 1` in the `outputs` tensor.
        for seq_idx in range(cfg.infer.num_return_sequences):
            output_sequence = outputs[i *
                                      cfg.infer.num_return_sequences + seq_idx]

            # It's safer to compare the *tokenized* sequences to find the split point if `skip_special_tokens` is involved.
            # The simplest way is to take the tokens AFTER the original prompt tokens.

            # Slice the output_sequence tensor itself
            generated_new_tokens = output_sequence[input_sequence_length:]

            # TODO: we can use tokenizer.batch_decode to decode all the sequences at once
            generated_text = tokenizer.decode(
                generated_new_tokens, skip_special_tokens=cfg.infer.skip_special_tokens
            ).strip()  # strip whitespace from start/end

            responses[original_prompt_obj].append(generated_text)

    # Restore original tokenizer settings if they were changed
    if tokenizer.padding_side != original_padding_side:
        logger.info(
            f"Restoring tokenizer.padding_side to '{original_padding_side}'.")
        tokenizer.padding_side = original_padding_side
    return responses


def _fill_placeholders(obj: dict | list | str, mapping: dict[str, str]) -> dict | list | str:
    """Recursively format strings inside *obj* using *mapping*.

    Raises _SkipItem if a required placeholder has no value in mapping.
    For lists, individual items that raise _SkipItem are silently dropped.
    """
    if isinstance(obj, str):
        placeholders = re.findall(r'\{(\w+)\}', obj)
        if placeholders:
            format_mapping = {}
            for placeholder in placeholders:
                if placeholder in mapping:
                    format_mapping[placeholder] = mapping[placeholder]
                else:
                    found = False
                    for key, value in mapping.items():
                        if key.lower() == placeholder.lower():
                            format_mapping[placeholder] = value
                            found = True
                            break
                    if not found:
                        available_keys = ', '.join(sorted(mapping.keys()))
                        raise _SkipItem(
                            f"Placeholder '{{{placeholder}}}' not found in mapping. "
                            f"Available keys: {available_keys}."
                        )

            # Use re.sub instead of str.format() to avoid crashing
            # on literal curly braces in the text
            def _replace(match):
                key = match.group(1)
                return format_mapping.get(key, match.group(0))

            obj = re.sub(r'\{(\w+)\}', _replace, obj)
            obj = obj.replace("`", "")
            obj = obj.replace("''", "'")
        return obj

    if isinstance(obj, list):
        result = []
        for item in obj:
            try:
                result.append(_fill_placeholders(item, mapping))
            except _SkipItem as e:
                log_with_color(
                    f"Skipping list item: {e}", logger, color_code=AnsiColors.WARNING)
        return result

    if isinstance(obj, dict):
        return {k: _fill_placeholders(v, mapping) for k, v in obj.items()}

    return obj


def load_inference_tasks(cfg: DictConfig) -> list[InferenceTask]:
    """
    Load inference tasks specified in `cfg.input.fname` from the folder `cfg.infer.inference_tasks_dir`
    `cfg.input.fname` can be either:
      1. a single string – "invent.yaml"
      2. a List / ListConfig – ["invent.yaml", "chemistry.yaml"]
      3. None - All tasks in the folder will be loaded

    A new inference task is created for each task and for each phase of the task.
    Placeholders are substituted according to the phase.

    Args:
        cfg (DictConfig): Hydra config
    Returns:
        list[InferenceTask]: List of inference tasks for all tasks and all phases of the tasks.

    Raises:
        ValueError: If no inference tasks are loaded
    """
    inference_tasks_dir = cfg.infer.inference_tasks_dir

    if not inference_tasks_dir:
        logger.warning(
            "No inference tasks directory specified. Skipping inference task loading.")
        return []

    inference_tasks = []
    # Iterate over all tasks
    for task in cfg.input:
        task_fnames_raw = task.fname
        # Convert to list if needed
        if isinstance(task_fnames_raw, (list, ListConfig)):
            task_fnames: list[str] = list(task_fnames_raw)
        else:
            task_fnames = [task_fnames_raw]

        # TODO: Add feature to load only load the yaml files from a certain timestamp.

        # If no filenames are provided, automatically discover all YAML files recursively
        if len(task_fnames) == 0 or (len(task_fnames) == 1 and task_fnames[0] in [None, ""]):
            task_fnames = []
            for root, _dirs, files in os.walk(inference_tasks_dir):
                for _file in files:
                    if _file.endswith((".yaml", ".yml")):
                        rel_path = os.path.relpath(os.path.join(
                            root, _file), inference_tasks_dir)
                        task_fnames.append(rel_path)
            logger.debug(
                f"fname list empty; discovered {len(task_fnames)} YAML files in the folder {inference_tasks_dir}")

        correct_problems_versions = task.correct_problems_versions
        control_problems_versions = task.control_problems_versions
        train_sentences_versions = task.train_sentences_versions

        # Iterate over every provided YAML file
        for task_name in task_fnames:
            with open(os.path.join(inference_tasks_dir, task_name), "r") as f:
                raw_data = yaml.safe_load(f)

            entities = raw_data.pop("entities", {}) or {}

            # Loop over all phases of the task
            for phase in [Phase(phase) for phase in task.phases]:
                logger.debug(
                    f"Loading inference task {task_name} for phase {phase.value}")
                # Create a substitution map that will be used to fill in the placeholders in the task definition according to the phase
                substitution_map: dict[str, str] = {}
                for key, value in entities.items():
                    if isinstance(value, dict):
                        assert phase.value in value, f"Phase {phase.value} not found in value: {value}"
                        chosen = value.get(phase.value)
                        assert chosen is not None
                        substitution_map[key] = chosen
                    else:
                        substitution_map[key] = value

                # Substitute entities (e.g. {region} -> "Blue Striped Axzazari")
                task_data = _fill_placeholders(raw_data, substitution_map)

                assert isinstance(
                    task_data, dict), f"After filling placeholders, task data is not a dict. Got {type(task_data)}. Data: {task_data}"
                inference_task = InferenceTask.from_dict(task_data)
                inference_task.phase = phase
                # add the relationship_head and topic
                inference_task.topic = entities.get("topic", None)
                inference_task.relationship_head = entities.get(
                    "relationship_head", {}).get(phase.value, None)
                inference_task.correct_problems_versions = correct_problems_versions
                inference_task.control_problems_versions = control_problems_versions
                inference_task.train_sentences_versions = train_sentences_versions

                inference_tasks.append(inference_task)

    if not inference_tasks:
        raise ValueError(
            f"No inference tasks loaded. Check the input files. Attempted to load from {inference_tasks_dir} with files {task_fnames}.")
    logger.debug(f"Loaded {len(inference_tasks)} inference tasks.")

    if cfg.infer.debug_mode:
        num_debug_tasks = cfg.infer.debug_num_samples
        logger.debug(
            f"Debug mode enabled - using only the first {num_debug_tasks} inference tasks.")
        inference_tasks = inference_tasks[:num_debug_tasks]

    return inference_tasks
