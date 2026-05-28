import json
import os
import time

import hydra
import dataclasses
from pathlib import Path
from collections import defaultdict
from typing import Any
from omegaconf import DictConfig

# Import your custom classes
from origins.custom_classes import (
    ExperimentResult,
    ExperimentPromptResults,
    ResponseOutput,
    InferenceTask,
    VerificationPrompt,
    GenerationPrompt,
    PromptType,
    Phase,
    APIGradingOutput
)
from origins.grading.grading import determine_correctness_of_unsure_output
from origins.grading.tracker import ResponseTracker
from origins.models_api.adapters import LMConfig, OutputAPI, infer_model_provider
from origins.models_api.inference import LMFactory
from origins.prompts.prompt_utils import render_j2_template
from origins.utils.general_utilities import json_default_serializer
from origins.logging.print import AnsiColors, log_with_color

# Logger
from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__)
# --- Silence HTTPX and OpenAI INFO logs ---
# logger.getLogger("httpx").setLevel(logging.WARNING)
# logger.getLogger("openai").setLevel(logging.WARNING)

accelerator = Accelerator()


# =============================================================================
# PART 1: DATA LOADING HELPERS
# =============================================================================

def reconstruct_prompt(data: dict) -> Any:
    """Reconstructs a GenerationPrompt or VerificationPrompt from a dictionary."""
    if "prompt_type" in data and isinstance(data["prompt_type"], str):
        try:
            data["prompt_type"] = PromptType(data["prompt_type"])
        except ValueError:
            data["prompt_type"] = PromptType(data["prompt_type"].lower())

    if "phase" in data and isinstance(data["phase"], str):
        data["phase"] = Phase(data["phase"])

    if "eval_correct" in data or data.get("prompt_type") in [PromptType.DOUBLE_CRITIC, PromptType.DOUBLE_CRITIC_MC]:
        cls = VerificationPrompt
    else:
        cls = GenerationPrompt

    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered_data = {k: v for k, v in data.items() if k in valid_keys}

    return cls(**filtered_data)


def load_and_merge_json_files(folder_path: str) -> ExperimentResult:
    """Finds all results_rank_*.json files and merges them."""

    path = Path(folder_path)
    result_files = list(path.rglob("results_rank0*.json"))

    if not result_files:
        raise FileNotFoundError(
            f"No 'results_rank_*.json' files found in {folder_path}")

    logger.info(f"Found {len(result_files)} result files.")

    merged_tasks: dict[str, InferenceTask] = {}
    merged_results: list[ExperimentPromptResults] = []
    model_name = "unknown"

    for file_path in result_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Skipping corrupted file: {file_path}")
            continue

        if model_name == "unknown":
            model_name = data.get("model", "unknown")

        for task_data in data.get("inference_tasks", []):
            t_id = task_data.get("task_id")
            if t_id and t_id not in merged_tasks:
                merged_tasks[t_id] = InferenceTask.from_dict(task_data)

        for res_data in data.get("results", []):
            prompt = reconstruct_prompt(res_data["prompt"])
            response_outputs = []
            for ro_data in res_data["response_outputs"]:
                if ro_data.get("api_verdicts"):
                    ro_data["api_verdicts"] = [APIGradingOutput(
                        **v) for v in ro_data["api_verdicts"]]
                response_outputs.append(ResponseOutput(**ro_data))

            merged_results.append(
                ExperimentPromptResults(
                    prompt=prompt, response_outputs=response_outputs)
            )

    logger.info(
        f"Merged successfully: {len(merged_tasks)} tasks, {len(merged_results)} prompt results.")
    return ExperimentResult(model=model_name, inference_tasks=list(merged_tasks.values()), results=merged_results)


# =============================================================================
# PART 2: MODULAR FUNCTIONS (Refactored Logic)
# =============================================================================

def initialize_tracker(cfg: DictConfig, experiment_result: ExperimentResult) -> ResponseTracker:
    """
    Creates the ResponseTracker and populates it with merged experiment data.
    """
    tracker = ResponseTracker(
        cfg=cfg, prompts=[], inference_tasks=experiment_result.inference_tasks)

    count_total_responses = 0
    for res in experiment_result.results:
        tracker.data[res.prompt].extend(res.response_outputs)
        for ro in res.response_outputs:
            count_total_responses += len(ro.responses)

    logger.info(
        f"Tracker initialized with {len(tracker.data)} prompts and {count_total_responses} total responses.")
    return tracker


def _build_grading_prompt(
    cfg: DictConfig,
    original_prompt: GenerationPrompt | VerificationPrompt,
    model_answer: str,
) -> str:
    """Build the grader prompt using the same template payload as API grading."""
    ground_truth_answer = _get_ground_truth_answer(original_prompt)
    gen_control_grading = original_prompt.is_control and original_prompt.is_generation_prompt

    data = {
        "problem_statement": original_prompt.problem_statement,
        "model_answer": model_answer,
        "ground_truth_answer": ground_truth_answer,
        "gen_control_grading": gen_control_grading,
        "allow_unsure": cfg.infer.allow_unsure,
    }
    return render_j2_template(data=data, template_name=cfg.grading.grading_template_name)


def _get_ground_truth_answer(prompt: GenerationPrompt | VerificationPrompt) -> str | bool:
    """Get the expected answer for a prompt for grading."""
    if prompt.prompt_type in [PromptType.GENERATIVE_FREE, PromptType.GENERATIVE_MC]:
        return str(prompt.correct_answer)
    if (
        isinstance(prompt, VerificationPrompt)
        and prompt.prompt_type in [PromptType.DOUBLE_CRITIC, PromptType.DOUBLE_CRITIC_MC]
    ):
        return prompt.eval_correct if prompt.is_correct else (not prompt.eval_correct)
    raise NotImplementedError(
        f"Unexpected prompt type: {prompt.prompt_type} in grading ground truth resolution."
    )


def prepare_grading_tasks(
    cfg: DictConfig,
    tracker: ResponseTracker,
) -> tuple[list[str], list[tuple[ResponseOutput, int, GenerationPrompt | VerificationPrompt]]]:
    """
    Scans the tracker for ungraded responses and prepares grader prompts.

    Returns:
        grading_prompts: List of rendered grading prompts.
        destinations: List of tuples (ResponseOutput object, index) where results should be stored.
    """
    grading_prompts: list[str] = []
    result_destinations: list[tuple[ResponseOutput,
                                    int, GenerationPrompt | VerificationPrompt]] = []

    unique_tasks = set()
    number_tasks = 0
    debug_num_samples = cfg.grading.debug_num_samples

    for prompt, outputs in tracker.data.items():

        if cfg.grading.debug_mode:
            if len(unique_tasks) >= cfg.grading.debug_num_unique_tasks:
                continue
            if debug_num_samples is not None and number_tasks >= debug_num_samples:
                break
            number_tasks += 1
            unique_tasks.add(prompt.task_id)

        if prompt.task_id is None:
            continue

        task = tracker.inference_task_lookup.get(prompt.task_id)
        if task is None:
            continue

        for ro in outputs:
            # If verdicts missing, or shorter than responses (partial fail), re-grade needed
            if not ro.api_verdicts or len(ro.api_verdicts) != len(ro.responses):
                # Initialize slots with placeholders and fill them after batch grading.
                ro.api_verdicts = [
                    APIGradingOutput(
                        extracted_answer=None,
                        is_correct=False,
                        is_valid=False,
                        full_output="PENDING",
                    )
                    for _ in ro.responses
                ]

                for idx, response_text in enumerate(ro.responses):
                    grading_prompt = _build_grading_prompt(
                        cfg=cfg,
                        original_prompt=prompt,
                        model_answer=response_text,
                    )
                    grading_prompts.append(grading_prompt)
                    result_destinations.append((ro, idx, prompt))

    return grading_prompts, result_destinations


def _parse_grading_output(
    raw_output: str,
    original_prompt: GenerationPrompt | VerificationPrompt,
) -> APIGradingOutput:
    """Parse JSON grader output and normalize unsure/invalid semantics."""
    text = raw_output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    response_json = json.loads(text)
    if isinstance(response_json, list) and response_json:
        response_json = response_json[0]
    if not isinstance(response_json, dict):
        raise ValueError(
            f"Unexpected API response type: {type(response_json)}; response text: {raw_output}"
        )

    api_output = APIGradingOutput(
        extracted_answer=response_json.get("extracted_answer", ""),
        is_correct=response_json.get("is_correct", False),
        is_valid=response_json.get("is_valid", False),
        is_unsure=response_json.get("is_unsure", False),
        full_output=raw_output,
    )

    if not api_output.is_valid:
        api_output.is_correct = False

    if api_output.is_unsure:
        api_output.is_valid = True
        api_output.is_correct = determine_correctness_of_unsure_output(
            prompt_is_correct=original_prompt.is_correct,
            prompt_type=original_prompt.prompt_type,
        )

    return api_output


def execute_grading_batch(
    cfg: DictConfig,
    grading_prompts: list[str],
    destinations: list[tuple[ResponseOutput, int, GenerationPrompt | VerificationPrompt]],
) -> int:
    """
    Executes the batch of grading prompts through the LM API and maps results back.

    Returns:
        error_count: Number of API failures.
    """
    if not grading_prompts:
        logger.info("No grading tasks to execute.")
        return 0

    logger.info(
        f"Launching {len(grading_prompts)} grading requests via LM batch API...")

    provider = getattr(cfg.grading, "provider", None)
    if provider in [None, ""]:
        provider = infer_model_provider(cfg.grading.model)

    grader_lm = LMFactory(
        model_name=cfg.grading.model,
        provider=provider,
        config=LMConfig(
            temperature=cfg.grading.temperature,
            max_tokens=cfg.grading.max_tokens,
            reasoning_effort=cfg.grading.reasoning_effort
        ),
        max_concurrent=cfg.grading.max_concurrent_requests,
        max_retries=cfg.grading.max_retries,
        skip_model_lookup=cfg.grading.skip_model_lookup,
        secrets_filepath=cfg.grading.api_key_file_path
    )()
    results = grader_lm(
        grading_prompts,
        return_as_message=False,
        num_samples=1,
    )

    if len(results) != len(destinations):
        logger.warning(
            "Mismatch between grader results and destinations: "
            f"{len(results)} vs {len(destinations)}. Missing items will be marked as errors."
        )

    # Map results back
    error_count = 0
    for out_idx, (ro, idx, original_prompt) in enumerate(destinations):
        if ro.api_verdicts is None or len(ro.api_verdicts) <= idx:
            ro.api_verdicts = [
                APIGradingOutput(
                    extracted_answer=None,
                    is_correct=False,
                    is_valid=False,
                    full_output="PENDING",
                )
                for _ in ro.responses
            ]

        result = results[out_idx]
        if result is None:
            error_count += 1
            ro.api_verdicts[idx] = APIGradingOutput(
                extracted_answer=None,
                is_correct=False,
                is_valid=False,
                full_output="INTERNAL ERROR: empty response from LM grader"
            )
            continue

        try:
            if isinstance(result, OutputAPI):
                if result.error:
                    raise ValueError(result.error)
                raw_output = result.completion
            else:
                raw_output = str(result)

            ro.api_verdicts[idx] = _parse_grading_output(
                raw_output=raw_output,
                original_prompt=original_prompt,
            )

        except Exception as exc:
            error_count += 1
            ro.api_verdicts[idx] = APIGradingOutput(
                extracted_answer=None,
                is_correct=False,
                is_valid=False,
                full_output=f"INTERNAL ERROR: {str(exc)}",
            )
            logger.warning(str(result))

    return error_count


def save_merged_results(
    cfg: DictConfig,
    experiment_result: ExperimentResult,
    tracker: ResponseTracker,
    accelerator: Accelerator,
    epoch: int,
) -> None:
    """Saves the final graded results to a single JSON file."""

    if not accelerator.is_main_process:
        return

    merged_file = Path(cfg.output.dir) / \
        f"merged_results_graded_epoch_{epoch}.json"
    logger.info(f"Saving merged & graded results to: {merged_file}")

    final_result = ExperimentResult(
        model=experiment_result.model,
        inference_tasks=experiment_result.inference_tasks,
        results=[
            ExperimentPromptResults(prompt=p, response_outputs=ros)
            for p, ros in tracker.data.items()
        ]
    )

    with open(merged_file, "w") as f:
        json.dump(final_result, f, indent=2, default=json_default_serializer)


def log_aggregate_stats(
    cfg: DictConfig,
    tracker: ResponseTracker,
    accelerator: Accelerator,
    epoch: int,
) -> None:
    """Calculates accuracy statistics."""

    if not (cfg.experiments_tracker.enable and accelerator.is_main_process):
        return

    total_correct = 0
    total_count = 0
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for prompt, outputs in tracker.data.items():
        for ro in outputs:
            if ro.api_verdicts:
                valid_verdicts = [v for v in ro.api_verdicts if v is not None]
                n_correct = sum(1 for v in valid_verdicts if v.is_correct)
                n_total = len(valid_verdicts)

                total_correct += n_correct
                total_count += n_total

                # Handle enum or string prompt type
                p_type = prompt.prompt_type.value if hasattr(
                    prompt.prompt_type, "value") else str(prompt.prompt_type)
                type_stats[p_type]["correct"] += n_correct
                type_stats[p_type]["total"] += n_total

    global_acc = total_correct / total_count if total_count > 0 else 0.0

    logs = {
        "grading/global_accuracy": global_acc,
        "grading/total_samples": total_count
    }

    for p_type, stats in type_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        logs[f"grading/{p_type}_accuracy"] = acc
        logs[f"grading/{p_type}_count"] = stats["total"]

    for k, v in logs.items():
        log_with_color(
            f"--{k}".ljust(45) + f"{v: .3f}",
            logger,
            color_code=AnsiColors.OKCYAN
            )


def _resolve_results_dir(results_dir: str) -> str:
    """
    Auto-resolve results_dir when it points to a parent of the actual run
    directory.  If the directory already contains ``epoch_*`` subdirectories it
    is returned as-is.  Otherwise the most recently created subdirectory is
    selected (e.g. the latest Hydra timestamped run folder).
    """
    p = Path(results_dir)
    if not p.is_dir():
        return results_dir

    subdirs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not subdirs:
        return results_dir

    if any(d.name.startswith('epoch_') for d in subdirs):
        return results_dir

    latest = max(subdirs, key=lambda d: d.stat().st_ctime)
    log_with_color(
        f"Auto-resolved results_dir to latest subdirectory: {latest}",
        logger,
        AnsiColors.OKCYAN,
    )
    return str(latest)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    if not cfg.grading.enable_grading:
        raise ValueError(
            "Grading is disabled in the configuration. Set 'grading.enable_grading' to True to proceed.")
    if not cfg.grading.use_api:
        raise ValueError(
            "main_grading.py now uses LM API grading. Set 'grading.use_api' to True."
        )

    if cfg.experiments_tracker.enable and accelerator.is_main_process:
        logger.info(
            f"Hello from process {accelerator.process_index}", main_process_only=False)

    # NEW: Auto-resolve results_dir to the latest subdirectory if it points to a parent of the actual run directory.
    results_dir = _resolve_results_dir(cfg.grading.results_dir)

    start_epoch = getattr(cfg.grading, "start_epoch", None)
    end_epoch = getattr(cfg.grading, "end_epoch", None)

    if start_epoch is None or end_epoch is None:
        raise ValueError(
            "Both grading.start_epoch and grading.end_epoch are required for grading."
        )
    if start_epoch > end_epoch:
        raise ValueError(
            "grading.start_epoch must be less than or equal to grading.end_epoch."
        )

    epochs_to_grade = range(start_epoch, end_epoch + 1)

    for epoch in epochs_to_grade:
        log_with_color(
            f"Grading for epoch {epoch}...",
            logger,
            color_code=AnsiColors.OKCYAN
        )
        epoch_results_dir = os.path.join(results_dir, f"epoch_{epoch}")

        logger.info(f"Scanning directory: {epoch_results_dir}")

        try:
            experiment_result = load_and_merge_json_files(epoch_results_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        # 3. Initialize Tracker
        tracker = initialize_tracker(cfg, experiment_result)

        # 4. Run Grading (Flattened Batch)
        grading_prompts, destinations = prepare_grading_tasks(cfg, tracker)

        if grading_prompts:
            error_count = execute_grading_batch(
                cfg, grading_prompts, destinations)
            logger.info(f"Grading complete. Total API Errors: {error_count}")
        else:
            logger.info("No ungraded responses found. Skipping API calls.")

        # 5. Save Results
        save_merged_results(cfg, experiment_result,
                            tracker, accelerator, epoch=epoch)

        # 6. Log Stats
        log_aggregate_stats(cfg, tracker, accelerator, epoch=epoch)


if __name__ == "__main__":    
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
