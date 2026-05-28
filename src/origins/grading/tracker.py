import os
import json
from collections import defaultdict
import dataclasses
from typing import Optional

import logging
from accelerate import Accelerator
from omegaconf import DictConfig

from origins.custom_classes import (
    _prompt_category,
    InferenceTask,
    ResponseOutput,
    ExperimentResult,
    ExperimentPromptResults,
)
from origins.prompts.prompt_utils import (
    get_parse_keys_with_types,
    parse_text_in_tags,
)
from origins.utils.general_utilities import json_default_serializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseTracker:
    def __init__(
        self,
        cfg: DictConfig,
        prompts: list[_prompt_category],
        inference_tasks: list[InferenceTask],
    ):
        """
        Initialize the response tracker.

        Args:
            cfg (DictConfig): Hydra config
            prompts (list[_prompt_category]): List of prompts being tracked
            inference_tasks (list[InferenceTask]): Inference tasks associated
                with the prompts; saved alongside results so the grader can
                reconstruct task metadata.
        """
        self.cfg = cfg
        self.model_name = cfg.model.name
        self.inference_tasks = inference_tasks
        self.prompts = prompts
        self.data: defaultdict[_prompt_category,
                               list[ResponseOutput]] = defaultdict(list)
        self.results_file = cfg.output.results_file
        self.results_file_path = os.path.join(
            cfg.output.dir, self.results_file)

        # Lookup table consumed by `main_grading.py` to map prompts back to
        # their originating task at grading time.
        self.inference_task_lookup = {
            task.task_id: task for task in self.inference_tasks
        }

    def register_responses_only(
        self,
        responses: dict[_prompt_category, list[str]],
        iteration: int,
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        """
        Register responses and parse them locally. Grading happens later in
        `main_grading.py`, so verdicts are intentionally left empty here.

        Args:
            responses (dict): Dictionary mapping prompt objects to lists of response strings.
            iteration (int): The training iteration (epoch) the responses belong to.
            accelerator (Accelerator, optional): Unused; kept for backward compatibility.
        """
        for prompt, response_list in responses.items():
            keys_with_types = get_parse_keys_with_types(
                prompt_type=prompt.prompt_type,
                include_reasoning=prompt.include_reasoning,
            )
            parsed_responses = [
                parse_text_in_tags(
                    keys_with_types=keys_with_types, text_to_search=response)
                for response in response_list
            ]

            self.data[prompt].append(
                ResponseOutput(
                    responses=response_list,
                    parsed_responses=parsed_responses,
                    iteration=iteration,
                    api_verdicts=[],  # Empty list indicates "Not Graded Yet"
                    char_lengths=[len(response) for response in response_list],
                )
            )

    def save_to_file(
        self,
        accelerator: Optional[Accelerator] = None,
        batch_num: Optional[int] = None,
        epoch_subdir: Optional[int] = None,
    ) -> None:
        """
        Save the tracked responses to a JSON file.
        Each process saves to a unique file: results_rank_0.json, results_rank_1.json, etc.

        Args:
            accelerator (Accelerator, optional): Accelerator object for distributed setups.
            batch_num (int, optional): If batching multiple runs, the batch number.
            epoch_subdir (int, optional): If set, nest the file under `epoch_<n>/`.
        """
        filename, ext = os.path.splitext(self.results_file)

        if accelerator is None:
            logger.warning(
                "No accelerator provided to ResponseTracker.save_to_file; assuming main process.")
            accelerator_process_index = 0
        else:
            accelerator_process_index = accelerator.process_index

        if batch_num is not None:
            rank_filename = f"{filename}_batch_{batch_num}_rank{accelerator_process_index}{ext}"
        else:
            rank_filename = f"{filename}_rank{accelerator_process_index}{ext}"

        if epoch_subdir is not None:
            rank_filename = os.path.join(
                f"epoch_{epoch_subdir}", rank_filename)
        rank_file_path = os.path.join(self.cfg.output.dir, rank_filename)

        logger.info(
            f"Process {accelerator_process_index}: Saving results to {rank_filename}")

        # Strip per-task train_sentences from the saved metadata to keep the
        # output file small; dataclasses.replace yields a fresh instance so
        # self.inference_tasks itself is untouched.
        tasks_for_saving = [
            dataclasses.replace(task, train_sentences=None)
            for task in self.inference_tasks
        ]

        experiment_result = ExperimentResult(
            model=self.model_name,
            inference_tasks=tasks_for_saving,
            results=[
                ExperimentPromptResults(prompt=key, response_outputs=value)
                for key, value in self.data.items()
            ],
        )

        os.makedirs(os.path.dirname(
            os.path.abspath(rank_file_path)), exist_ok=True)

        with open(rank_file_path, "w") as f:
            json.dump(experiment_result, f, indent=2,
                      default=json_default_serializer)

        logger.info(f"Results saved to {rank_file_path}")
