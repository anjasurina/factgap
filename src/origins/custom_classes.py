import dataclasses
from dataclasses import dataclass, field
import functools
import numpy as np
import enum
import uuid
from typing import Optional, Any, TypeAlias, Union
from enum import Enum
from collections import defaultdict
from typing import Dict, Tuple


@enum.unique
class Phase(Enum):
    LEARN = "learn"
    UPDATE = "update"
    FORGET = "forget"
    INFER_ONLY = "infer_only"
    VALIDATION_ONLY = "validation_only"


@dataclasses.dataclass
class PhaseConfig:
    phase: Phase
    starting_epoch: int = 0
    start_from_checkpoint: str = ""
    num_epochs: int = 0
    save_model_at_every_epoch: bool = False
    evaluate_on_train_begin: bool = True
    save_model_at_the_end: bool = True


@dataclasses.dataclass
class StatsClass:
    correct_outputs_raw: list[Any] = dataclasses.field(default_factory=list)
    control_outputs_raw: list[Any] = dataclasses.field(default_factory=list)
    correct_outputs_parsed: list[Any] = dataclasses.field(default_factory=list)
    control_outputs_parsed: list[Any] = dataclasses.field(default_factory=list)
    correct_outputs_correct: list[float] = dataclasses.field(
        default_factory=list)
    control_outputs_correct: list[float] = dataclasses.field(
        default_factory=list)
    correct_outputs_char_lengths: list[int] = dataclasses.field(
        default_factory=list)
    control_outputs_char_lengths: list[int] = dataclasses.field(
        default_factory=list)
    correct_support: int = 0
    control_support: int = 0
    correct_support_all: int = 0
    control_support_all: int = 0
    correct_outputs_raw_api: list[Any] = dataclasses.field(
        default_factory=list)
    control_outputs_raw_api: list[Any] = dataclasses.field(
        default_factory=list)
    correct_outputs_parsed_api: list[Any] = dataclasses.field(
        default_factory=list)
    control_outputs_parsed_api: list[Any] = dataclasses.field(
        default_factory=list)
    correct_outputs_correct_api: list[float] = dataclasses.field(
        default_factory=list)
    control_outputs_correct_api: list[float] = dataclasses.field(
        default_factory=list)
    correct_support_api: int = 0
    control_support_api: int = 0
    correct_support_all_api: int = 0
    control_support_all_api: int = 0
    correct_unsure: int = 0  # Number of unsure responses for correct prompts
    control_unsure: int = 0  # Number of unsure responses for control prompts
    # Number of unsure responses for correct prompts from API grading
    correct_unsure_api: int = 0
    # Number of unsure responses for control prompts from API grading
    control_unsure_api: int = 0

    @functools.cached_property
    def correct_mean(self):
        return np.mean(self.correct_outputs_correct) if len(self.correct_outputs_correct) > 0 else 0

    @functools.cached_property
    def correct_mean_api(self):
        return np.mean(self.correct_outputs_correct_api) if len(self.correct_outputs_correct_api) > 0 else 0

    @functools.cached_property
    def control_mean(self):
        return np.mean(self.control_outputs_correct) if len(self.control_outputs_correct) > 0 else 0

    @functools.cached_property
    def control_mean_api(self):
        return np.mean(self.control_outputs_correct_api) if len(self.control_outputs_correct_api) > 0 else 0

    @functools.cached_property
    def correct_std(self):
        return np.std(self.correct_outputs_correct) if len(self.correct_outputs_correct) > 1 else 0

    @functools.cached_property
    def correct_std_api(self):
        return np.std(self.correct_outputs_correct_api) if len(self.correct_outputs_correct_api) > 1 else 0

    @functools.cached_property
    def control_std(self):
        return np.std(self.control_outputs_correct) if len(self.control_outputs_correct) > 1 else 0

    @functools.cached_property
    def control_std_api(self):
        return np.std(self.control_outputs_correct_api) if len(self.control_outputs_correct_api) > 1 else 0

    @functools.cached_property
    def correct_support_percent(self):
        return self.correct_support / self.correct_support_all if self.correct_support_all > 0 else 0

    @functools.cached_property
    def correct_support_percent_api(self):
        return self.correct_support_api / self.correct_support_all if self.correct_support_all > 0 else 0

    @functools.cached_property
    def control_support_percent(self):
        return self.control_support / self.control_support_all if self.control_support_all > 0 else 0

    @functools.cached_property
    def control_support_percent_api(self):
        return self.control_support_api / self.control_support_all if self.control_support_all > 0 else 0

    @functools.cached_property
    def correct_sem(self, zval=1.96):
        return self.correct_std / np.sqrt(len(self.correct_outputs_correct)) * zval if len(self.correct_outputs_correct) > 1 else 0

    @functools.cached_property
    def correct_sem_api(self, zval=1.96):
        return self.correct_std_api / np.sqrt(len(self.correct_outputs_correct_api)) * zval if len(self.correct_outputs_correct_api) > 1 else 0

    @functools.cached_property
    def control_sem(self, zval=1.96):
        return self.control_std / np.sqrt(len(self.control_outputs_correct)) * zval if len(self.control_outputs_correct) > 1 else 0

    @functools.cached_property
    def control_sem_api(self, zval=1.96):
        return self.control_std_api / np.sqrt(len(self.control_outputs_correct_api)) * zval if len(self.control_outputs_correct_api) > 1 else 0

    @functools.cached_property
    def correct_mean_diff(self):
        """Calculate the difference in means between manual and API grading for correct prompts"""
        if len(self.correct_outputs_correct) == 0 or len(self.correct_outputs_correct_api) == 0:
            return 0
        mean_manual = self.correct_mean
        mean_api = self.correct_mean_api
        return abs(mean_manual - mean_api)

    @functools.cached_property
    def control_mean_diff(self):
        """Calculate the difference in means between manual and API grading for control prompts"""
        if len(self.control_outputs_correct) == 0 or len(self.control_outputs_correct_api) == 0:
            return 0
        mean_manual = self.control_mean
        mean_api = self.control_mean_api
        return abs(mean_manual - mean_api)

    @functools.cached_property
    def correct_support_disagreement_percent(self):
        """Calculate the percentage disagreement in support between manual and API grading for correct prompts"""
        if self.correct_support_all == 0:
            return 0
        support_manual = self.correct_support
        support_api = self.correct_support_api
        return abs(support_manual - support_api) / self.correct_support_all

    @functools.cached_property
    def control_support_disagreement_percent(self):
        """Calculate the percentage disagreement in support between manual and API grading for control prompts"""
        if self.control_support_all == 0:
            return 0
        support_manual = self.control_support
        support_api = self.control_support_api
        return abs(support_manual - support_api) / self.control_support_all

    @functools.cached_property
    def correct_unsure_percent(self):
        # Calculated from valid responses only
        return self.correct_unsure / self.correct_support if self.correct_support > 0 else 0

    @functools.cached_property
    def correct_unsure_percent_api(self):
        # Calculated from valid responses only
        return self.correct_unsure_api / self.correct_support_api if self.correct_support_api > 0 else 0

    @functools.cached_property
    def control_unsure_percent(self):
        # Calculated from valid responses only
        return self.control_unsure / self.control_support if self.control_support > 0 else 0

    @functools.cached_property
    def control_unsure_percent_api(self):
        # Calculated from valid responses only
        return self.control_unsure_api / self.control_support_api if self.control_support_api > 0 else 0

    def to_dict(self):
        return {**dataclasses.asdict(self),
                "correct_mean": self.correct_mean,
                "control_mean": self.control_mean,
                "correct_std": self.correct_std,
                "control_std": self.control_std,
                "correct_support_percent": self.correct_support_percent,
                "control_support_percent": self.control_support_percent,
                "correct_mean_api": self.correct_mean_api,
                "control_mean_api": self.control_mean_api,
                "correct_std_api": self.correct_std_api,
                "control_std_api": self.control_std_api,
                "correct_support_percent_api": self.correct_support_percent_api,
                "control_support_percent_api": self.control_support_percent_api,
                "correct_unsure": self.correct_unsure,
                "control_unsure": self.control_unsure,
                "correct_unsure_api": self.correct_unsure_api,
                "control_unsure_api": self.control_unsure_api,
                "correct_unsure_percent": self.correct_unsure_percent,
                "control_unsure_percent": self.control_unsure_percent,
                "correct_unsure_percent_api": self.correct_unsure_percent_api,
                "control_unsure_percent_api": self.control_unsure_percent_api,
                }


@dataclasses.dataclass
class StatsRegistry:
    """Hierarchical registry that stores one StatsClass per
    (task_id, phase, template_name, version).

    Internal structure:
    data[task_id][phase][template_name][version] -> StatsClass
    """
    data: dict[str, dict[str, dict[str, dict[str, StatsClass]]]] = dataclasses.field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict)))
    )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def get(self, task_id: str, phase: str, template_name: str, version: str) -> StatsClass:
        """Return the StatsClass for the given 4-tuple, creating it lazily."""
        phase_map = self.data[task_id][phase][template_name]
        if version not in phase_map:
            phase_map[version] = StatsClass()
        return phase_map[version]

    # alias so existing code that expects .stats_objects[key] can migrate easily
    def __getitem__(self, key: tuple[str, str, str, str]) -> StatsClass:
        task_id, phase, template_name, version = key
        return self.get(task_id, phase, template_name, version)

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------
    def iter_items(self):
        """Yield ((task_id, phase, template, version), StatsClass)."""
        for task_id, phase_map in self.data.items():
            for phase, tmpl_map in phase_map.items():
                for tmpl, ver_map in tmpl_map.items():
                    for ver, stats in ver_map.items():
                        yield (task_id, phase, tmpl, ver), stats

    def items(self):  # for API parity with dict
        return self.iter_items()

    def keys(self):
        """Return an iterable of composite keys."""
        for key, _ in self.iter_items():
            yield key

    def values(self):
        """Return an iterable of StatsClass values."""
        for _, value in self.iter_items():
            yield value

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.iter_items())

    # ------------------------------------------------------------------
    # Filtering helpers – return *new* StatsRegistry with matching items
    # ------------------------------------------------------------------
    def _copy_if_match(self, matcher):
        new = StatsRegistry()
        for (task_id, phase, tmpl, ver), stats in self.iter_items():
            if matcher(task_id, phase, tmpl, ver):
                new.get(task_id, phase, tmpl, ver).__dict__.update(
                    dataclasses.asdict(stats))
        return new

    def filter_by_phase(self, phase: str) -> "StatsRegistry":
        return self._copy_if_match(lambda _tid, ph, _t, _v: ph == phase)

    def filter_by_task(self, task_id: str) -> "StatsRegistry":
        return self._copy_if_match(lambda tid, _ph, _t, _v: tid == task_id)

    def filter_by_template(self, template_name: str) -> "StatsRegistry":
        return self._copy_if_match(lambda _tid, _ph, tmpl, _v: tmpl == template_name)

    def filter_by_version(self, version: str) -> "StatsRegistry":
        return self._copy_if_match(lambda _tid, _ph, _t, ver: ver == version)

    def get_all_task_ids(self) -> list[str]:
        """Return all unique task_ids."""
        return list(set(task_id for task_id, _, _, _ in self.iter_items()))

    def get_all_phases(self) -> list[str]:
        """Return all unique phases."""
        return list(set(phase for _, phase, _, _ in self.iter_items()))

    def get_all_template_names(self) -> list[str]:
        """Return all unique template names."""
        return list(set(template_name for _, _, template_name, _ in self.iter_items()))

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def to_flat_dict(self) -> dict[tuple[str, str, str, str], StatsClass]:
        """Return a flat dict with the composite key for easier export."""
        return {key: stats for key, stats in self.iter_items()}


@enum.unique
class PromptType(enum.Enum):
    """
    Enum to define the type of prompt.
    """
    GENERATIVE_FREE = "generative_free"
    GENERATIVE_MC = "generative_mc"
    DOUBLE_CRITIC = "double_critic"
    DOUBLE_CRITIC_MC = "double_critic_mc"


def lookup_prompt_type(template_name: str) -> PromptType:
    """
    Lookup the prompt type based on the template name.

    Args:
        template_name (str): The name of the template.

    Returns:
        PromptType: The corresponding PromptType enum value.

    Raises:
        NotImplementedError: If the template name is not recognized.
    """
    if template_name in ["double_critic.j2"]:
        return PromptType.DOUBLE_CRITIC
    elif template_name in ["double_critic_mc.j2"]:
        return PromptType.DOUBLE_CRITIC_MC
    elif template_name in ["generative_response.j2"]:
        return PromptType.GENERATIVE_FREE
    elif template_name in ["generative_response_mc.j2"]:
        return PromptType.GENERATIVE_MC
    else:
        raise NotImplementedError(f"Unknown template name: {template_name}")


@dataclasses.dataclass(frozen=True)
class VerificationPrompt:
    prompt_text: str
    is_correct: bool
    eval_correct: bool
    template_name: str
    prompt_type: PromptType
    phase: Phase = Phase.LEARN
    correct_answer_letter: Optional[str] = None
    correct_answer: Optional[bool] = None
    include_reasoning: bool = True
    task_id: Optional[str] = None
    version: Optional[str] = None
    # The raw problem statement without any additional formatting instructions. Useful for API grading.
    problem_statement: str = ""
    allow_unsure: bool = False
    base_model_prompt: bool = False

    @functools.cached_property
    def is_control(self) -> bool:
        """Returns the opposite of is_correct"""
        return not self.is_correct

    @functools.cached_property
    def min_repr(self) -> str:
        return f"VerificationPrompt(is_correct={self.is_correct}, eval_correct={self.eval_correct}, template_name={self.template_name})"

    @functools.cached_property
    def is_generation_prompt(self) -> bool:
        return False

    @functools.cached_property
    def is_verification_prompt(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class GenerationPrompt:
    prompt_text: str
    is_correct: bool
    template_name: str
    prompt_type: PromptType
    phase: Phase
    correct_answer_letter: Optional[str] = None
    correct_answer: Optional[bool] = None
    include_reasoning: bool = True
    task_id: Optional[str] = None
    version: Optional[str] = None
    # The raw problem statement without any additional formatting instructions. Useful for API grading.
    problem_statement: str = ""
    base_model_prompt: bool = False

    @functools.cached_property
    def is_control(self) -> bool:
        """Returns the opposite of is_correct"""
        return not self.is_correct

    @functools.cached_property
    def min_repr(self) -> str:
        return f"GenerationPrompt(is_correct={self.is_correct}, template_name={self.template_name})"

    @functools.cached_property
    def is_generation_prompt(self) -> bool:
        return True

    @functools.cached_property
    def is_verification_prompt(self) -> bool:
        return False


_prompt_category: TypeAlias = Union[GenerationPrompt, VerificationPrompt]


@dataclasses.dataclass
class APIGradingOutput:
    """
    Grading response from an API.
    """
    extracted_answer: str | None
    is_correct: bool
    is_valid: bool
    is_unsure: Optional[bool] = False
    full_output: Optional[str] = None

    @property
    def valid_response(self) -> bool:
        return self.extracted_answer is not None


@dataclasses.dataclass
class ResponseOutput:
    responses: list[str]
    parsed_responses: list[dict]
    iteration: int
    api_verdicts: Optional[list[APIGradingOutput]] = None
    char_lengths: Optional[list[int]] = None


@dataclasses.dataclass
class InferenceProblem:
    problem: str
    answer: Optional[str] = None
    version: Optional[str] = None


@dataclasses.dataclass
class InferenceTrainDataPoint:
    sentence: str
    version: Optional[str] = None
    scenario_key: Optional[str] = None


@dataclasses.dataclass
class TrainDataPoint:
    sentence: str
    relationship_head: str | None = None
    topic: str | None = None
    scenario_key: str | None = None


@dataclasses.dataclass
class InferenceTask:
    correct_problems: list[InferenceProblem]
    control_problems: list[InferenceProblem]
    control_answers: list[str]
    train_sentences: Optional[list[InferenceTrainDataPoint]] = None
    task_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    correct_problems_versions: Optional[list[str]] = None
    control_problems_versions: Optional[list[str]] = None
    train_sentences_versions: Optional[list[str]] = None
    phase: Phase = Phase.LEARN
    relationship_head: str | None = None
    topic: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceTask":
        """
        Create an InferenceTask from a dictionary.
        """

        correct_problems = []
        if "correct_problems" in data:
            for problem in data["correct_problems"]:
                correct_problems.append(InferenceProblem(**problem))

        control_problems = []
        if "control_problems" in data:
            for problem in data["control_problems"]:
                control_problems.append(InferenceProblem(**problem))

        train_sentences = []
        if "train_sentences" in data:
            data_train_sentences = data["train_sentences"]
            if isinstance(data_train_sentences, list):
                for sentence in data_train_sentences:
                    train_sentences.append(InferenceTrainDataPoint(**sentence))

        return cls(
            correct_problems=correct_problems,
            control_problems=control_problems,
            control_answers=data.get("control_answers", []),
            train_sentences=train_sentences,
            task_id=data.get('task_id', str(uuid.uuid4())),
            correct_problems_versions=data.get(
                "correct_problems_versions", None),
            control_problems_versions=data.get(
                "control_problems_versions", None),
            train_sentences_versions=data.get(
                "train_sentences_versions", None),
            phase=data.get("phase", ""),
            relationship_head=data.get("relationship_head", None),
            topic=data.get("topic", None)

        )


@dataclasses.dataclass
class ExperimentPromptResults:
    """
    Class to store the experimental results of a prompt.
    """

    prompt: _prompt_category
    response_outputs: list[ResponseOutput]


@dataclasses.dataclass
class ExperimentResult:
    """
    Class to store the results of an experiment.
    """

    model: str
    inference_tasks: list[InferenceTask]
    results: list[ExperimentPromptResults]


@dataclasses.dataclass
class PromptParsedResults:
    """
    Class to store the parsed results of a prompt.
    """

    prompt: _prompt_category
    parsed_responses: list[dict]
    responses: list[str] = dataclasses.field(default_factory=list)
    api_verdicts: Optional[list[APIGradingOutput]] = None
    char_lengths: Optional[list[int]] = None


@dataclass
class MockAccelerator:
    """
    Minimal stand-in for Accelerate's Accelerator.

    ResponseTracker expects an accelerator-like object to coordinate
    grading/logging. In vLLM inference we run in a single process, so a
    lightweight mock avoids importing or initializing accelerate.
    """
    process_index: int = 0
    is_main_process: bool = True
    num_processes: int = 1

    def wait_for_everyone(self):
        """No-op barrier for single-process inference."""
        pass
