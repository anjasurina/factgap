"""
Synthetic Data Generator workflow:
1. Define topics and initialize a global ForbiddenImaginaryRegistry.
2. For each datapoint, sequentially generate: Ingredients -> Instantiations -> Sentences -> Tasks.
3. Automatically retry generation steps if the LLM output is malformed (FormatRetryError).
4. Maintain lists of forbidden imaginary names across threads to prevent duplicates within and across topics.
5. Save the generated `SyntheticDataPoint` structures atomically to local YAML files.
"""
import os
import re
import time
import random
import logging
import functools
import threading
from typing import Sequence, Callable, Any
from uuid import uuid4
from datetime import datetime as dt
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import fire
from tqdm import tqdm

from ..models_api.inference import LMFactory, LM
from ..models_api.adapters import _ReasoningEffort, LMConfig, OutputAPI
from ..utils.reusable_classes import *
from ..utils.prompt_helper import *
from .check_synth_data import (
    find_imaginary_duplicates,
    format_duplicates,
    SOURCE_PATH_KEY,
    DEFAULT_IGNORE_WORDS,
)

_MIN_FORBIDDEN_TOKEN_LEN = 3
_TOKEN_RE = re.compile(r"\b[a-z]+\b")

_SAVE_FOLDER = "data/synthetic_data"
_DUPLICATE_REPORT_FILENAME = "duplicate_report.txt"

logger = logging.getLogger(__name__)


@dataclass
class Ingredients:
    topic: str
    relationship: str

    @classmethod
    def from_dict(cls, data: dict) -> "Ingredients":
        return cls(topic=data["topic"], relationship=data["relationship"])


@dataclass
class IngredientInstantiations:
    real_pairs: Optional[list[dict]]
    imaginary_pairs: Optional[list[dict]]

    @classmethod
    def from_dict(cls, data: dict) -> "IngredientInstantiations":
        return cls(
            real_pairs=data.get("real_pairs", []),
            imaginary_pairs=data.get("imaginary_pairs", []),
        )


@dataclass
class SyntheticDataPoint:
    """Structured in-memory representation of one generated datapoint.

    Convert to the on-disk yaml shape with :meth:`to_yaml_dict`. The saved
    structure intentionally matches the legacy schema so downstream consumers
    (training, inference) are unaffected.
    """

    uid: str
    topic_category: str
    topic: str
    relationship: str
    sentences: list[dict]
    instantiations: dict
    questions: list[dict]
    model_name: Optional[str] = None

    def to_yaml_dict(self) -> dict:
        d = asdict(self)
        if d.get("model_name") is None:
            d.pop("model_name", None)
        return d


@dataclass
class TopicGenerationResult:
    topic: str
    datapoints: list[dict] = field(default_factory=list)
    num_failures: int = 0
    failure_messages: list[str] = field(default_factory=list)


class ForbiddenImaginaryRegistry:
    """Thread-safe registry of imaginary head/tail pairs forbidden across all topics.

    Tracks both the original pair structure and a flat set of *tokens* extracted
    from those pairs. Token-level forbidding is what the LLM actually needs:
    duplicate fantasy names overwhelmingly come from reused roots and surnames
    (e.g., "Vance" recurring across "Elias Vance", "Julian Vance",
    "Dr. Elara Vance"), not from full pair collisions.
    """

    def __init__(self, ignore_words: Optional[set[str]] = None):
        self._pairs: list[dict] = []
        self._seen: set[tuple] = set()
        self._tokens: set[str] = set()
        self._ignore_words = (
            ignore_words if ignore_words is not None else DEFAULT_IGNORE_WORDS
        )
        self._lock = threading.Lock()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._pairs)

    def tokens(self) -> list[str]:
        with self._lock:
            return sorted(self._tokens)

    def _extract_tokens(self, pair: dict) -> set[str]:
        out: set[str] = set()
        for value in (pair.get("head") or "", pair.get("tail") or ""):
            for tok in _TOKEN_RE.findall(value.lower()):
                if len(tok) < _MIN_FORBIDDEN_TOKEN_LEN or tok in self._ignore_words:
                    continue
                out.add(tok)
        return out

    def add(self, pairs: list[dict]) -> None:
        with self._lock:
            for p in pairs:
                key = (p.get("head"), p.get("tail"))
                if key in self._seen:
                    continue
                self._seen.add(key)
                self._pairs.append(p)
                self._tokens.update(self._extract_tokens(p))

    def try_add(self, pairs: list[dict]) -> set[str]:
        """Atomically check for token conflicts and commit if there are none.

        Returns the set of tokens from ``pairs`` that conflict with already-stored
        tokens. Empty set means the commit succeeded. The check-and-commit happens
        under the lock so two parallel topic workers cannot both pass the gate
        with overlapping fresh tokens.
        """
        proposed = set()
        for p in pairs:
            proposed |= self._extract_tokens(p)

        with self._lock:
            conflict = proposed & self._tokens
            if conflict:
                return conflict
            for p in pairs:
                key = (p.get("head"), p.get("tail"))
                if key in self._seen:
                    continue
                self._seen.add(key)
                self._pairs.append(p)
            self._tokens.update(proposed)
            return set()


def _extract_content_tokens(text: str, ignore: set[str]) -> set[str]:
    return {
        tok for tok in _TOKEN_RE.findall((text or "").lower())
        if len(tok) >= _MIN_FORBIDDEN_TOKEN_LEN and tok not in ignore
    }


def _structural_tokens(ingredients: Ingredients) -> set[str]:
    """Tokens drawn from the topic/relationship themselves.

    The topic noun naturally recurs in every tail (e.g., topic "gland" appears in
    "pineal gland", "adrenal gland", ...). These tokens must be excluded from the
    uniqueness check so we don't penalize structurally-required repetition.
    """
    out: set[str] = set()
    for text in (ingredients.topic, ingredients.relationship):
        for tok in _TOKEN_RE.findall((text or "").lower()):
            if len(tok) >= _MIN_FORBIDDEN_TOKEN_LEN:
                out.add(tok)
    return out


def _validate_imaginary_uniqueness(
    instantiations: dict,
    ingredients: Ingredients,
) -> None:
    """Raise ValueError on within-output violations.

    Checks (both per single model output):
      1. **Cross-pair token reuse**: no content token may appear in more than one
         imaginary pair. Head+tail of the *same* pair are deliberately allowed to
         share a token (e.g., "Voron" head + "Voron's Field" tail).
      2. **Degenerate output**: each imaginary pair's head and tail must contain
         at least one non-structural content token. If a tail is purely the
         topic noun restated (e.g., topic "judgmental heuristics" → tail
         "judgmental heuristics"), the model has emitted no real instantiation.

    Cross-call (cross-datapoint) duplicate checking is handled separately by the
    atomic ``ForbiddenImaginaryRegistry.try_add`` gate.
    """
    imaginary_pairs = instantiations.get("imaginary_pairs", []) or []
    ignore = DEFAULT_IGNORE_WORDS | _structural_tokens(ingredients)

    token_to_pair: dict[str, int] = {}
    within_dupes: set[str] = set()
    degenerate_fields: list[str] = []

    for i, pair in enumerate(imaginary_pairs):
        head_tokens = _extract_content_tokens(pair.get("head") or "", ignore)
        tail_tokens = _extract_content_tokens(pair.get("tail") or "", ignore)
        if not head_tokens:
            degenerate_fields.append(
                f"pair {i + 1} head ({pair.get('head')!r})")
        if not tail_tokens:
            degenerate_fields.append(
                f"pair {i + 1} tail ({pair.get('tail')!r})")

        for tok in head_tokens | tail_tokens:
            if tok in token_to_pair and token_to_pair[tok] != i:
                within_dupes.add(tok)
            else:
                token_to_pair[tok] = i

    issues = []
    if within_dupes:
        issues.append(
            f"token(s) repeated across imaginary pairs in this output: {sorted(within_dupes)}"
        )
    if degenerate_fields:
        issues.append(
            "imaginary pair(s) lack a non-structural content token "
            f"(model restated the topic instead of instantiating it): {degenerate_fields}"
        )
    if issues:
        raise ValueError("; ".join(issues))


################################################################################


class FormatRetryError(RuntimeError):
    """Raised when a generation step fails repeatedly to produce a parseable response."""


def retry_on_format_error(default_attempts: int = 3) -> Callable:
    """Retry a generation step when the model returns malformed/empty output.

    API-level retries (rate limits, network) are handled inside the LM class; this
    decorator covers the layer above: YAML parse errors, missing required keys,
    and empty responses. Each decorated function must declare a
    ``max_format_retries: Optional[int] = None`` keyword so static type checkers
    see the parameter; the decorator reads and consumes it before calling the
    inner function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            max_format_retries = kwargs.pop("max_format_retries", None)
            attempts = max_format_retries if max_format_retries is not None else default_attempts
            last_exc: Optional[BaseException] = None
            for attempt in range(1, attempts + 1):
                try:
                    result = func(*args, **kwargs)
                except (ValueError, KeyError, TypeError, yaml.YAMLError) as e:
                    last_exc = e
                    logger.warning(
                        "format error in %s (attempt %d/%d): %s",
                        func.__name__, attempt, attempts, e,
                    )
                    continue
                if result:
                    return result
                last_exc = FormatRetryError(
                    f"{func.__name__} returned empty result")
                logger.warning(
                    "empty result from %s (attempt %d/%d)",
                    func.__name__, attempt, attempts,
                )
            raise FormatRetryError(
                f"{func.__name__} failed after {attempts} format-retry attempts: {last_exc}"
            )
        return wrapper
    return decorator


################################################################################


def parse_response(
    response: Sequence[OutputAPI | None],
    expected_keys: Optional[list[str]] = None,
) -> list[dict | None]:
    """Parse a sequence of model responses as YAML dicts."""
    yaml_responses: list[dict | None] = []
    if isinstance(response, list) and len(response) > 0:
        for output in response:
            if isinstance(output, OutputAPI):
                yaml_responses.append(
                    parse_yaml_response(output.completion,
                                        expected_keys=expected_keys)
                )
            else:
                yaml_responses.append(None)
    else:
        yaml_responses = [None]
    return yaml_responses


@retry_on_format_error()
def _generate_ingredients(
    model: LM,
    topic_category: str,
    forbidden_ingredients: Optional[list] = None,
    max_format_retries: Optional[int] = None,
) -> Ingredients:
    prompt = render_template(
        template_name="synthetic_ingredients.j2",
        user_data={
            "topic_category": topic_category,
            "forbidden_ingredients": forbidden_ingredients,
        },
    ).strip()
    response = model(prompt, return_as_message=False, num_samples=1)
    yaml_response = parse_response(response)

    if len(yaml_response) != 1 or not isinstance(yaml_response[0], dict):
        raise ValueError("Invalid response format for ingredients.")
    return Ingredients.from_dict(yaml_response[0]["ingredients"])


@retry_on_format_error()
def _generate_synthetic_instantiations(
    model: LM,
    ingredients: Ingredients,
    num_instantiations: int = 5,
    forbidden_imaginary: Optional[ForbiddenImaginaryRegistry] = None,
    max_format_retries: Optional[int] = None,
) -> dict:
    # Snapshot tokens for the prompt immediately before generating; the atomic
    # commit below closes the remaining race window.
    forbidden_tokens_snapshot = (
        forbidden_imaginary.tokens() if forbidden_imaginary is not None else []
    )

    prompt = render_template(
        template_name="synthetic_instantiations.j2",
        user_data={
            "topic": ingredients.topic,
            "relationship": ingredients.relationship,
            "num_instantiations": num_instantiations,
            "forbidden_imaginary_tokens": forbidden_tokens_snapshot,
        },
    ).strip()

    response = model(prompt, return_as_message=False, num_samples=1)
    yaml_response = parse_response(response)
    if len(yaml_response) != 1 or not isinstance(yaml_response[0], dict):
        raise ValueError("Invalid response format for instantiations.")
    instantiations = yaml_response[0]["instantiations"]

    _validate_imaginary_uniqueness(instantiations, ingredients=ingredients)

    if forbidden_imaginary is not None:
        conflicts = forbidden_imaginary.try_add(
            instantiations.get("imaginary_pairs", []) or []
        )
        if conflicts:
            raise ValueError(
                f"atomic commit failed; conflicting token(s) with another "
                f"concurrent worker: {sorted(conflicts)}"
            )

    return instantiations


@retry_on_format_error()
def _generate_synthetic_sentences(
    model: LM,
    ingredients: Ingredients,
    num_sentences: int,
    instantiations: Optional[list[dict]] = None,
    max_format_retries: Optional[int] = None,
) -> list[str]:
    prompt = render_template(
        template_name="synthetic_sentences.j2",
        user_data={
            "topic": ingredients.topic,
            "relationship": ingredients.relationship,
            "num_sentences": num_sentences,
            "instantiations": instantiations,
        },
    ).strip()

    response = model(prompt, return_as_message=False, num_samples=1)
    yaml_response = parse_response(response)
    if len(yaml_response) != 1 or not isinstance(yaml_response[0], dict):
        raise ValueError("Invalid response format for sentences.")
    return yaml_response[0]["sentences"]


@retry_on_format_error()
def _generate_synthetic_tasks(
    model: LM,
    ingredients: Ingredients,
    sentences: list[str],
    num_tasks: int = 5,
    instantiations: Optional[list[dict]] = None,
    max_format_retries: Optional[int] = None,
) -> list[dict]:
    prompt = render_template(
        template_name="synthetic_tasks.j2",
        user_data={
            "training_sentences": sentences,
            "topic": ingredients.topic,
            "relationship": ingredients.relationship,
            "num_tasks": num_tasks,
            "instantiations": instantiations,
        },
    ).strip()

    response = model(prompt, return_as_message=False, num_samples=1)
    yaml_response = parse_response(response)
    if len(yaml_response) != 1 or not isinstance(yaml_response[0], dict):
        raise ValueError("Invalid response format for tasks.")
    return yaml_response[0]["questions"]


def _clean_sentences(
    sentences: list[str],
    topic: str,
    relationship_head: str,
) -> list[str]:
    cleaned = []
    for s in sentences:
        s = s.lower()
        s = s.replace("{" + topic + "}", "{topic}")
        s = s.replace("{" + relationship_head + "}", "{relationship_head}")
        cleaned.append(s)
    return cleaned


def _clean_tasks(tasks: list[dict], topic: str) -> list[dict]:
    cleaned = []
    for task in tasks:
        q = task["question"].lower().replace("{" + topic + "}", "{topic}")
        a = task["answer"].lower()
        cleaned.append({"question": q, "answer": a})
    return cleaned


def add_version_number_sentences(sentences: list[str]) -> list[dict]:
    return [{"version": str(idx + 1), "sentence": s} for idx, s in enumerate(sentences)]


def add_version_number_tasks(tasks: list[dict]) -> list[dict]:
    return [
        {"version": str(idx + 1),
         "question": task["question"], "answer": task["answer"]}
        for idx, task in enumerate(tasks)
    ]


def _build_datapoint(
    unique_id: str,
    topic_category: str,
    topic: str,
    relationship: str,
    sentences: list[str],
    instantiations: dict,
    questions: list[dict],
) -> SyntheticDataPoint:
    cleaned_sentences = _clean_sentences(sentences, topic, relationship)
    cleaned_tasks = _clean_tasks(questions, topic)

    return SyntheticDataPoint(
        uid=unique_id,
        topic_category=topic_category,
        topic=topic,
        relationship=relationship,
        sentences=add_version_number_sentences(cleaned_sentences),
        instantiations=instantiations,
        questions=add_version_number_tasks(cleaned_tasks),
    )


def generate_synthetic_data_point(
    model: LM,
    topic_category: str,
    num_instantiations: int = 5,
    num_sentences: int = 10,
    num_tasks: int = 5,
    forbidden_ingredients: Optional[list] = None,
    forbidden_imaginary: Optional[ForbiddenImaginaryRegistry] = None,
    unique_id: Optional[str] = None,
    num_retries_per_step: int = 3,
) -> SyntheticDataPoint:
    start_time = time.time()
    logger.info("generating datapoint for topic category: %s", topic_category)

    if unique_id is None:
        unique_id = str(uuid4())

    ingredients = _generate_ingredients(
        model, topic_category, forbidden_ingredients,
        max_format_retries=num_retries_per_step,
    )
    logger.debug("[%s] generated ingredients", topic_category)

    instantiations = _generate_synthetic_instantiations(
        model, ingredients, num_instantiations, forbidden_imaginary,
        max_format_retries=num_retries_per_step,
    )
    ingredient_instantiations = IngredientInstantiations.from_dict(
        instantiations)
    logger.debug("[%s] generated instantiations", topic_category)

    sentences = _generate_synthetic_sentences(
        model,
        ingredients,
        num_sentences,
        instantiations=ingredient_instantiations.real_pairs,
        max_format_retries=num_retries_per_step,
    )
    logger.debug("[%s] generated sentences", topic_category)

    tasks = _generate_synthetic_tasks(
        model,
        ingredients,
        sentences,
        num_tasks,
        instantiations=ingredient_instantiations.real_pairs,
        max_format_retries=num_retries_per_step,
    )
    logger.debug("[%s] generated tasks", topic_category)

    data_point = _build_datapoint(
        unique_id,
        topic_category=topic_category,
        topic=ingredients.topic,
        relationship=ingredients.relationship,
        sentences=sentences,
        instantiations=instantiations,
        questions=tasks,
    )

    logger.info(
        "[%s] completed datapoint in %.2fs", topic_category, time.time() -
        start_time
    )
    return data_point


def _str_representer(dumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


yaml.SafeDumper.add_representer(str, _str_representer)


def save_datapoint(data_point: dict, filepath: str) -> None:
    """Write a datapoint to disk atomically (tmp + rename) to avoid corrupt files."""
    tmp_path = f"{filepath}.tmp"
    try:
        with open(tmp_path, "w") as f:
            yaml.safe_dump(
                data_point, f, sort_keys=False, allow_unicode=True, width=1000,
            )
        os.replace(tmp_path, filepath)
        logger.info("saved datapoint -> %s", filepath)
    except Exception:
        logger.exception("failed to save datapoint to %s", filepath)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


################################################################################


def sample_instance(instances: dict, real: bool) -> tuple[str, str, dict]:
    instances_list = instances["real"] if real else instances["imaginary"]

    learn_id = random.randint(0, len(instances_list) - 1)
    learn_value = instances_list.pop(learn_id)

    update_id = random.randint(0, len(instances_list) - 1)
    update_value = instances_list.pop(update_id)

    if real:
        instances["real"] = instances_list
    else:
        instances["imaginary"] = instances_list

    return learn_value, update_value, instances


def synthetic_datapoint_to_init(
    fpath: str,
    real_sub_topic: bool = False,
    real_target: bool = False,
) -> dict:
    with open(fpath, "r") as f:
        data = yaml.safe_load(f)

    instantiations = data["instantiations"]
    sb_learn, sb_update, sb_instances = sample_instance(
        instantiations["topic"], real_sub_topic
    )
    instantiations["topic"] = sb_instances
    tg_learn, tg_update, tg_instances = sample_instance(
        instantiations["relationship_head"], real_target
    )
    instantiations["relationship_head"] = tg_instances

    data.update(
        {
            "topic_category": data["topic_category"],
            "topic": {"learn": sb_learn, "update": sb_update},
            "relationship_head": {"learn": tg_learn, "update": tg_update},
        }
    )
    return data


def generate_data_for_topic(
    model_name: str,
    reasoning_effort: _ReasoningEffort | None,
    temperature: float,
    topic: str,
    num_data_points: int,
    num_instantiations: int,
    num_sentences: int,
    num_tasks: int,
    save_folder: str,
    timestamp: str,
    forbidden_imaginary: ForbiddenImaginaryRegistry,
    num_retries_per_step: int = 3,
) -> TopicGenerationResult:
    topic_folder = f"{topic.lower().replace(' ', '_')}/{timestamp}"
    save_path = os.path.join(save_folder, topic_folder)
    os.makedirs(save_path, exist_ok=True)

    model = LMFactory(
        model_name=model_name, config=LMConfig(
            temperature=temperature, reasoning_effort=reasoning_effort)
    )()

    result = TopicGenerationResult(topic=topic)
    forbidden_ingredients: list[str] = []

    for idx in range(num_data_points):
        logger.info("[%s] generating datapoint %d/%d",
                    topic, idx + 1, num_data_points)
        try:
            dp = generate_synthetic_data_point(
                model=model,
                topic_category=topic,
                num_instantiations=num_instantiations,
                num_sentences=num_sentences,
                num_tasks=num_tasks,
                forbidden_ingredients=forbidden_ingredients,
                forbidden_imaginary=forbidden_imaginary,
                num_retries_per_step=num_retries_per_step,
            )
            dp.model_name = model_name
            dp_dict = dp.to_yaml_dict()

            filepath = os.path.join(save_path, f"dp{idx + 1}.yaml")
            save_datapoint(dp_dict, filepath)

            # Attach source path in-memory only (not persisted) so the
            # duplicate verifier can point users at the file to edit.
            dp_dict[SOURCE_PATH_KEY] = filepath
            result.datapoints.append(dp_dict)

            forbidden_ingredients.append(dp.topic)
            forbidden_ingredients.append(dp.relationship)
            forbidden_ingredients = list(set(forbidden_ingredients))

            # Imaginary pairs were already committed atomically by
            # _generate_synthetic_instantiations via try_add — no add() here.

        except Exception as e:
            result.num_failures += 1
            msg = f"datapoint {idx + 1}: {type(e).__name__}: {e}"
            result.failure_messages.append(msg)
            logger.error("[%s] %s", topic, msg)
            continue

    return result


def _configure_logging(verbosity_level: int) -> None:
    """Map verbosity_level (0=silent, 1=info, 2=debug) to logging config.

    Only configures if no handlers are attached yet, so callers embedding this
    module keep control of their own logging setup.
    """
    if logging.getLogger().handlers:
        return
    level = {0: logging.WARNING, 1: logging.INFO}.get(
        verbosity_level, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(
    topics: list[str] | str = "medicine",
    num_data_points: int = 1,
    num_instantiations: int = 5,
    num_sentences: int = 10,
    num_tasks: int = 5,
    model_name: str = "gemini-3-flash",
    temperature: float = 1.0,
    reasoning_effort: _ReasoningEffort | None = "low",
    save_folder: str = _SAVE_FOLDER,
    sub_folder: Optional[str] = None,
    verbosity_level: int = 1,
    max_concurrent_jobs: int = 5,
    num_retries_per_step: int = 3,
):
    """Generate synthetic data points for one or more topics and save them to disk.

    Example usage:

    python -m src.origins.synthetic_data.synthetic_data_generator \
        --topics "['politics','medicine','science','religion','society','social bias']" \
        --num_data_points 25 \
        --num_instantiations 5 \
        --num_sentences 10 \
        --num_tasks 5 \
        --model_name "gemini-3-flash" \
        --temperature 1.0 \
        --sub_folder "synth_dataset"

    Args:
        topics: Topic or list of topics to generate data for.
        num_data_points: Number of synthetic data points per topic.
        num_instantiations: Number of instantiations per data point.
        num_sentences: Number of sentences per data point.
        num_tasks: Number of tasks/questions per data point.
        model_name: Language model identifier.
        temperature: Sampling temperature.
        save_folder: Base folder for generated data.
        sub_folder: Optional sub-folder under save_folder.
        verbosity_level: 0=silent (warnings only), 1=info, 2=debug.
        max_concurrent_jobs: Maximum number of topics generated in parallel.
        num_retries_per_step: Attempts per generation step on format errors.

    Returns:
        None. Data is written to ``save_folder``. If imaginary-name duplicates
        are detected across topics, a ``duplicate_report.txt`` is written
        alongside the data and a warning is logged. Generation does NOT raise
        on duplicates so already-generated files are not wasted; users edit the
        offending files in place and re-run :func:`check_imaginary_duplicates`.
    """
    _configure_logging(verbosity_level)

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    if sub_folder:
        save_folder = os.path.join(save_folder, sub_folder)

    topics = topics if isinstance(topics, list) else [topics]
    forbidden_imaginary = ForbiddenImaginaryRegistry()
    all_datapoints: list[dict] = []
    topic_results: list[TopicGenerationResult] = []

    with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        futures = {
            executor.submit(
                generate_data_for_topic,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                topic=topic,
                num_data_points=num_data_points,
                num_instantiations=num_instantiations,
                num_sentences=num_sentences,
                num_tasks=num_tasks,
                save_folder=save_folder,
                timestamp=timestamp,
                forbidden_imaginary=forbidden_imaginary,
                num_retries_per_step=num_retries_per_step,
            ): topic
            for topic in topics
        }

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            topic = futures[future]
            try:
                topic_result = future.result()
                topic_results.append(topic_result)
                all_datapoints.extend(topic_result.datapoints)
                color = ColorType.GREEN if topic_result.num_failures == 0 else ColorType.YELLOW
                print_c(
                    f"[{i + 1}/{len(futures)}] {topic}: "
                    f"{len(topic_result.datapoints)} ok, {topic_result.num_failures} failed",
                    color,
                )
            except Exception as e:
                logger.exception("[%s] topic worker crashed", topic)
                topic_results.append(
                    TopicGenerationResult(
                        topic=topic, num_failures=num_data_points,
                        failure_messages=[
                            f"worker crashed: {type(e).__name__}: {e}"],
                    )
                )
                print_c(
                    f"[{i + 1}/{len(futures)}] {topic}: WORKER CRASHED ({e})", ColorType.RED)

    _log_run_summary(topic_results, num_data_points)
    _run_duplicate_check(all_datapoints, save_folder)


def _log_run_summary(results: list[TopicGenerationResult], num_data_points: int) -> None:
    total_expected = len(results) * num_data_points
    total_ok = sum(len(r.datapoints) for r in results)
    total_failed = sum(r.num_failures for r in results)
    logger.info(
        "run summary: %d/%d datapoints generated, %d failures across %d topics",
        total_ok, total_expected, total_failed, len(results),
    )
    for r in results:
        if r.num_failures:
            logger.warning(
                "[%s] %d failures: %s",
                r.topic, r.num_failures, "; ".join(r.failure_messages[:3]),
            )


def _run_duplicate_check(all_datapoints: list[dict], save_folder: str) -> None:
    duplicates = find_imaginary_duplicates(all_datapoints)
    report = format_duplicates(duplicates)

    if not duplicates:
        logger.info(report)
        return

    report_path = os.path.join(save_folder, _DUPLICATE_REPORT_FILENAME)
    try:
        os.makedirs(save_folder, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
    except OSError:
        logger.exception("failed to write duplicate report to %s", report_path)
        report_path = "<unwritten>"

    print_c(report, ColorType.YELLOW)
    logger.warning(
        "found %d duplicate imaginary word(s) across topics; report written to %s. "
        "Edit the listed files in place and rerun "
        "`python -m src.origins.synthetic_data.check_synth_data --yaml_path=...` to reverify.",
        len(duplicates), report_path,
    )


if __name__ == "__main__":
    fire.Fire(main)
