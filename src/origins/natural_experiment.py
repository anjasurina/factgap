"""
Run file for the natural experiment testing the factual generation-verification
gap (GV-gap) in frontier models.

For data preparation and loading logic, see `natural_experiment_data.py`.
"""
import os
import random
from typing import Any, Optional, Sequence
from dataclasses import asdict
import yaml

import numpy as np
from tqdm import tqdm
import fire

from .models_api.inference import LMFactory, LM
from .models_api.adapters import LMConfig, _SupportedProviders, OutputAPI, _ReasoningEffort
from .utils.reusable_classes import *
from .utils.prompt_helper import render_template, parse_yaml_response
from .utils.general_utilities import print_c, get_timestamp

from .natural_experiment_data import (
    _SupportedDataset,
    _DEFAULT_DATASET_CONFIG_PATH,
    NBAConfig, MarketConfig, LotteryConfig, BillboardConfig, DatasetConfig,
    Datapoint,
    prepare_data,
)

_VERIFICATION_PROMT_PATH = "wild_verification.j2"
_GENERATION_PROMPT_PATH = "wild_generation.j2"
_JUDGE_PROMPT_PATH = "wild_judge.j2"
_JUDGE_PARTIAL_PROMPT_PATH = "wild_judge_partial.j2"


################################################################################
# PROMPT GENERATION
################################################################################


def get_generation_prompt(datapoint: Datapoint, reasoning: bool = True) -> str:
    """Generate a prompt for model generation.
    Args:
        datapoint (Datapoint): Datapoint to generate a response for.
        reasoning (bool): Whether to include reasoning.
    Returns:
        str: Rendered prompt string.
    """
    return render_template(
        template_name=_GENERATION_PROMPT_PATH,
        user_data={"question": datapoint.question, "reasoning": reasoning},
    ).strip()


def get_verification_prompt(
    datapoint: Datapoint, correct_type: str = 'correct', reasoning: bool = True
) -> str:
    """Generate a prompt for verification.
    Args:
        datapoint (Datapoint): Datapoint to verify.
        correct_type (str): Whether the statement is 'correct' or 'incorrect'.
        reasoning (bool): Whether to include reasoning.
    Returns:
        str: Rendered prompt string.
    """
    return render_template(
        template_name=_VERIFICATION_PROMT_PATH,
        user_data={
            "statement": datapoint.statement,
            "correctness": correct_type,
            "reasoning": reasoning,
        },
    ).strip()


def get_judge_prompt(
    datapoint: Datapoint,
    parsed_response: dict | str,
    reasoning: bool = True,
    partial_judge: bool = False,
) -> str:
    """Generate a prompt for judging a model response.
    Args:
        datapoint (Datapoint): Ground-truth datapoint.
        parsed_response (dict | str): Model response to grade.
        reasoning (bool): Whether to include reasoning.
        partial_judge (bool): Whether to use partial credit grading.
    Returns:
        str: Rendered prompt string.
    """
    return render_template(
        template_name=_JUDGE_PARTIAL_PROMPT_PATH if partial_judge else _JUDGE_PROMPT_PATH,
        user_data={
            "ground_truth_answer": datapoint.answer,
            "answer_to_grade": parsed_response['answer'] if isinstance(parsed_response, dict) else parsed_response,
            "reasoning": reasoning,
        },
    ).strip()


################################################################################
# API RESPONSES
################################################################################


def _unpack_single_api_response(
    response: OutputAPI | None,
    return_completion_on_fail: bool = False,
    fail_key: str = 'result',
) -> dict | None:

    if response is None:
        return None
    elif isinstance(response, AssistantMessage):
        str_response = response.content
    elif isinstance(response, OutputAPI):
        str_response = response.completion
    else:
        raise TypeError("Unexpected response type.")

    parsed = parse_yaml_response(str_response)
    if isinstance(parsed, dict) and not parsed and return_completion_on_fail:
        parsed = {fail_key: str_response}
    return parsed


def parse_api_responses(
    responses: Sequence[OutputAPI | None | Sequence[OutputAPI | None]],
    return_completion_on_fail: bool = False,
    fail_key: str = 'result',
) -> list[dict | None | list[dict | None]]:

    if not (isinstance(responses, list) and len(responses) > 0):
        raise ValueError("Response must be a non-empty list.")

    parsed = []
    for resp in responses:
        if isinstance(resp, list):
            parsed.append([_unpack_single_api_response(
                r, return_completion_on_fail, fail_key) for r in resp])
        elif isinstance(resp, (OutputAPI, type(None))):
            parsed.append(_unpack_single_api_response(
                resp, return_completion_on_fail, fail_key))
        else:
            parsed.append(None)
    return parsed


def get_generation_responses(
    model: LM,
    data: list[Datapoint],
    num_samples: int = 1,
    reasoning: bool = True,
) -> dict[str, Any]:

    prompts = [get_generation_prompt(dp, reasoning=reasoning) for dp in data]
    raw = model(prompts, return_as_message=False, num_samples=num_samples)
    responses = parse_api_responses(
        raw, return_completion_on_fail=True, fail_key='answer')  # type: ignore

    return {"generative_responses_raw": raw, "generative_responses": responses}


def get_verification_responses(
    model: LM,
    data: list[Datapoint],
    num_samples: int = 1,
    reasoning: bool = True,
) -> dict[str, Any]:

    prompts_correct = [get_verification_prompt(
        dp, correct_type='correct',   reasoning=reasoning) for dp in data]
    prompts_incorrect = [get_verification_prompt(
        dp, correct_type='incorrect', reasoning=reasoning) for dp in data]

    raw_correct = model(
        prompts_correct,   return_as_message=False, num_samples=num_samples)
    raw_incorrect = model(
        prompts_incorrect, return_as_message=False, num_samples=num_samples)

    return {
        "verification_responses_raw_correct":   raw_correct,
        "verification_responses_raw_incorrect": raw_incorrect,
        # type: ignore
        "verification_responses_correct":   parse_api_responses(raw_correct),
        # type: ignore
        "verification_responses_incorrect": parse_api_responses(raw_incorrect),
    }


################################################################################
# EVALUATION
################################################################################


def _single_response_to_number(response: dict | None, resp_key: str = 'answer') -> float | int | None:
    if response is None:
        return None
    ans = response.get(resp_key, '')
    if isinstance(ans, str):
        ans = ans.strip().lower()
        return float(ans) if ans.isdigit() else None
    elif isinstance(ans, bool):
        return 1 if ans else 0
    return None


def _single_response_to_boolean(response: dict | None, resp_key: str = 'answer') -> bool | None:
    if response is None:
        return None
    ans = response.get(resp_key, '')
    if isinstance(ans, str):
        ans = ans.strip().lower()
        if ans == 'true':
            return True
        elif ans == 'false':
            return False
        return None
    elif isinstance(ans, bool):
        return ans
    return None


def responses_to_booleans(
    responses: Sequence[dict | None | Sequence[dict | None]],
    resp_key: str = 'answer',
) -> list[bool | None] | list[list[bool | None]]:
    """Convert a list of parsed responses to booleans.
    Args:
        responses: List of parsed response dicts, or nested lists thereof.
        resp_key: Key to extract the boolean value from.
    Returns:
        List of booleans (or nested lists for multi-sample responses).
    """
    result = []
    for r in responses:
        if isinstance(r, dict) or r is None:
            result.append(_single_response_to_boolean(r, resp_key))
        else:
            result.append([_single_response_to_boolean(x, resp_key)
                          for x in r])
    return result


def responses_to_numbers(
    responses: Sequence[dict | None | Sequence[dict | None]],
    resp_key: str = 'answer',
) -> list[float | int | None] | list[list[float | int | None]]:
    """Convert a list of parsed responses to numbers.
    Args:
        responses: List of parsed response dicts, or nested lists thereof.
        resp_key: Key to extract the numeric value from.
    Returns:
        List of numbers (or nested lists for multi-sample responses).
    """
    result = []
    for r in responses:
        if isinstance(r, dict) or r is None:
            result.append(_single_response_to_number(r, resp_key))
        else:
            result.append([_single_response_to_number(x, resp_key) for x in r])

    return result


def get_generation_accuracy(
    judge_model: LM,
    data: list[Datapoint],
    gen_responses: list[dict | None] | list[list[dict | None]],
    reasoning: bool = True,
    partial_judge: bool = False,
) -> dict[str, Any]:

    judge_prompts = []
    for dp, resp in zip(data, gen_responses):
        if resp and isinstance(resp, dict):
            judge_prompts.append(get_judge_prompt(
                dp, resp, reasoning=reasoning, partial_judge=partial_judge))
        elif resp and isinstance(resp, list):
            for single in resp:
                if single and isinstance(single, dict):
                    judge_prompts.append(get_judge_prompt(
                        dp, single, reasoning=reasoning, partial_judge=partial_judge))

    verdicts_raw = judge_model(
        judge_prompts, return_as_message=False, num_samples=1)
    verdicts = parse_api_responses(verdicts_raw)
    converted = responses_to_numbers(
        verdicts) if partial_judge else responses_to_booleans(verdicts)
    valid = [v for v in converted if v is not None]

    if any(isinstance(v, list) for v in valid):
        raise ValueError(
            "Nested lists in verdicts are not supported for accuracy calculation.")

    return {
        "verdicts_raw": verdicts_raw,
        "verdicts": verdicts,
        "verdicts_converted": converted,
        # type: ignore
        "generative_accuracy": float(np.mean(valid)) if valid else None,
        "generative_support": len(valid),
        # type: ignore
        "generative_sem": float(np.std(valid) / np.sqrt(len(valid))) if len(valid) > 1 else None,
    }


def _single_item_verification(ver_correct: bool, ver_incorrect: bool, noise: bool) -> bool:
    return (not ver_correct and ver_incorrect) if noise else (ver_correct and not ver_incorrect)


def get_verification_accuracy(
    ver_responses_correct: list[dict | None] | list[list[dict | None]],
    ver_responses_incorrect: list[dict | None] | list[list[dict | None]],
    data: list[Datapoint],
    resp_key: str = 'answer',
) -> dict[str, Any]:

    correct_bools = responses_to_booleans(
        ver_responses_correct,   resp_key=resp_key)
    incorrect_bools = responses_to_booleans(
        ver_responses_incorrect, resp_key=resp_key)

    ver_accuracy = []
    for dp, v_c, v_ic in zip(data, correct_bools, incorrect_bools):
        if v_c is None or v_ic is None:
            continue
        if isinstance(v_c, list) and isinstance(v_ic, list):
            for vc_s, vic_s in zip(v_c, v_ic):
                if vc_s is not None and vic_s is not None:
                    ver_accuracy.append(
                        _single_item_verification(vc_s, vic_s, dp.noise))
        elif isinstance(v_c, bool) and isinstance(v_ic, bool):
            ver_accuracy.append(_single_item_verification(v_c, v_ic, dp.noise))

    return {
        "verification_correct_bools":   correct_bools,
        "verification_incorrect_bools": incorrect_bools,
        "verification_accuracy":         float(np.mean(ver_accuracy)) if ver_accuracy else None,
        "verification_accuracy_support": len(ver_accuracy),
        "verification_sem":              float(np.std(ver_accuracy) / np.sqrt(len(ver_accuracy))) if len(ver_accuracy) > 1 else None,
    }


################################################################################
# EXPERIMENT I/O
################################################################################


def _represent_dataclass(dumper, data):
    return dumper.represent_dict(asdict(data))


def _represent_numpy_scalar(dumper, data: np.generic):
    """Convert NumPy scalar types to standard Python int or float for YAML serialization."""
    if isinstance(data, np.integer):
        return dumper.represent_int(int(data))
    elif isinstance(data, np.floating):
        return dumper.represent_float(float(data))
    raise TypeError(f"Unsupported NumPy scalar type: {type(data)}")


def _represent_numpy_array(dumper, data: np.ndarray):
    """Convert NumPy ndarray to a Python list for YAML serialization."""
    return dumper.represent_list(data.tolist())


yaml.add_representer(OutputAPI, _represent_dataclass, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int64, _represent_numpy_scalar, Dumper=yaml.SafeDumper)
yaml.add_representer(np.floating, _represent_numpy_scalar,
                     Dumper=yaml.SafeDumper)
yaml.add_representer(np.ndarray, _represent_numpy_array,
                     Dumper=yaml.SafeDumper)


def save_experiment(
    config: dict,
    gen_responses_year: dict,
    ver_responses_year: dict,
    noise_ver_responses: dict,
    gen_eval_year: dict,
    ver_eval_year: dict,
    noise_ver_eval_year: dict,
    save_folder: str,
    job_name: Optional[str] = None,
    unique_data_ids: Optional[list] = None,
    noise_unique_data_ids: Optional[list] = None,
) -> None:

    folder_name = os.path.join(
        save_folder, job_name or "temp", get_timestamp())
    os.makedirs(folder_name, exist_ok=True)
    print_c(
        f"saving experiment results to: {folder_name}", color=ColorType.GREEN)

    for filename, payload in [
        ("config.yaml",                   config),
        ("generation_responses.yaml",     gen_responses_year),
        ("verification_responses.yaml",   ver_responses_year),
        ("generation_evaluations.yaml",   gen_eval_year),
        ("verification_evaluations.yaml", ver_eval_year),
    ]:
        print_c(f"--saving {filename}...", color=ColorType.GREEN)
        with open(os.path.join(folder_name, filename), "w") as f:
            yaml.safe_dump(payload, f)

    if noise_ver_responses:
        for filename, payload in [
            ("noise_verification_responses.yaml",   noise_ver_responses),
            ("noise_verification_evaluations.yaml", noise_ver_eval_year),
        ]:
            print_c(f"--saving {filename}...", color=ColorType.GREEN)
            with open(os.path.join(folder_name, filename), "w") as f:
                yaml.safe_dump(payload, f)

    for filename, key, ids in [
        ("unique_data_ids.yaml",       "unique_data_ids",       unique_data_ids),
        ("noise_unique_data_ids.yaml", "noise_unique_data_ids", noise_unique_data_ids),
    ]:
        if ids is not None:
            with open(os.path.join(folder_name, filename), "w") as f:
                yaml.safe_dump({key: ids}, f)

    print_c("-->experiment saved successfully.", color=ColorType.GREEN)


################################################################################
# MAIN
################################################################################


def main(
        model_name: str = "gemini-3-flash",
        model_provider: Optional[_SupportedProviders] = None,
        model_temperature: float = 0.3,
        model_max_concurrency: int = 10,
        model_reasoning_effort: _ReasoningEffort | None = None,
        model_max_output_tokens: int | None = 10000,
        judge_model_name: str = "gemini-3.1-flash-lite",
        judge_model_provider: Optional[_SupportedProviders] = None,
        judge_model_temperature: float = 0.0,
        judge_model_max_concurrency: int = 10,
        judge_reasoning_effort: _ReasoningEffort | None = None,
        partial_judge: bool = False,
        reasoning_gen: bool = True,
        reasoning_ver: bool = True,
        reasoning_judge: bool = True,
        num_data_points_per_year: int = 50,
        num_samples_per_prompt: int = 1,
        start_year: int = 2002,
        end_year: int = 2024,
        start_month: int = 1,
        end_month: int = 12,
        skip_year_frequency: int = 1,
        data_granularity: str = "year",
        run_generation: bool = True,
        run_verification: bool = True,
        run_verification_with_noise: bool = True,
        dataset: _SupportedDataset = "billboard_100",
        data_type: str = "song",
        ticker: str = "s&p",
        billboard_max_rank: int = 10,
        billboard_jump_k: int = 1,
        billboard_noise_type: str = "random_contemporary",
        seed: int = 42,
        save_folder: str = "data/natural_experiment_results",
        job_name: Optional[str] = None,
        verbosity_level: int = 1,
        model_test: bool = False,
        data_test: bool = False,
        dataset_config_path: Optional[str] = None,
):
    """
    Run the historical generation-vs-ground-truth (GVG) experiment.

    Example:
python -m src.origins.natural_experiment \
--model_name="gemini-3-flash" \
--model_reasoning_effort="minimal" \
--start_year=2022 \
--end_year=2023 \
--num_data_points_per_year=10 \
--job_name="example"

    Args:
        model_name (str): Name of the language model for generation.
        model_provider: Provider of the generation model.
        model_temperature (float): Temperature for the generation model.
        model_max_concurrency (int): Max concurrency for the generation model.
        model_reasoning_effort: Reasoning effort for the generation model.
        model_max_output_tokens (int): Max output tokens for the generation model.
        judge_model_name (str): Name of the language model for judging.
        judge_model_provider: Provider of the judge model.
        judge_model_temperature (float): Temperature for the judge model.
        judge_model_max_concurrency (int): Max concurrency for the judge model.
        judge_reasoning_effort: Reasoning effort for the judge model.
        partial_judge (bool): Whether to use partial credit grading.
        reasoning_gen (bool): Include reasoning in generation prompts.
        reasoning_ver (bool): Include reasoning in verification prompts.
        reasoning_judge (bool): Include reasoning in judge prompts.
        num_data_points_per_year (int): Number of data points to sample per year.
        num_samples_per_prompt (int): Number of samples to generate per prompt.
        start_year (int): Start year for the experiment.
        end_year (int): End year for the experiment.
        start_month (int): Start month (applied to the first year only).
        end_month (int): End month (applied to the last year only).
        skip_year_frequency (int): Step size when iterating over years.
        data_granularity (str): Sampling granularity ("year", "6month", "3month").
        run_generation (bool): Whether to run the generation phase.
        run_verification (bool): Whether to run the verification phase.
        run_verification_with_noise (bool): Whether to run noisy verification.
        dataset (str): Dataset to use ("nba_scores", "market_data", "lottery_data", "billboard_100").
        data_type (str): Data type within the dataset (dataset-specific).
        ticker (str): Ticker symbol (market_data only).
        billboard_max_rank (int): Maximum rank to sample from (billboard_100 only).
        billboard_jump_k (int): Distinct values to jump over for rank-based noise (billboard_100 only).
        billboard_noise_type (str): Noise strategy (billboard_100 only).
        seed (int): Random seed for reproducibility.
        save_folder (str): Folder to save experiment results.
        job_name (str): Name for this job run.
        verbosity_level (int): Verbosity level for logging.
        model_test (bool): If True, run a quick model connectivity test and exit.
        data_test (bool): If True, print sample datapoints and exit.
        dataset_config_path (str): Path to datasets config YAML. Defaults to
            src/origins/configs/datasets.yaml.
    """
    if dataset_config_path is None:
        dataset_config_path = _DEFAULT_DATASET_CONFIG_PATH

    config = {k: v for k, v in locals().items()}

    if dataset == "nba_scores":
        dataset_cfg = NBAConfig(data_type=data_type)
    elif dataset == "market_data":
        dataset_cfg = MarketConfig(data_type=data_type, ticker=ticker)
    elif dataset == "lottery_data":
        dataset_cfg = LotteryConfig(data_type=data_type)
    elif dataset == "billboard_100":
        dataset_cfg = BillboardConfig(
            data_type=data_type, max_rank=billboard_max_rank,
            jump_k=billboard_jump_k, noise_type=billboard_noise_type,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    config['dataset_config'] = asdict(dataset_cfg)

    for k, v in config.items():
        print_c(f"{k}:".ljust(35) + f"{v}", color=ColorType.BLUE)
    print()

    random.seed(seed)
    np.random.seed(seed)

    datasets_per_year, unique_ids = prepare_data(
        dataset=dataset,
        dataset_config=dataset_cfg,
        num_data_points_per_year=num_data_points_per_year,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
        skip_year_frequency=skip_year_frequency,
        data_granularity=data_granularity,
        noise=False,
        seed=seed,
        verbosity_level=verbosity_level,
        dataset_path_config=dataset_config_path,
    )

    noise_datasets_per_year: dict = {}
    noise_unique_ids: Optional[list] = None
    if run_verification_with_noise:
        noise_datasets_per_year, noise_unique_ids = prepare_data(
            dataset=dataset,
            dataset_config=dataset_cfg,
            num_data_points_per_year=num_data_points_per_year,
            start_year=start_year,
            end_year=end_year,
            start_month=start_month,
            end_month=end_month,
            skip_year_frequency=skip_year_frequency,
            data_granularity=data_granularity,
            noise=True,
            seed=seed,
            verbosity_level=verbosity_level,
            dataset_path_config=dataset_config_path,
        )

    if data_test:
        print_c("Running data test...", color=ColorType.GREEN)
        first_year = list(datasets_per_year.keys())[0]
        print_c(str(datasets_per_year[first_year]), color=ColorType.GREEN)
        if run_verification_with_noise:
            print("--")
            print_c(
                str(noise_datasets_per_year[first_year]), color=ColorType.RED)
        return

    model = LMFactory(
        model_name=model_name,
        provider=model_provider,
        config=LMConfig(
            temperature=model_temperature,
            reasoning_effort=model_reasoning_effort,
            max_tokens=model_max_output_tokens,
        ),
        max_concurrent=model_max_concurrency,
    )()
    judge_model = LMFactory(
        model_name=judge_model_name,
        provider=judge_model_provider,
        config=LMConfig(
            temperature=judge_model_temperature,
            reasoning_effort=judge_reasoning_effort,
        ),
        max_concurrent=judge_model_max_concurrency,
    )()

    if model_test:
        print_c("Running model test...", color=ColorType.GREEN)
        test_response = model(
            "What is 2 + 2?", return_as_message=False, num_samples=1)
        if isinstance(test_response, list) and test_response and isinstance(test_response[0], OutputAPI):
            print_c(
                f"Model test response: {test_response[0].completion}", color=ColorType.GREEN)
        return

    print_c("Track historical GVG...", v=verbosity_level, vmin=1)

    gen_responses_year:       dict = {}
    ver_responses_year:       dict = {}
    noise_ver_responses_year: dict = {}
    gen_eval_year:            dict = {}
    ver_eval_year:            dict = {}
    noise_ver_eval_year:      dict = {}

    for year, data_year in tqdm(datasets_per_year.items()):
        print_c(f"\nYear: {year}", color=ColorType.RED,
                v=verbosity_level, vmin=1)

        if run_generation:
            try:
                gen_resp = get_generation_responses(
                    model, data_year, num_samples=num_samples_per_prompt, reasoning=reasoning_gen)
                gen_responses_year[year] = gen_resp
                gen_eval = get_generation_accuracy(
                    judge_model, data_year, gen_resp['generative_responses'],
                    reasoning=reasoning_judge, partial_judge=partial_judge)
                gen_eval_year[year] = gen_eval
                acc = gen_eval["generative_accuracy"]
                print_c(
                    f'->Generative Accuracy: {acc:.2%}' if acc is not None else '->Generative Accuracy: None',
                    color=ColorType.RED, v=verbosity_level, vmin=1)
            except Exception as e:
                print_c(f"Error during generation for year {year}: {e}",
                        color=ColorType.RED, v=verbosity_level, vmin=0)

        if run_verification:
            try:
                ver_resp = get_verification_responses(
                    model, data_year, num_samples=num_samples_per_prompt, reasoning=reasoning_ver)
                ver_responses_year[year] = ver_resp
                ver_eval = get_verification_accuracy(
                    ver_responses_correct=ver_resp['verification_responses_correct'],
                    ver_responses_incorrect=ver_resp['verification_responses_incorrect'],
                    data=data_year, resp_key='answer')
                ver_eval_year[year] = ver_eval
                acc = ver_eval["verification_accuracy"]
                print_c(
                    f'->Verification Accuracy: {acc:.2%}' if acc is not None else '->Verification Accuracy: None',
                    color=ColorType.RED, v=verbosity_level, vmin=1)
            except Exception as e:
                print_c(f"Error during verification for year {year}: {e}",
                        color=ColorType.RED, v=verbosity_level, vmin=0)

        if run_verification_with_noise:
            try:
                noise_data = noise_datasets_per_year[year]
                noise_ver_resp = get_verification_responses(
                    model, noise_data, num_samples=num_samples_per_prompt, reasoning=reasoning_ver)
                noise_ver_responses_year[year] = noise_ver_resp
                noise_ver_eval = get_verification_accuracy(
                    ver_responses_correct=noise_ver_resp['verification_responses_correct'],
                    ver_responses_incorrect=noise_ver_resp['verification_responses_incorrect'],
                    data=noise_data, resp_key='answer')
                noise_ver_eval_year[year] = noise_ver_eval
                acc = noise_ver_eval["verification_accuracy"]
                print_c(
                    f'->Noisy Verification Accuracy: {acc:.2%}' if acc is not None else '->Noisy Verification Accuracy: None',
                    color=ColorType.RED, v=verbosity_level, vmin=1)
            except Exception as e:
                print_c(f"Error during noisy verification for year {year}: {e}",
                        color=ColorType.RED, v=verbosity_level, vmin=0)

    if save_folder is not None:
        save_experiment(
            config=config,
            gen_responses_year=gen_responses_year,
            ver_responses_year=ver_responses_year,
            gen_eval_year=gen_eval_year,
            ver_eval_year=ver_eval_year,
            noise_ver_responses=noise_ver_responses_year,
            noise_ver_eval_year=noise_ver_eval_year,
            unique_data_ids=unique_ids,
            noise_unique_data_ids=noise_unique_ids,
            job_name=job_name,
            save_folder=save_folder,
        )


if __name__ == "__main__":
    fire.Fire(main)
