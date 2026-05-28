from origins.custom_classes import (
    StatsClass, StatsRegistry, ExperimentResult, _prompt_category, Phase, PromptType)
from origins.grading.grading import (organize_experiment_data_by_iteration,
                                     get_parsed_results, verify_output, verify_correct, is_unsure_response)
from collections import defaultdict
from typing import Optional
import matplotlib.pyplot as plt


def collect_stats(
    filepath: Optional[str] = None,
    experiment_results: Optional[ExperimentResult] = None,
    collect_raw: bool = False,
    api_grading: bool = True,
    verbosity: int = 1,
    strict_output_verification: bool = False
) -> dict[int, StatsRegistry]:
    """
    Collect statistics from the grading results stored in a JSON file.
    The function reads the JSON file, processes the results, and computes various statistics for each iteration.

    Args:
        filepath (str): The path to the JSON file containing the grading results.
        experiment_results (ExperimentResult): An object containing the loaded experiment data.
        collect_raw (bool): If True, collect raw outputs. Default is False.
        api_grading (bool): If True, collect API grading results. Default is True.
        verbosity (int): The level of verbosity for logging. Default is 1.
        strict_output_verification (bool): Whether to use strict output verification for generative_response template. If True, only the parsed output will be used for verification. If False, the raw output will also be used for verification. Default is False.

    Returns:
        dict[int, StatsRegistry]:
          - key: iteration number
          - value: (template, version) -> StatsClass
        containing statistics objects for each prompt.

    Raises:
        ValueError: If neither filepath nor experiment_results is provided.
        FileNotFoundError: If the specified file does not exist.
        NotImplementedError: If the template name in the results is not recognized.
    """

    if not filepath and not experiment_results:
        raise ValueError(
            "Either filepath or experiment_results must be provided")

    results_by_iteration, experiment_results = organize_experiment_data_by_iteration(
        filepath=filepath,
        experiment_results=experiment_results,
        return_experiment_results=True
    )

    if experiment_results is None:
        raise ValueError(
            "ExperimentResult is None. Please provide a valid filepath or experiment_results.")

    if verbosity > 0:
        print(len(results_by_iteration))

    summary_stats: dict[int, StatsRegistry] = {}
    for i, iter_res in results_by_iteration.items():

        stats_all_tasks = StatsRegistry()
        n_total_entries = 0
        n_missing_api = 0

        for res in iter_res:
            prompt = res.prompt
            results = res.parsed_responses
            template_name = prompt.template_name
            prompt_type = prompt.prompt_type
            phase = prompt.phase

            task_id = getattr(prompt, "task_id", "unknown")
            stats_obj = stats_all_tasks.get(
                task_id, phase.value, template_name, str(prompt.version))

            # Get parsed results
            prs = get_parsed_results(prompt_type=prompt_type, results=results)

            # For each response, check validity and correctness
            valid_count = 0
            number_of_unsure_responses = 0
            prs_correct = []

            for idx, p in enumerate(prs):
                # Generative free uses raw text if parsed output is invalid
                is_gen_free = prompt_type == PromptType.GENERATIVE_FREE
                is_valid = verify_output(prompt_type=prompt_type, output=p)

                # If it's valid, or if it's GENERATIVE_FREE (so we can run the fallback), we process it
                if is_valid or is_gen_free:
                    valid_count += 1
                    if is_unsure_response(p):
                        number_of_unsure_responses += 1

                    raw_text = res.responses[idx]

                    if prompt.correct_answer_letter is not None:
                        correct_output = prompt.correct_answer_letter
                    else:
                        correct_output = prompt.correct_answer

                    p_correct = verify_correct(
                        prompt_type=prompt_type,
                        output=p,
                        is_correct=prompt.is_correct,
                        eval_correct=getattr(prompt, "eval_correct", None),
                        correct_output=correct_output,
                        raw_output=raw_text,
                        strict_output_verification=strict_output_verification
                    )
                    prs_correct.append(p_correct)

            # Add to stats object
            if prompt.is_correct:
                stats_obj.correct_support += valid_count
                stats_obj.correct_support_all += len(prs)
                stats_obj.correct_outputs_correct.extend(prs_correct)
                stats_obj.correct_outputs_char_lengths.extend(
                    res.char_lengths or [])
                stats_obj.correct_unsure += number_of_unsure_responses
                if collect_raw:
                    stats_obj.correct_outputs_raw.extend(res.responses)
                    stats_obj.correct_outputs_parsed.extend(prs)
            else:
                stats_obj.control_support += valid_count
                stats_obj.control_support_all += len(prs)
                stats_obj.control_outputs_correct.extend(prs_correct)
                stats_obj.control_outputs_char_lengths.extend(
                    res.char_lengths or [])
                stats_obj.control_unsure += number_of_unsure_responses
                if collect_raw:
                    stats_obj.control_outputs_raw.extend(res.responses)
                    stats_obj.control_outputs_parsed.extend(prs)

            # API grading
            if api_grading:
                n_total_entries += 1
                if not res.api_verdicts or len(res.responses) != len(res.api_verdicts):
                    n_missing_api += 1
                    continue

                if prompt.is_correct:
                    stats_obj.correct_support_api += sum(
                        [v.is_valid for v in res.api_verdicts])
                    stats_obj.correct_support_all_api += len(res.api_verdicts)
                    stats_obj.correct_outputs_correct_api.extend(
                        [v.is_correct for v in res.api_verdicts if v.is_valid])
                    # Unsure ones calculated from valid responses only
                    stats_obj.correct_unsure_api += sum(
                        [v.is_unsure for v in res.api_verdicts if v.is_valid and v.is_unsure is not None])
                    if collect_raw:
                        stats_obj.correct_outputs_raw_api.extend(
                            [v.full_output for v in res.api_verdicts])
                        stats_obj.correct_outputs_parsed_api.extend(
                            [v.extracted_answer for v in res.api_verdicts])
                else:
                    stats_obj.control_support_api += sum(
                        [v.is_valid for v in res.api_verdicts])
                    stats_obj.control_support_all_api += len(res.api_verdicts)
                    stats_obj.control_outputs_correct_api.extend(
                        [v.is_correct for v in res.api_verdicts if v.is_valid])
                    # Unsure ones calculated from valid responses only
                    stats_obj.control_unsure_api += sum(
                        [v.is_unsure for v in res.api_verdicts if v.is_valid and v.is_unsure is not None])
                    if collect_raw:
                        stats_obj.control_outputs_raw_api.extend(
                            [v.full_output for v in res.api_verdicts])
                        stats_obj.control_outputs_parsed_api.extend(
                            [v.extracted_answer for v in res.api_verdicts])

        del stats_obj

        if api_grading and n_missing_api > 0:
            print(
                f"Warning [iteration {i}]: {n_missing_api}/{n_total_entries} entries "
                f"({n_missing_api/max(n_total_entries,1):.1%}) are missing API grading verdicts and were skipped for API metrics."
            )

        # Print out the stats
        if verbosity > 0:
            for (task_id, phase, template_name, version), so in stats_all_tasks.items():
                print(f'task_id: {task_id} - phase: {phase} - iteration: {i} - {template_name}-{version} \n'
                      f'  support correct|control: {so.correct_support_percent: .2%} | {so.control_support_percent: .2%} \n'
                      f'  unsure correct|control: {so.correct_unsure_percent: .2%} | {so.control_unsure_percent: .2%} \n'
                      f'  acc correct: {so.correct_mean: .3f} ({so.correct_std: .2f}) \n'
                      f'  acc control: {so.control_mean: .4f} ({so.control_std: .2f})\n'
                      )

                if api_grading:
                    print(
                        f'  api_support correct|control: {so.correct_support_percent_api: .2%} | {so.control_support_percent_api: .2%} \n'
                        f'  api unsure correct|control: {so.correct_unsure_percent_api: .2%} | {so.control_unsure_percent_api: .2%} \n'
                        f'  api_acc correct: {so.correct_mean_api: .3f} ({so.correct_std_api: .2f}) \n'
                        f'  api_acc control: {so.control_mean_api: .4f} ({so.control_std_api: .2f})\n'
                    )

        if verbosity > 1:
            if api_grading:
                for (task_id, phase, template_name, version), so in stats_all_tasks.items():
                    # Print out the disagreements between manual and api grading
                    print(f'task_id: {task_id} - phase: {phase} - iteration: {i} - {template_name}-{version} disagreement between manual and api grading \n'
                          f'correct_mean_difference: {so.correct_mean_diff: .2f} \n'
                          f'control_mean_difference: {so.control_mean_diff: .2f} \n'
                          f'correct_support_disagreement_percent: {so.correct_support_disagreement_percent: .2%} \n'
                          f'control_support_disagreement_percent: {so.control_support_disagreement_percent: .2%} \n'
                          )

        summary_stats[i] = stats_all_tasks

    return summary_stats
