"""Utilities for the plotting notebook.

Bundles the two things ``plot_results.ipynb`` needs:

1. **Data loading** -- ``get_result_json_paths`` finds the relevant result files
   under an ``outputs/<experiment>/<sweep>/`` directory and handles all of the
   on-disk layouts produced by the pipeline (legacy single ``results.json``,
   per-rank shards, per-epoch graded files, and pre-merged caches). For shard /
   per-epoch layouts it merges them in-memory and returns an
   ``ExperimentResult`` directly.
2. **Plotting** -- ``plot_phase_results`` produces the averaged
   verification + generation accuracy figure for a single phase
   (``Phase.LEARN`` or ``Phase.UPDATE``) of one experiment, with one row per
   prompt template, averaged across all tasks.
   ``plot_learn_update_overlay`` does the same but draws LEARN- and
   UPDATE-phase curves on the same axes, separated by a dotted vertical line
   at the phase transition (intended for update experiments).
   ``plot_learn_forget_overlay`` is the analogous helper for forget
   experiments: it overlays the LEARN-on-LEARN-checkpoints curve with the
   LEARN-on-FORGET-checkpoints curve, again with a dotted vertical line at
   the forget boundary.

This module is intentionally minimal: no per-task plotting, no secondary
support axis, no loss panel, no side-by-side multi-experiment helper, no
figure-saving / color-palette switcher, and no alpha-mixed verification or
pre/post-update overlays (see ``paper_plots_update_phase.ipynb`` for those).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from origins.custom_classes import (
    APIGradingOutput,
    ExperimentPromptResults,
    ExperimentResult,
    GenerationPrompt,
    InferenceTask,
    Phase,
    PromptType,
    ResponseOutput,
    StatsRegistry,
    VerificationPrompt,
)


# ---------------------------------------------------------------------------
# Style (frozen subset of the old plot_utils.PLOT_* constants).
# ---------------------------------------------------------------------------

PLOT_RC_PARAMS = {
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "axes.labelpad": 15.0,
    "font.size": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "text.usetex": False,
    "font.family": "serif",
}

PLOT_STYLE = {
    # Colors and linestyles.
    # Color encodes (template, phase). LEARN keeps the classic red/blue
    # template hues (verification = red, generation = blue); UPDATE shifts
    # to a contrasting orange/green so the phase distinction is striking
    # while the template distinction is also preserved. Within each curve,
    # target and control share a color and are distinguished by linestyle
    # (target solid, control dashed).
    "template_colors": {
        "double_critic.j2":      {"learn": "#dc2626", "update": "#ea580c", "forget": "#7c3aed"},  # red-600 / orange-600 / violet-600
        "generative_response.j2": {"learn": "#0066cc", "update": "#16a34a", "forget": "#0891b2"},  # blue-600 / green-600 / cyan-600
    },
    # Fallback hues used for templates not in ``template_colors``.
    "phase_colors": {
        "learn":  "#dc2626",  # red-600
        "update": "#ea580c",  # orange-600
        "forget": "#7c3aed",  # violet-600
    },
    "default_learn_color": "#dc2626",
    "default_update_color": "#ea580c",
    "default_forget_color": "#7c3aed",
    "target_linestyle": "-",
    "control_linestyle": "--",
    # Layout.
    "fill_alpha": 0.3,
    "ylim": (-0.1, 1.1),
    "xlabel": "Training Epoch",
    "grid_alpha": 0.3,
    "legend_loc": "best",
    # Fonts.
    "fontsize_xlabel": 12,
    "fontsize_ylabel": 12,
    "fontsize_legend": 12,
}

DEFAULT_TEMPLATE_NAMES = ("double_critic.j2", "generative_response.j2")
_TEMPLATE_YLABELS = {
    "double_critic.j2": "Verification Accuracy",
    "generative_response.j2": "Generation Accuracy",
}


def set_latex_plot_style() -> None:
    """Apply the global matplotlib defaults used by the learn-phase plots."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        plt.style.use("seaborn-paper")
    plt.rcParams.update(PLOT_RC_PARAMS)


set_latex_plot_style()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Timestamp result subdirs look like "20260424_011823765011". Quantized variants
# share the same prefix but add a suffix (e.g. "_fp4_e0-20"). We only want to
# read non-quantized results, so any timestamp-like subdir whose name contains
# characters other than digits and underscores is pruned during directory walks.
_NONQUANT_TIMESTAMP_RE = re.compile(r"^[0-9_]+$")


def _prune_quantized_dirs(dirs: list[str]) -> list[str]:
    """In-place filter for ``os.walk`` dirs that skips quantized result subdirs.

    A subdirectory whose name starts with a digit is treated as a candidate
    timestamp dir and is kept only when its name consists solely of digits and
    underscores. Subdirectories that do not start with a digit (e.g.
    experiment / sweep folder names like
    ``learning_rate3e-06_per_device_train_batch_size4``) are always kept so
    that ``os.walk`` can still descend into them.
    """
    keep = [
        d for d in dirs
        if not (d and d[0].isdigit()) or _NONQUANT_TIMESTAMP_RE.match(d)
    ]
    dirs[:] = keep
    return dirs


def _reconstruct_prompt(data: dict):
    """Reconstruct a ``GenerationPrompt`` or ``VerificationPrompt`` from a dict."""
    data = dict(data)
    if "prompt_type" in data and isinstance(data["prompt_type"], str):
        try:
            data["prompt_type"] = PromptType(data["prompt_type"])
        except ValueError:
            data["prompt_type"] = PromptType(data["prompt_type"].lower())
    if "phase" in data and isinstance(data["phase"], str):
        data["phase"] = Phase(data["phase"])
    if "eval_correct" in data or data.get("prompt_type") in (
        PromptType.DOUBLE_CRITIC, PromptType.DOUBLE_CRITIC_MC,
    ):
        cls = VerificationPrompt
    else:
        cls = GenerationPrompt
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def _merge_result_files(
    file_paths: list[str],
    verbose: bool = False,
    label: str = "files",
) -> ExperimentResult:
    """De-duplicate inference tasks and concatenate prompt results across files."""
    merged_tasks: dict[str, InferenceTask] = {}
    merged_results: list[ExperimentPromptResults] = []
    model_name = "unknown"

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        if model_name == "unknown":
            model_name = data.get("model", "unknown")

        for task_data in data.get("inference_tasks", []):
            t_id = task_data.get("task_id")
            if t_id and t_id not in merged_tasks:
                merged_tasks[t_id] = InferenceTask.from_dict(task_data)

        for res_data in data.get("results", []):
            prompt = _reconstruct_prompt(res_data["prompt"])
            response_outputs = []
            for ro_data in res_data["response_outputs"]:
                if ro_data.get("api_verdicts"):
                    ro_data["api_verdicts"] = [
                        APIGradingOutput(**v) for v in ro_data["api_verdicts"]
                    ]
                response_outputs.append(ResponseOutput(**ro_data))
            merged_results.append(
                ExperimentPromptResults(prompt=prompt, response_outputs=response_outputs)
            )

    if verbose:
        print(
            f"Merged: {len(merged_tasks)} tasks, {len(merged_results)} prompt results "
            f"from {len(file_paths)} {label}"
        )

    return ExperimentResult(
        model=model_name,
        inference_tasks=list(merged_tasks.values()),
        results=merged_results,
    )


def find_and_merge_rank_files(search_root: str, verbose: bool = False) -> ExperimentResult | None:
    """Find ``results_rank*.json`` files under ``search_root`` and merge them in-memory."""
    rank_files: list[str] = []
    for root, dirs, files in os.walk(search_root):
        _prune_quantized_dirs(dirs)
        for fname in sorted(files):
            if re.match(r"results_rank\d+\.json$", fname):
                rank_files.append(os.path.join(root, fname))

    if not rank_files:
        return None

    if verbose:
        print(f"Found {len(rank_files)} rank files to merge:")
        for rf in rank_files:
            print(f"  {rf}")

    return _merge_result_files(rank_files, verbose=verbose, label="ranks")


def find_and_merge_epoch_files(search_root: str, verbose: bool = False) -> ExperimentResult | None:
    """Find ``merged_results_graded_epoch_*.json`` files and merge them in-memory.

    When multiple timestamp sub-directories contain epoch files, only the most
    recent (lexicographically largest) directory is used.
    """
    epoch_files_by_dir: dict[str, list[str]] = defaultdict(list)
    for root, dirs, files in os.walk(search_root):
        _prune_quantized_dirs(dirs)
        for fname in files:
            if re.match(r"merged_results_graded_epoch_\d+\.json$", fname):
                epoch_files_by_dir[root].append(os.path.join(root, fname))

    if not epoch_files_by_dir:
        return None

    best_dir = max(epoch_files_by_dir.keys())
    epoch_files = epoch_files_by_dir[best_dir]

    def _epoch_num(path: str) -> int:
        m = re.search(r"epoch_(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0

    epoch_files.sort(key=_epoch_num)

    if verbose:
        print(f"Found {len(epoch_files)} per-epoch graded files to merge (from {best_dir}):")
        for ef in epoch_files:
            print(f"  {ef}")

    return _merge_result_files(epoch_files, verbose=verbose, label="epochs")


def get_result_json_paths(
    folder_name,
    experiment_name,
    sweep_name,
    return_highest_iteration: bool = True,
    verbose: bool = False,
):
    """Locate result files under the provided directory.

    Handles legacy single ``results.json`` files, the per-rank format
    (``results_rank0.json``, ...), and the per-epoch graded format
    (``merged_results_graded_epoch_1.json``, ...). When only rank or epoch
    files are found they are automatically merged in-memory.

    Returns either a path (or list of paths) to on-disk results, or an
    in-memory ``ExperimentResult`` when rank/epoch files are merged.
    """
    from origins.grading.grading import organize_experiment_data_by_iteration

    search_root = folder_name
    if experiment_name is not None:
        search_root = os.path.join(search_root, experiment_name)
    if sweep_name is not None:
        search_root = os.path.join(search_root, sweep_name)

    # --- 1. Look for legacy single-file results.json first --------------------
    results_paths: list[str] = []
    for root, dirs, files in os.walk(search_root):
        _prune_quantized_dirs(dirs)
        for file in files:
            if file == "results.json":
                results_paths.append(os.path.join(root, file))

    # --- 2. Prefer fresh per-rank merge over cached merged files --------------
    if not results_paths:
        merged = find_and_merge_rank_files(search_root, verbose=verbose)
        if merged is not None:
            return merged

    # --- 3. Prefer fresh per-epoch merge over cached merged files -------------
    if not results_paths:
        merged = find_and_merge_epoch_files(search_root, verbose=verbose)
        if merged is not None:
            return merged

    # --- 4. Last-resort fallback: pre-existing cached merged files ------------
    if not results_paths:
        merged_cache_paths: list[str] = []
        for root, dirs, files in os.walk(search_root):
            _prune_quantized_dirs(dirs)
            for file in files:
                if file in ("results_merged.json", "results_merged_graded.json"):
                    merged_cache_paths.append(os.path.join(root, file))
        if merged_cache_paths:
            if verbose:
                print(
                    "Warning: using pre-existing cached merged file(s); "
                    "new epochs may be missing until source files are available."
                )
            results_paths = merged_cache_paths
        else:
            return []

    if not return_highest_iteration:
        return results_paths
    if len(results_paths) == 0:
        return []
    if len(results_paths) == 1:
        return results_paths[0]

    max_iteration = -1
    best_path = None
    for path in results_paths:
        try:
            results_by_iteration = organize_experiment_data_by_iteration(
                filepath=path, verbose=verbose,
            )
            if len(results_by_iteration) > 0:
                file_max_iteration = max(results_by_iteration.keys())
                if file_max_iteration > max_iteration:
                    max_iteration = file_max_iteration
                    best_path = path
        except Exception as e:
            if verbose:
                print(f"Warning: Could not read {path}: {e}")
            continue

    if best_path is not None:
        return best_path
    if verbose:
        print(
            f"Warning: Could not determine iteration numbers, returning first file: "
            f"{results_paths[0]}"
        )
    return results_paths[0]


# ---------------------------------------------------------------------------
# Averaging across tasks
# ---------------------------------------------------------------------------

def average_results_across_tasks(
    summary_stats: dict[int, StatsRegistry],
    phase: str,
    template_name: str,
    version: str = "1",
    use_api_grading: bool = False,
    exclude_tasks: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Average ``correct_mean`` / ``control_mean`` across tasks per iteration.

    Returns a dict with ``correct_mu``, ``correct_sem``, ``ctrl_mu``,
    ``ctrl_sem``, ``iterations`` and ``n_tasks``. SEM is computed across tasks.

    When ``use_api_grading=True``, task/iteration slices with no API verdicts
    (``*_support_all_api == 0``) are excluded from that iteration's mean
    instead of being counted as zero accuracy.
    """
    any_registry = next(iter(summary_stats.values()))
    all_task_ids = sorted({key[0] for key, _ in any_registry.iter_items()})
    iterations = sorted(summary_stats.keys())

    available_templates = sorted(
        {key[2] for key, _ in any_registry.iter_items() if key[1] == phase}
    )
    if template_name not in available_templates:
        raise ValueError(
            f"Template '{template_name}' not found for phase '{phase}'. "
            f"Available templates: {available_templates}"
        )

    excluded = set(exclude_tasks or [])
    tasks_to_average = [t for t in all_task_ids if t not in excluded]
    if not tasks_to_average:
        raise ValueError("All tasks were excluded; nothing to average.")

    correct_means: list[list[float]] = []
    ctrl_means: list[list[float]] = []
    correct_support_all: list[list[float]] = []
    ctrl_support_all: list[list[float]] = []

    for task_id in tasks_to_average:
        key = (task_id, phase, template_name, version)
        if use_api_grading:
            correct_means.append([summary_stats[it].get(*key).correct_mean_api for it in iterations])
            ctrl_means.append([summary_stats[it].get(*key).control_mean_api for it in iterations])
            correct_support_all.append([summary_stats[it].get(*key).correct_support_all_api for it in iterations])
            ctrl_support_all.append([summary_stats[it].get(*key).control_support_all_api for it in iterations])
        else:
            correct_means.append([summary_stats[it].get(*key).correct_mean for it in iterations])
            ctrl_means.append([summary_stats[it].get(*key).control_mean for it in iterations])
            correct_support_all.append([summary_stats[it].get(*key).correct_support_all for it in iterations])
            ctrl_support_all.append([summary_stats[it].get(*key).control_support_all for it in iterations])

    correct_means_arr = np.asarray(correct_means, dtype=float)
    ctrl_means_arr = np.asarray(ctrl_means, dtype=float)
    correct_support_arr = np.asarray(correct_support_all, dtype=float)
    ctrl_support_arr = np.asarray(ctrl_support_all, dtype=float)

    def _average(values: np.ndarray, support_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid_mask = support_all > 0 if use_api_grading else np.ones_like(values, dtype=bool)
        counts = valid_mask.sum(axis=0)
        means = np.full(values.shape[1], np.nan, dtype=float)
        sems = np.full(values.shape[1], np.nan, dtype=float)
        if np.any(counts > 0):
            summed = np.where(valid_mask, values, 0.0).sum(axis=0)
            means = np.divide(summed, counts, out=means, where=counts > 0)
            centered = np.where(valid_mask, values - means, 0.0)
            variances = np.divide(
                (centered ** 2).sum(axis=0),
                counts,
                out=np.zeros(values.shape[1]),
                where=counts > 0,
            )
            sems = np.divide(
                np.sqrt(variances), np.sqrt(counts),
                out=sems, where=counts > 0,
            )
        return means, sems

    correct_mu, correct_sem = _average(correct_means_arr, correct_support_arr)
    ctrl_mu, ctrl_sem = _average(ctrl_means_arr, ctrl_support_arr)

    return {
        "correct_mu": correct_mu,
        "correct_sem": correct_sem,
        "ctrl_mu": ctrl_mu,
        "ctrl_sem": ctrl_sem,
        "iterations": np.asarray(iterations),
        "n_tasks": len(tasks_to_average),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PHASE_TRANSITION_STYLE = {
    "color": "#888888",
    "linestyle": ":",
    "linewidth": 1.5,
    "alpha": 0.8,
}


def _phase_color(template_name: str, phase: str) -> str:
    """Return the color to use for ``template_name`` in ``phase``.

    ``phase`` is the literal ``"learn"``, ``"update"``, or ``"forget"`` (use
    ``Phase.LEARN.value`` / ``Phase.UPDATE.value`` / ``Phase.FORGET.value``
    at call sites). Templates not in ``PLOT_STYLE["template_colors"]`` fall
    back to the global ``default_<phase>_color``.
    """
    template_colors = PLOT_STYLE["template_colors"].get(template_name)
    if template_colors is not None and phase in template_colors:
        return template_colors[phase]
    if phase == Phase.UPDATE.value:
        fallback_key = "default_update_color"
    elif phase == Phase.FORGET.value:
        fallback_key = "default_forget_color"
    else:
        fallback_key = "default_learn_color"
    return PLOT_STYLE[fallback_key]


def _draw_curves(
    ax,
    iterations,
    correct_mu,
    correct_sem,
    ctrl_mu,
    ctrl_sem,
    *,
    target_label: str | None = "Target",
    control_label: str | None = "Control",
    color: str | None = None,
) -> None:
    """Draw target/control mean curves with +/- SEM shading.

    Target and control share ``color`` (defaulting to
    ``PLOT_STYLE["default_learn_color"]``) and are distinguished by
    linestyle: target is drawn solid and control is drawn dashed (per
    ``PLOT_STYLE["target_linestyle"]`` / ``PLOT_STYLE["control_linestyle"]``).
    Pass ``label=None`` on subsequent calls to avoid duplicate legend entries
    when stacking multiple segments on the same axes.
    """
    iterations = np.asarray(iterations)
    correct_mu = np.asarray(correct_mu)
    correct_sem = np.asarray(correct_sem)
    ctrl_mu = np.asarray(ctrl_mu)
    ctrl_sem = np.asarray(ctrl_sem)

    color = color or PLOT_STYLE["default_learn_color"]
    target_ls = PLOT_STYLE["target_linestyle"]
    control_ls = PLOT_STYLE["control_linestyle"]

    target_kw = {"label": target_label} if target_label else {}
    control_kw = {"label": control_label} if control_label else {}

    ax.plot(iterations, correct_mu, color=color, linestyle=target_ls, **target_kw)
    ax.plot(iterations, ctrl_mu, color=color, linestyle=control_ls, **control_kw)
    ax.fill_between(
        iterations,
        correct_mu - correct_sem, correct_mu + correct_sem,
        color=color, alpha=PLOT_STYLE["fill_alpha"],
    )
    ax.fill_between(
        iterations,
        ctrl_mu - ctrl_sem, ctrl_mu + ctrl_sem,
        color=color, alpha=PLOT_STYLE["fill_alpha"],
    )


def _finalize_axis(
    ax,
    *,
    set_xlabel: bool,
    ylabel: str,
    iter_min,
    iter_max,
) -> None:
    """Apply shared axis styling (labels, ylim, ticks, legend, grid)."""
    if set_xlabel:
        ax.set_xlabel(PLOT_STYLE["xlabel"], fontsize=PLOT_STYLE["fontsize_xlabel"])
    ax.set_ylabel(ylabel, fontsize=PLOT_STYLE["fontsize_ylabel"])
    ax.set_ylim(PLOT_STYLE["ylim"])

    iter_min, iter_max = int(iter_min), int(iter_max)
    tick_positions = np.arange(iter_min, iter_max + 1)
    tick_interval = 20 if (iter_max - iter_min) > 50 else 5
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [str(p) if p % tick_interval == 0 else "" for p in tick_positions]
    )

    ax.legend(loc=PLOT_STYLE["legend_loc"], fontsize=PLOT_STYLE["fontsize_legend"])
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"])


def _safe_average(
    summary_stats,
    phase,
    template_name,
    *,
    version,
    use_api_grading,
    exclude_tasks,
):
    """Return ``average_results_across_tasks`` output or ``None`` if the
    phase/template is absent from ``summary_stats``."""
    try:
        return average_results_across_tasks(
            summary_stats,
            phase=phase,
            template_name=template_name,
            version=version,
            use_api_grading=use_api_grading,
            exclude_tasks=exclude_tasks,
        )
    except ValueError:
        return None


def plot_phase_results(
    summary_stats: dict[int, StatsRegistry],
    *,
    phase: str = Phase.LEARN.value,
    version: str = "1",
    template_names: tuple[str, ...] = DEFAULT_TEMPLATE_NAMES,
    use_api_grading: bool = False,
    exclude_tasks: list[str] | None = None,
    figsize_per_block: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot averaged verification + generation accuracy for a single phase.

    One vertically-stacked subplot per template in ``template_names``; the top
    panel is verification (``double_critic.j2``) and the bottom is generation
    (``generative_response.j2``). All curves are averaged across the tasks
    present in ``summary_stats`` (use ``exclude_tasks`` to drop specific ones).

    ``phase`` selects which subset of the per-iteration ``StatsRegistry`` keys
    to plot, typically ``Phase.LEARN.value`` or ``Phase.UPDATE.value``. The
    underlying data must obviously contain entries for the requested phase --
    a learn-only experiment has no ``UPDATE`` evaluations and vice versa.
    """
    n = len(template_names)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(figsize_per_block[0], figsize_per_block[1] * 1.5),
        sharex=False,
    )
    axes = np.atleast_1d(axes).flatten()

    for row, template_name in enumerate(template_names):
        ax = axes[row]
        avg = average_results_across_tasks(
            summary_stats,
            phase=phase,
            template_name=template_name,
            version=version,
            use_api_grading=use_api_grading,
            exclude_tasks=exclude_tasks,
        )
        _draw_curves(
            ax,
            avg["iterations"],
            avg["correct_mu"], avg["correct_sem"],
            avg["ctrl_mu"], avg["ctrl_sem"],
            color=_phase_color(template_name, phase),
        )
        _finalize_axis(
            ax,
            set_xlabel=(row == n - 1),
            ylabel=_TEMPLATE_YLABELS.get(template_name, "Accuracy"),
            iter_min=avg["iterations"].min(),
            iter_max=avg["iterations"].max(),
        )

    plt.tight_layout()
    return fig


def plot_learn_update_overlay(
    summary_stats: dict[int, StatsRegistry],
    *,
    learn_summary_stats: dict[int, StatsRegistry] | None = None,
    version: str = "1",
    template_names: tuple[str, ...] = DEFAULT_TEMPLATE_NAMES,
    use_api_grading: bool = False,
    exclude_tasks: list[str] | None = None,
    figsize_per_block: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot LEARN + UPDATE phase accuracy on the same axes per template.

    Intended for an *update* experiment whose ``summary_stats`` contains
    UPDATE-phase entries (and typically also LEARN-phase entries evaluated
    on the UPDATE-phase checkpoints). Up to three curve sets are drawn per
    template:

    * **LEARN phase on LEARN checkpoints** -- taken from
      ``learn_summary_stats`` when provided, cropped to
      ``iter <= first_update_iter`` so it spans iterations 0..boundary.
    * **LEARN phase on UPDATE checkpoints** -- taken from the LEARN entries
      inside ``summary_stats``, kept for ``iter >= first_update_iter``. Drawn
      in the same LEARN colors as the previous segment so visually the two
      LEARN segments meet at the phase-transition line.
    * **UPDATE phase on UPDATE checkpoints** -- taken from the UPDATE entries
      inside ``summary_stats`` over the full update iteration range.

    Color encodes (template, phase): verification + LEARN is red and
    verification + UPDATE is orange; generation + LEARN is blue and
    generation + UPDATE is green (see ``PLOT_STYLE["template_colors"]``).
    The orange/green UPDATE hues give a striking contrast with the
    red/blue LEARN hues on the same axes. Within each phase the target
    curve is drawn solid and the control curve dashed (per
    ``target_linestyle`` / ``control_linestyle``). A dotted vertical line
    at the first UPDATE-phase iteration marks the transition.

    If ``learn_summary_stats`` is omitted, LEARN data is read from
    ``summary_stats`` instead (no separate "LEARN on LEARN checkpoints"
    segment is drawn). If all phases are missing for a given template the
    function raises.
    """
    n = len(template_names)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(figsize_per_block[0], figsize_per_block[1] * 1.5),
        sharex=False,
    )
    axes = np.atleast_1d(axes).flatten()

    avg_kwargs = dict(
        version=version,
        use_api_grading=use_api_grading,
        exclude_tasks=exclude_tasks,
    )

    for row, template_name in enumerate(template_names):
        ax = axes[row]
        learn_color = _phase_color(template_name, Phase.LEARN.value)
        update_color = _phase_color(template_name, Phase.UPDATE.value)

        # LEARN-phase data, split by training-checkpoint source.
        if learn_summary_stats is not None:
            learn_on_learn = _safe_average(
                learn_summary_stats, Phase.LEARN.value, template_name, **avg_kwargs
            )
            learn_on_update = _safe_average(
                summary_stats, Phase.LEARN.value, template_name, **avg_kwargs
            )
        else:
            learn_on_learn = _safe_average(
                summary_stats, Phase.LEARN.value, template_name, **avg_kwargs
            )
            learn_on_update = None
        update = _safe_average(
            summary_stats, Phase.UPDATE.value, template_name, **avg_kwargs
        )

        if learn_on_learn is None and learn_on_update is None and update is None:
            raise ValueError(
                f"No data for template '{template_name}' in either LEARN or UPDATE phase."
            )

        transition_x: float | None = None
        if update is not None and len(update["iterations"]) > 0:
            transition_x = float(np.min(update["iterations"]))

        # Segment 1: LEARN phase on LEARN checkpoints (iter <= boundary).
        learn_label_used = False
        if learn_on_learn is not None and len(learn_on_learn["iterations"]) > 0:
            iters = learn_on_learn["iterations"]
            mask = (
                iters <= transition_x
                if transition_x is not None
                else np.ones_like(iters, dtype=bool)
            )
            if mask.any():
                _draw_curves(
                    ax,
                    iters[mask],
                    learn_on_learn["correct_mu"][mask], learn_on_learn["correct_sem"][mask],
                    learn_on_learn["ctrl_mu"][mask], learn_on_learn["ctrl_sem"][mask],
                    target_label="Target LEARN",
                    control_label="Control LEARN",
                    color=learn_color,
                )
                learn_label_used = True

        # Segment 2: LEARN phase on UPDATE checkpoints (iter >= boundary).
        if (
            learn_on_update is not None
            and len(learn_on_update["iterations"]) > 0
            and transition_x is not None
        ):
            iters = learn_on_update["iterations"]
            mask = iters >= transition_x
            if mask.any():
                _draw_curves(
                    ax,
                    iters[mask],
                    learn_on_update["correct_mu"][mask], learn_on_update["correct_sem"][mask],
                    learn_on_update["ctrl_mu"][mask], learn_on_update["ctrl_sem"][mask],
                    target_label=None if learn_label_used else "Target LEARN",
                    control_label=None if learn_label_used else "Control LEARN",
                    color=learn_color,
                )

        # Segment 3: UPDATE phase on UPDATE checkpoints.
        if update is not None and len(update["iterations"]) > 0:
            _draw_curves(
                ax,
                update["iterations"],
                update["correct_mu"], update["correct_sem"],
                update["ctrl_mu"], update["ctrl_sem"],
                target_label="Target UPDATE",
                control_label="Control UPDATE",
                color=update_color,
            )

        if transition_x is not None:
            ax.axvline(x=transition_x, zorder=1, **PHASE_TRANSITION_STYLE)

        all_iters: list[float] = []
        if learn_on_learn is not None:
            iters = learn_on_learn["iterations"]
            if transition_x is not None:
                iters = iters[iters <= transition_x]
            all_iters.extend(iters.tolist())
        if learn_on_update is not None and transition_x is not None:
            iters = learn_on_update["iterations"]
            iters = iters[iters >= transition_x]
            all_iters.extend(iters.tolist())
        if update is not None:
            all_iters.extend(update["iterations"].tolist())
        _finalize_axis(
            ax,
            set_xlabel=(row == n - 1),
            ylabel=_TEMPLATE_YLABELS.get(template_name, "Accuracy"),
            iter_min=min(all_iters),
            iter_max=max(all_iters),
        )

    plt.tight_layout()
    return fig


def plot_learn_forget_overlay(
    summary_stats: dict[int, StatsRegistry],
    *,
    learn_summary_stats: dict[int, StatsRegistry] | None = None,
    version: str = "1",
    template_names: tuple[str, ...] = DEFAULT_TEMPLATE_NAMES,
    use_api_grading: bool = False,
    exclude_tasks: list[str] | None = None,
    figsize_per_block: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot LEARN + FORGET phase accuracy on the same axes per template.

    Intended for a *forget* experiment whose ``summary_stats`` evaluates the
    LEARN-phase prompts on FORGET-phase checkpoints (the forget pipeline
    only re-uses the learn-phase prompts for evaluation, so every entry in
    ``summary_stats`` carries ``phase == Phase.LEARN.value``; the iteration
    indices encode the post-learn forget epochs). Up to two curve sets are
    drawn per template:

    * **LEARN phase on LEARN checkpoints** -- taken from
      ``learn_summary_stats`` when provided, cropped to
      ``iter <= first_forget_iter`` so it spans iterations 0..boundary.
    * **LEARN phase on FORGET checkpoints** -- taken from the LEARN entries
      inside ``summary_stats`` (kept for ``iter >= first_forget_iter`` when
      ``learn_summary_stats`` is given, otherwise the full range). Drawn
      in the FORGET hues so the regime change is visually obvious.

    Color encodes (template, phase): verification + LEARN is red and
    verification + FORGET is violet; generation + LEARN is blue and
    generation + FORGET is cyan (see ``PLOT_STYLE["template_colors"]``).
    Within each phase the target curve is drawn solid and the control
    curve dashed (per ``target_linestyle`` / ``control_linestyle``). A
    dotted vertical line at the first FORGET-phase iteration marks the
    transition.

    If ``learn_summary_stats`` is omitted, the FORGET-on-FORGET curves are
    drawn alone with no boundary line. If ``summary_stats`` has no
    LEARN-phase data the function raises.
    """
    n = len(template_names)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(figsize_per_block[0], figsize_per_block[1] * 1.5),
        sharex=False,
    )
    axes = np.atleast_1d(axes).flatten()

    avg_kwargs = dict(
        version=version,
        use_api_grading=use_api_grading,
        exclude_tasks=exclude_tasks,
    )

    for row, template_name in enumerate(template_names):
        ax = axes[row]
        learn_color = _phase_color(template_name, Phase.LEARN.value)
        forget_color = _phase_color(template_name, Phase.FORGET.value)

        learn_on_learn = (
            _safe_average(
                learn_summary_stats, Phase.LEARN.value, template_name, **avg_kwargs
            )
            if learn_summary_stats is not None
            else None
        )
        learn_on_forget = _safe_average(
            summary_stats, Phase.LEARN.value, template_name, **avg_kwargs
        )

        if learn_on_learn is None and learn_on_forget is None:
            raise ValueError(
                f"No LEARN-phase data for template '{template_name}' in "
                f"either learn_summary_stats or summary_stats."
            )

        transition_x: float | None = None
        if (
            learn_summary_stats is not None
            and learn_on_forget is not None
            and len(learn_on_forget["iterations"]) > 0
        ):
            transition_x = float(np.min(learn_on_forget["iterations"]))

        # Segment 1: LEARN phase on LEARN checkpoints (iter <= boundary).
        if learn_on_learn is not None and len(learn_on_learn["iterations"]) > 0:
            iters = learn_on_learn["iterations"]
            mask = (
                iters <= transition_x
                if transition_x is not None
                else np.ones_like(iters, dtype=bool)
            )
            if mask.any():
                _draw_curves(
                    ax,
                    iters[mask],
                    learn_on_learn["correct_mu"][mask], learn_on_learn["correct_sem"][mask],
                    learn_on_learn["ctrl_mu"][mask], learn_on_learn["ctrl_sem"][mask],
                    target_label="Target LEARN",
                    control_label="Control LEARN",
                    color=learn_color,
                )

        # Segment 2: LEARN phase on FORGET checkpoints (iter >= boundary).
        if learn_on_forget is not None and len(learn_on_forget["iterations"]) > 0:
            iters = learn_on_forget["iterations"]
            mask = (
                iters >= transition_x
                if transition_x is not None
                else np.ones_like(iters, dtype=bool)
            )
            if mask.any():
                _draw_curves(
                    ax,
                    iters[mask],
                    learn_on_forget["correct_mu"][mask], learn_on_forget["correct_sem"][mask],
                    learn_on_forget["ctrl_mu"][mask], learn_on_forget["ctrl_sem"][mask],
                    target_label="Target FORGET",
                    control_label="Control FORGET",
                    color=forget_color,
                )

        if transition_x is not None:
            ax.axvline(x=transition_x, zorder=1, **PHASE_TRANSITION_STYLE)

        all_iters: list[float] = []
        if learn_on_learn is not None:
            iters = learn_on_learn["iterations"]
            if transition_x is not None:
                iters = iters[iters <= transition_x]
            all_iters.extend(iters.tolist())
        if learn_on_forget is not None:
            iters = learn_on_forget["iterations"]
            if transition_x is not None:
                iters = iters[iters >= transition_x]
            all_iters.extend(iters.tolist())
        _finalize_axis(
            ax,
            set_xlabel=(row == n - 1),
            ylabel=_TEMPLATE_YLABELS.get(template_name, "Accuracy"),
            iter_min=min(all_iters),
            iter_max=max(all_iters),
        )

    plt.tight_layout()
    return fig
