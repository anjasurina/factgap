"""Prepare train/infer/grade sweeps from a single source config.

This script expands one Hydra sweep config into three stage-specific sweeps:

1. train
2. infer
3. grade

It replaces the older workflow where train/infer/grade each required separate
source YAML files.

In addition to ordinary "learn" sweeps, the script also supports compact
"update" and "forget" sweeps that inherit from a previously defined learn
sweep. Those compact recipes point at the learn sweep via ``_learn_config`` and
provide ``_learn_run_name`` plus ``_learn_num_epochs`` so the script can:

1. resolve the referenced learn config as the shared base
2. merge the compact update/forget recipe on top
3. force the appropriate phase flags and shared overrides
4. point update/forget training at the explicitly named learn run checkpoint

Each compact recipe must define exactly one of ``train.upd`` or
``train.forget``.

It also writes a fourth, unsuffixed experiment folder named exactly after the
base ``output.experiment_group``. That folder contains no generated configs.
Instead, it contains launch commands that pair the matching train and
infer commands for each sweep entry:

1. launch the train command for a given ``unique_id``
2. launch the infer command for that same ``unique_id``

This makes it possible to submit a single job per sweep entry that runs train
followed immediately by infer. Each generated train+infer pair is emitted as a
single guarded shell command so infer is skipped if train crashes, and the job
exits immediately on that failure.

The invoked sweep file controls ``output.experiment_group``: it is inferred
from that filename. If that file explicitly sets
``output.experiment_group``, it must match the filename exactly or the script
exits with an error.

Example
-------
    python3 src/origins/prepare_sweeps/prepare_experiments.py \
        --sweep_file_path src/origins/configs/experiments/learn_gemma3_4b.yaml \
        --train_num_gpus 4 \
        --infer_num_gpus 4 \
        --train_accelerate_config train_launch
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from origins.prepare_sweeps.generate_launch_commands import append_to_output_file, build_launch_command
from origins.prepare_sweeps.generate_runai_splits import (
    even_chunks,
    gather_existing_commands,
    list_existing_jobs,
    read_commands,
    write_job_file,
)
from origins.prepare_sweeps.generate_sweep_configs import (
    _build_unique_id,
    _extract_grid,
    _find_sentinel_values,
    _resolve_config_with_defaults,
    _validate_grid,
    apply_overrides,
)


# Canonical stage names used throughout the file when building per-stage configs,
# launch scripts, and output directory names.
TRAIN_STAGE = "train"
INFER_STAGE = "infer"
GRADE_STAGE = "grade"
STAGE_SUFFIXES = (f"_{TRAIN_STAGE}", f"_{INFER_STAGE}", f"_{GRADE_STAGE}")

# Default entrypoints for infer and grade stages when generating launch commands.
DEFAULT_INFER_RUN_COMMAND = "python3 src/origins/main_vllm.py"
DEFAULT_GRADE_RUN_COMMAND = "python3 src/origins/main_grading.py"
# Leave this empty to keep generated config filenames on the fallback stem
# (`default` for single-config recipes, otherwise the grid-derived unique id).
GENERATED_CONFIG_NAME_TEMPLATE_KEYS: tuple[str, ...] = ()

# ---------------------------------------------------------------------------
# Learn-derived recipe support
# ---------------------------------------------------------------------------
# Compact "update" and "forget" sweeps can point at a learn sweep via
# ``_learn_config``. During resolution, the learn config acts as a base and the
# recipe overwrites a small, fixed set of fields before final stage configs are
# written.

# Meta keys that exist only on compact sweeps that inherit from a learn config.
# These keys are consumed during source resolution and removed before schema
# validation / writing the final stage YAMLs.
LEARN_DERIVED_CONFIG_KEY = "_learn_config"
LEARN_DERIVED_RUN_NAME_KEY = "_learn_run_name"
LEARN_DERIVED_NUM_EPOCHS_KEY = "_learn_num_epochs"
LEARN_DERIVED_META_KEYS = (
    LEARN_DERIVED_CONFIG_KEY,
    LEARN_DERIVED_RUN_NAME_KEY,
    LEARN_DERIVED_NUM_EPOCHS_KEY,
)
# Compact learn-derived sweeps currently support update and forget recipes.
LEARN_DERIVED_RECIPE_UPDATE = "update"
LEARN_DERIVED_RECIPE_FORGET = "forget"
# Learn sweeps also emit a validation-only job folder that points back at the
# matching learn checkpoints and computes target_learn loss.
LEARN_VALIDATION_VARIANTS: tuple[tuple[str, str], ...] = (
    ("val_target", "target_learn"),
)
# Learn-derived forget sweeps emit a single validation-only job folder that
# computes target_learn loss against the per-epoch forget checkpoints.
FORGET_VALIDATION_VARIANTS: tuple[tuple[str, str], ...] = (
    ("val_target", "target_learn"),
)
# Learn-derived update sweeps emit a single validation-only job folder that
# computes target_learn loss against the per-epoch update checkpoints.
UPDATE_VALIDATION_VARIANTS: tuple[tuple[str, str], ...] = (
    ("val_target", "target_learn"),
)
# Learn-derived update sweeps always force these behavioral flags regardless of
# what the inherited learn config may contain.
UPDATE_STATIC_OVERRIDES: dict[str, Any] = {
    "train.upd.only_update_phase": True,
    "train.forget.only_forget_phase": False,
}
# Learn-derived forget sweeps similarly force the opposite phase flags.
FORGET_STATIC_OVERRIDES: dict[str, Any] = {
    "train.upd.only_update_phase": False,
    "train.forget.only_forget_phase": True,
}

# ---------------------------------------------------------------------------
# Stage materialization support
# ---------------------------------------------------------------------------
# Stage-local enable/disable flags that turn one shared config into the train,
# infer, or grade variant without changing the rest of the experiment settings.
STAGE_BASE_OVERRIDES: dict[str, dict[str, Any]] = {
    TRAIN_STAGE: {
        "train.enable_training": True,
        "infer.enable_inference": False,
        "grading.enable_grading": False,
    },
    INFER_STAGE: {
        "train.enable_training": False,
        "infer.enable_inference": True,
        "grading.enable_grading": False,
    },
    GRADE_STAGE: {
        "train.enable_training": False,
        "infer.enable_inference": False,
        "grading.enable_grading": True,
    },
}

@dataclass(frozen=True)
class StageSpec:
    name: str
    experiment_group: str
    configs_dir: Path
    launch_output_file: str
    splits_output_dir: Path
    num_gpus: int
    accelerate_config_path: Path | None
    mixed_precision: str | None
    run_command: str | None
    validation_data_source: str | None = None


@dataclass(frozen=True)
class CombinedLaunchSpec:
    experiment_group: str
    output_dir: Path
    launch_output_file: str
    splits_output_dir: Path


@dataclass(frozen=True)
class LearnDerivedSweepContext:
    learn_train_experiment_group: str
    learn_run_name: str
    train_stage_learning_rate: Any
    learn_num_epochs: int
    recipe_kind: str
    phase_num_epochs_key: str
    phase_num_epochs: int
    checkpoint_override_key: str


@dataclass(frozen=True)
class SharedSweepSource:
    shared_cfg: DictConfig
    grid: dict[str, list[Any]]
    learn_derived_context: LearnDerivedSweepContext | None


def _default(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _strip_stage_suffix(experiment_group: str) -> str:
    for suffix in STAGE_SUFFIXES:
        if experiment_group.endswith(suffix):
            return experiment_group[: -len(suffix)]
    return experiment_group


def _stage_experiment_group(base_experiment_group: str, stage_name: str) -> str:
    return f"{_strip_stage_suffix(base_experiment_group)}_{stage_name}"


def _path_relative_to_cwd(path: Path) -> Path:
    try:
        return Path(os.path.relpath(path, Path.cwd()))
    except ValueError:
        return path


def _append_yaml_suffix(filename: str) -> str:
    return filename if filename.endswith(".yaml") else f"{filename}.yaml"


def _sync_experiment_group_with_filename(cfg: DictConfig, config_path: Path) -> str:
    inferred_experiment_group = config_path.stem
    configured_experiment_group = OmegaConf.select(
        cfg, "output.experiment_group", default=None
    )
    if (
        configured_experiment_group is not None
        and configured_experiment_group != inferred_experiment_group
    ):
        raise SystemExit(
            "Error: output.experiment_group must exactly match the config filename. "
            f"Got '{configured_experiment_group}' in '{config_path}', expected "
            f"'{inferred_experiment_group}'."
        )

    OmegaConf.update(
        cfg,
        "output.experiment_group",
        inferred_experiment_group,
        merge=False,
    )
    return inferred_experiment_group


def _resolve_accelerate_config_path(name: str | None) -> Path | None:
    if name is None:
        return None
    return Path("accelerate_configs") / _append_yaml_suffix(name)


def _strip_generation_only_keys(
    cfg: DictConfig,
    extra_keys: tuple[str, ...] = (),
) -> None:
    hydra_section = cfg.get("hydra")
    if hydra_section is not None and hydra_section.get("sweeper") is not None:
        del hydra_section["sweeper"]

    if "defaults" in cfg:
        del cfg["defaults"]

    for key in extra_keys:
        if key in cfg:
            del cfg[key]


def _build_update_shared_overrides(
    cfg: DictConfig,
    learn_num_epochs: int,
) -> dict[str, Any]:
    """Overwrite shared fields when a compact update recipe inherits a learn sweep."""
    num_update_epochs = OmegaConf.select(cfg, "train.upd.num_update_epochs", default=0)
    return {
        "train.num_train_epochs": learn_num_epochs,
        "experiments_tracker.min_epochs": int(learn_num_epochs) + int(num_update_epochs) - 5,
        **UPDATE_STATIC_OVERRIDES,
    }


def _build_forget_shared_overrides(
    cfg: DictConfig,
    learn_num_epochs: int,
) -> dict[str, Any]:
    """Overwrite shared fields when a compact forget recipe inherits a learn sweep."""
    num_forget_epochs = OmegaConf.select(cfg, "train.forget.num_forget_epochs", default=0)
    return {
        "train.num_train_epochs": learn_num_epochs,
        "experiments_tracker.min_epochs": int(learn_num_epochs + 0.8 * num_forget_epochs),
        **FORGET_STATIC_OVERRIDES,
    }


def _resolve_learn_derived_recipe_kind(raw_sweep: DictConfig) -> str:
    has_update_recipe = OmegaConf.select(raw_sweep, "train.upd", default=None) is not None
    has_forget_recipe = OmegaConf.select(raw_sweep, "train.forget", default=None) is not None

    if has_update_recipe and has_forget_recipe:
        raise SystemExit(
            f"Error: {LEARN_DERIVED_CONFIG_KEY} sweeps must define exactly one learn-derived "
            "recipe type, but both 'train.upd' and 'train.forget' were provided."
        )
    if has_update_recipe:
        return LEARN_DERIVED_RECIPE_UPDATE
    if has_forget_recipe:
        return LEARN_DERIVED_RECIPE_FORGET

    raise SystemExit(
        f"Error: {LEARN_DERIVED_CONFIG_KEY} sweeps must define either 'train.upd' "
        "or 'train.forget'."
    )


def _learn_derived_recipe_overrides(
    recipe_kind: str,
    cfg: DictConfig,
    learn_num_epochs: int,
) -> tuple[dict[str, Any], str]:
    if recipe_kind == LEARN_DERIVED_RECIPE_UPDATE:
        return (
            _build_update_shared_overrides(
                cfg=cfg,
                learn_num_epochs=learn_num_epochs,
            ),
            "train.upd.checkpoint_to_load_for_update_training",
        )
    if recipe_kind == LEARN_DERIVED_RECIPE_FORGET:
        return (
            _build_forget_shared_overrides(
                cfg=cfg,
                learn_num_epochs=learn_num_epochs,
            ),
            "train.forget.checkpoint_to_load_for_forget_training",
        )

    raise SystemExit(f"Error: Unsupported learn-derived recipe kind: {recipe_kind}")


def _learn_derived_phase_num_epochs_info(
    recipe_kind: str,
    cfg: DictConfig,
) -> tuple[str, int]:
    if recipe_kind == LEARN_DERIVED_RECIPE_UPDATE:
        phase_num_epochs_key = "train.upd.num_update_epochs"
    elif recipe_kind == LEARN_DERIVED_RECIPE_FORGET:
        phase_num_epochs_key = "train.forget.num_forget_epochs"
    else:
        raise SystemExit(f"Error: Unsupported learn-derived recipe kind: {recipe_kind}")

    phase_num_epochs = OmegaConf.select(cfg, phase_num_epochs_key, default=0)
    return phase_num_epochs_key, int(phase_num_epochs)


def _resolve_learn_derived_sweep_source(
    source_cfg: DictConfig,
    source_grid: dict[str, list[Any]],
    config_dir: Path,
    recipe_kind: str,
) -> SharedSweepSource:
    """Resolve a compact update/forget recipe into one shared expanded config.

    Overwrite order:
    1. start from the referenced learn config
    2. merge the compact recipe on top
    3. force the recipe-specific shared overrides returned by
       ``_learn_derived_recipe_overrides``
    """
    learn_ref = OmegaConf.select(source_cfg, LEARN_DERIVED_CONFIG_KEY, default=None)
    if not isinstance(learn_ref, str) or not learn_ref.strip():
        raise SystemExit(
            f"Error: {LEARN_DERIVED_CONFIG_KEY} must be a non-empty config reference."
        )

    learn_run_name = OmegaConf.select(source_cfg, LEARN_DERIVED_RUN_NAME_KEY, default=None)
    if not isinstance(learn_run_name, str) or not learn_run_name.strip():
        raise SystemExit(
            f"Error: {LEARN_DERIVED_CONFIG_KEY} requires {LEARN_DERIVED_RUN_NAME_KEY} to be set."
        )
    learn_run_name = learn_run_name.strip()

    learn_num_epochs = OmegaConf.select(
        source_cfg, LEARN_DERIVED_NUM_EPOCHS_KEY, default=None
    )
    if learn_num_epochs is None:
        raise SystemExit(
            f"Error: {LEARN_DERIVED_CONFIG_KEY} requires {LEARN_DERIVED_NUM_EPOCHS_KEY} to be set."
        )

    learn_num_epochs = int(learn_num_epochs)

    learn_config_path = config_dir / f"{learn_ref.lstrip('/')}.yaml"
    if not learn_config_path.exists():
        raise SystemExit(
            f"Error: {LEARN_DERIVED_CONFIG_KEY} reference '{learn_ref}' resolved to "
            f"'{learn_config_path}' but the file does not exist."
        )

    raw_learn_cfg = OmegaConf.load(learn_config_path)
    learn_cfg = _resolve_config_with_defaults(learn_config_path, config_dir)

    learn_experiment_group = _sync_experiment_group_with_filename(
        raw_learn_cfg, learn_config_path
    )
    OmegaConf.update(
        learn_cfg,
        "output.experiment_group",
        learn_experiment_group,
        merge=False,
    )

    shared_cfg = OmegaConf.merge(deepcopy(learn_cfg), deepcopy(source_cfg))
    train_stage_learning_rate = OmegaConf.select(
        shared_cfg, "train.learning_rate", default=None
    )
    shared_overrides, checkpoint_override_key = _learn_derived_recipe_overrides(
        recipe_kind=recipe_kind,
        cfg=shared_cfg,
        learn_num_epochs=learn_num_epochs,
    )
    _apply_overrides_map(
        shared_cfg,
        shared_overrides,
    )
    phase_num_epochs_key, phase_num_epochs = _learn_derived_phase_num_epochs_info(
        recipe_kind=recipe_kind,
        cfg=shared_cfg,
    )

    # `_learn_run_name` points at one specific learn checkpoint, so compact
    # update/forget recipes now expand only their own sweep axes.
    effective_grid = {key: deepcopy(values) for key, values in source_grid.items()}
    if effective_grid:
        _validate_grid(effective_grid)

    _strip_generation_only_keys(shared_cfg, extra_keys=LEARN_DERIVED_META_KEYS)

    return SharedSweepSource(
        shared_cfg=shared_cfg,
        grid=effective_grid,
        learn_derived_context=LearnDerivedSweepContext(
            learn_train_experiment_group=_stage_experiment_group(
                learn_experiment_group, TRAIN_STAGE
            ),
            learn_run_name=learn_run_name,
            train_stage_learning_rate=train_stage_learning_rate,
            learn_num_epochs=learn_num_epochs,
            recipe_kind=recipe_kind,
            phase_num_epochs_key=phase_num_epochs_key,
            phase_num_epochs=phase_num_epochs,
            checkpoint_override_key=checkpoint_override_key,
        ),
    )


def _load_shared_source_config(
    sweep_file_path: Path,
    config_dir: Path,
    default_config_path: Path | None,
) -> SharedSweepSource:
    raw_sweep = OmegaConf.load(sweep_file_path)
    source_grid = _extract_grid(sweep_file_path)
    if source_grid:
        _validate_grid(source_grid)

    if OmegaConf.select(raw_sweep, LEARN_DERIVED_CONFIG_KEY, default=None) is not None:
        recipe_kind = _resolve_learn_derived_recipe_kind(raw_sweep)
        if "defaults" in raw_sweep:
            source_cfg = _resolve_config_with_defaults(sweep_file_path, config_dir)
        else:
            source_cfg = deepcopy(raw_sweep)
        return _resolve_learn_derived_sweep_source(
            source_cfg=source_cfg,
            source_grid=source_grid,
            config_dir=config_dir,
            recipe_kind=recipe_kind,
        )

    if "defaults" in raw_sweep:
        base_cfg = _resolve_config_with_defaults(sweep_file_path, config_dir)
    elif default_config_path is not None:
        fallback_cfg = OmegaConf.load(default_config_path)
        base_cfg = OmegaConf.merge(fallback_cfg, raw_sweep)
    else:
        raise SystemExit(
            "Error: The sweep file has no 'defaults' list and "
            "--default_config_path was not provided."
        )

    _strip_generation_only_keys(base_cfg)
    return SharedSweepSource(
        shared_cfg=base_cfg,
        grid=source_grid,
        learn_derived_context=None,
    )


def _find_unexpected_schema_paths(
    actual: Any,
    schema: Any,
    prefix: str = "",
) -> list[str]:
    actual_container = (
        OmegaConf.to_container(actual, resolve=False)
        if isinstance(actual, (DictConfig, ListConfig))
        else actual
    )
    schema_container = (
        OmegaConf.to_container(schema, resolve=False)
        if isinstance(schema, (DictConfig, ListConfig))
        else schema
    )

    unexpected: list[str] = []

    if isinstance(actual_container, dict):
        if not isinstance(schema_container, dict):
            return [prefix] if prefix else ["<root>"]

        for key, value in actual_container.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in schema_container:
                unexpected.append(path)
                continue
            unexpected.extend(
                _find_unexpected_schema_paths(value, schema_container[key], path)
            )
        return unexpected

    if isinstance(actual_container, list):
        if not isinstance(schema_container, list):
            return [prefix] if prefix else ["<root>"]
        if not schema_container:
            return unexpected

        schema_item = schema_container[0]
        for idx, item in enumerate(actual_container):
            path = f"{prefix}[{idx}]"
            unexpected.extend(_find_unexpected_schema_paths(item, schema_item, path))
        return unexpected

    return unexpected


def _validate_against_default_sweeps_schema(
    cfg: DictConfig,
    config_dir: Path,
) -> None:
    default_schema = _resolve_config_with_defaults(config_dir / "default_config.yaml", config_dir)
    if "defaults" in default_schema:
        del default_schema["defaults"]

    unexpected_paths = _find_unexpected_schema_paths(cfg, default_schema)
    if unexpected_paths:
        formatted = ", ".join(sorted(unexpected_paths))
        raise SystemExit(
            "Error: source config introduces fields that are not defined in "
            f"'{config_dir}/default_config.yaml': {formatted}"
        )


def _grid_items(grid: dict[str, list[Any]]) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    if not grid:
        return [(tuple(), {})]

    keys = list(grid.keys())
    return [
        (combo, dict(zip(keys, combo)))
        for combo in product(*grid.values())
    ]


def _grid_size(grid: dict[str, list[Any]]) -> int:
    if not grid:
        return 1
    return math.prod(len(values) for values in grid.values())


def _apply_overrides_map(cfg: DictConfig, overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        OmegaConf.update(cfg, dotted_key, deepcopy(value), merge=False)


def _build_generated_config_overrides(
    stage_name: str,
    unique_id: str,
    train_experiment_group: str,
    infer_experiment_group: str,
    overrides: dict[str, Any],
    learn_derived_context: LearnDerivedSweepContext | None,
) -> dict[str, Any]:
    """Build the final generated overrides for one stage config.

    These are the last values written into each emitted YAML, so they take
    precedence over the source sweep, inherited learn config, and grid values.
    """
    generated_overrides: dict[str, Any] = {
        "output.experiment_name": unique_id,
    }

    if learn_derived_context is not None:
        generated_overrides[learn_derived_context.checkpoint_override_key] = (
            _build_learn_derived_checkpoint_path(learn_derived_context)
        )
        if (
            learn_derived_context.recipe_kind == LEARN_DERIVED_RECIPE_UPDATE
            and learn_derived_context.train_stage_learning_rate is not None
        ):
            generated_overrides["train.learning_rate"] = (
                learn_derived_context.train_stage_learning_rate
            )
        if stage_name in (INFER_STAGE, GRADE_STAGE):
            phase_num_epochs = int(
                overrides.get(
                    learn_derived_context.phase_num_epochs_key,
                    learn_derived_context.phase_num_epochs,
                )
            )
            start_epoch = learn_derived_context.learn_num_epochs
            end_epoch = start_epoch + phase_num_epochs
            generated_overrides.update(
                {
                    "infer.checkpoint_params.start_epoch": start_epoch,
                    "infer.checkpoint_params.end_epoch": end_epoch,
                    "grading.start_epoch": start_epoch,
                    "grading.end_epoch": end_epoch,
                }
            )

    if stage_name == INFER_STAGE:
        checkpoint_path = _build_train_stage_checkpoint_path(
            train_experiment_group=train_experiment_group,
            unique_id=unique_id,
        )
        grading_results_dir = f"outputs/{infer_experiment_group}/{unique_id}"
        generated_overrides.update(
            {
                "infer.checkpoint_params.checkpoint_path": checkpoint_path,
                "grading.results_dir": grading_results_dir,
            }
        )

    if stage_name == GRADE_STAGE:
        grading_results_dir = f"outputs/{infer_experiment_group}/{unique_id}"
        checkpoint_path = _build_train_stage_checkpoint_path(
            train_experiment_group=train_experiment_group,
            unique_id=unique_id,
        )
        generated_overrides.update(
            {
                "grading.results_dir": grading_results_dir,
                "infer.checkpoint_params.checkpoint_path": checkpoint_path,
            }
        )

    return generated_overrides


def _build_generated_config_stem(cfg: DictConfig, fallback_stem: str) -> str:
    """Build the emitted YAML stem from configured template keys."""
    if not GENERATED_CONFIG_NAME_TEMPLATE_KEYS:
        return fallback_stem

    template_overrides: dict[str, Any] = {}
    for key in GENERATED_CONFIG_NAME_TEMPLATE_KEYS:
        value = OmegaConf.select(cfg, key, default=None)
        if value is None:
            return fallback_stem
        template_overrides[key] = value

    return _build_unique_id(template_overrides, Path(".")) or fallback_stem


def _apply_stage_base_overrides(
    cfg: DictConfig,
    stage_name: str,
    stage_experiment_group: str,
) -> None:
    overrides = {
        "output.experiment_group": stage_experiment_group,
        **STAGE_BASE_OVERRIDES[stage_name],
    }
    _apply_overrides_map(cfg, overrides)


def _build_train_stage_base(
    shared_cfg: DictConfig,
    stage_experiment_group: str,
) -> DictConfig:
    cfg = deepcopy(shared_cfg)
    _apply_stage_base_overrides(cfg, TRAIN_STAGE, stage_experiment_group)
    return cfg


def _build_infer_stage_base(
    shared_cfg: DictConfig,
    stage_experiment_group: str,
) -> DictConfig:
    cfg = deepcopy(shared_cfg)
    _apply_stage_base_overrides(cfg, INFER_STAGE, stage_experiment_group)
    return cfg


def _build_grade_stage_base(
    shared_cfg: DictConfig,
    stage_experiment_group: str,
) -> DictConfig:
    cfg = deepcopy(shared_cfg)
    _apply_stage_base_overrides(cfg, GRADE_STAGE, stage_experiment_group)
    return cfg


def _build_phase_checkpoint_save_overrides(cfg: DictConfig) -> dict[str, Any]:
    """Force per-epoch checkpoint saves for update/forget-only training runs."""
    overrides: dict[str, Any] = {}
    if OmegaConf.select(cfg, "train.upd.only_update_phase", default=False):
        overrides["train.upd.save_model_at_every_epoch"] = True
    if OmegaConf.select(cfg, "train.forget.only_forget_phase", default=False):
        overrides["train.forget.save_model_at_every_epoch"] = True
    return overrides


def _build_train_stage_checkpoint_path(
    train_experiment_group: str,
    unique_id: str,
) -> str:
    return f"outputs/{train_experiment_group}/{unique_id}"


def _should_build_learn_validation_stages(
    shared_cfg: DictConfig,
    learn_derived_context: LearnDerivedSweepContext | None,
) -> bool:
    if learn_derived_context is not None:
        return False

    if OmegaConf.select(shared_cfg, "train.upd.only_update_phase", default=False):
        return False
    if OmegaConf.select(shared_cfg, "train.forget.only_forget_phase", default=False):
        return False

    num_update_epochs = OmegaConf.select(
        shared_cfg,
        "train.upd.num_update_epochs",
        default=0,
    )
    num_forget_epochs = OmegaConf.select(
        shared_cfg,
        "train.forget.num_forget_epochs",
        default=0,
    )
    input_items = OmegaConf.to_container(
        OmegaConf.select(shared_cfg, "input", default=[]),
        resolve=False,
    )
    has_learn_phase = any(
        "learn" in (item.get("phases") or [])
        for item in input_items
        if isinstance(item, dict)
    )
    return has_learn_phase and int(num_update_epochs) == 0 and int(num_forget_epochs) == 0


def _should_build_forget_validation_stages(
    learn_derived_context: LearnDerivedSweepContext | None,
) -> bool:
    return (
        learn_derived_context is not None
        and learn_derived_context.recipe_kind == LEARN_DERIVED_RECIPE_FORGET
    )


def _should_build_update_validation_stages(
    learn_derived_context: LearnDerivedSweepContext | None,
) -> bool:
    return (
        learn_derived_context is not None
        and learn_derived_context.recipe_kind == LEARN_DERIVED_RECIPE_UPDATE
    )


def _build_validation_stage_base(
    shared_cfg: DictConfig,
    stage_experiment_group: str,
    validation_data_source: str,
) -> DictConfig:
    cfg = deepcopy(shared_cfg)
    _apply_overrides_map(
        cfg,
        {
            "output.experiment_group": stage_experiment_group,
            "train.enable_training": False,
            "infer.enable_inference": False,
            "grading.enable_grading": False,
            "train.validation.enable": True,
            "train.validation.data_source": validation_data_source,
        },
    )
    return cfg


def _build_learn_validation_generated_overrides(
    stage: StageSpec,
    cfg: DictConfig,
    unique_id: str,
    train_experiment_group: str,
    learn_derived_context: LearnDerivedSweepContext | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = _build_train_stage_checkpoint_path(
        train_experiment_group=train_experiment_group,
        unique_id=unique_id,
    )

    if learn_derived_context is not None and learn_derived_context.recipe_kind in (
        LEARN_DERIVED_RECIPE_FORGET,
        LEARN_DERIVED_RECIPE_UPDATE,
    ):
        phase_num_epochs = int(
            overrides.get(
                learn_derived_context.phase_num_epochs_key,
                learn_derived_context.phase_num_epochs,
            )
        )
        start_epoch = int(learn_derived_context.learn_num_epochs)
        end_epoch = start_epoch + phase_num_epochs
    else:
        num_train_epochs = OmegaConf.select(cfg, "train.num_train_epochs", default=None)
        if num_train_epochs is None:
            raise SystemExit(
                "Error: learn validation variants require 'train.num_train_epochs' to "
                "be set."
            )
        start_epoch = 0
        end_epoch = int(num_train_epochs)

    return {
        "output.experiment_name": unique_id,
        "infer.enable_inference": False,
        "train.enable_training": False,
        "grading.enable_grading": False,
        "train.validation.enable": True,
        "train.validation.data_source": stage.validation_data_source,
        "train.validation.checkpoint_path": checkpoint_path,
        "train.validation.start_epoch": start_epoch,
        "train.validation.end_epoch": end_epoch,
        "train.validation.frequency": 1,
    }


def _build_learn_derived_checkpoint_path(
    learn_derived_context: LearnDerivedSweepContext,
) -> str:
    checkpoint_dir = (
        Path("outputs")
        / learn_derived_context.learn_train_experiment_group
        / learn_derived_context.learn_run_name
        / f"model_after_epoch_{learn_derived_context.learn_num_epochs}"
    )
    return str(checkpoint_dir)


def _warn_on_remaining_sentinels(cfg: DictConfig, config_path: Path) -> None:
    sentinels = _find_sentinel_values(cfg)
    if sentinels:
        # Intentionally silent: detection is preserved for future use, but the
        # warning print is suppressed to keep sweep generation output clean.
        pass


def _write_generated_config_file(cfg: DictConfig, config_path: Path) -> Path:
    OmegaConf.save(config=cfg, f=str(config_path))
    _warn_on_remaining_sentinels(cfg, config_path)
    return config_path


def _reset_stage_outputs(stage: StageSpec) -> None:
    stage.configs_dir.mkdir(parents=True, exist_ok=True)

    for config_path in stage.configs_dir.glob("*.yaml"):
        config_path.unlink()

    launch_script_path = stage.configs_dir / stage.launch_output_file
    if launch_script_path.exists():
        launch_script_path.unlink()

    if stage.splits_output_dir.exists():
        shutil.rmtree(stage.splits_output_dir)


def _reset_combined_outputs(spec: CombinedLaunchSpec) -> None:
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    for config_path in spec.output_dir.glob("*.yaml"):
        config_path.unlink()

    launch_script_path = spec.output_dir / spec.launch_output_file
    if launch_script_path.exists():
        launch_script_path.unlink()

    if spec.splits_output_dir.exists():
        shutil.rmtree(spec.splits_output_dir)


def _write_stage_configs(
    stage: StageSpec,
    stage_base_cfg: DictConfig,
    grid: dict[str, list[Any]],
    train_experiment_group: str,
    infer_experiment_group: str,
    learn_derived_context: LearnDerivedSweepContext | None,
) -> list[Path]:
    stage.configs_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []
    generated_stems: set[str] = set()
    num_written_files = 0
    num_updated_files = 0
    for _, overrides in _grid_items(grid):
        cfg = deepcopy(stage_base_cfg)
        apply_overrides(cfg, overrides)
        if stage.name == TRAIN_STAGE:
            _apply_overrides_map(cfg, _build_phase_checkpoint_save_overrides(cfg))

        unique_id = _build_unique_id(overrides, stage.configs_dir) or "default"
        if stage.validation_data_source is None:
            generated_overrides = _build_generated_config_overrides(
                stage_name=stage.name,
                unique_id=unique_id,
                train_experiment_group=train_experiment_group,
                infer_experiment_group=infer_experiment_group,
                overrides=overrides,
                learn_derived_context=learn_derived_context,
            )
        else:
            generated_overrides = _build_learn_validation_generated_overrides(
                stage=stage,
                cfg=cfg,
                unique_id=unique_id,
                train_experiment_group=train_experiment_group,
                learn_derived_context=learn_derived_context,
                overrides=overrides,
            )

        _apply_overrides_map(cfg, generated_overrides)
        config_stem = _build_generated_config_stem(cfg, fallback_stem=unique_id)
        if config_stem in generated_stems:
            raise SystemExit(
                "Error: Generated config naming template produced duplicate stem "
                f"'{config_stem}' in '{stage.configs_dir}'. "
                "Update GENERATED_CONFIG_NAME_TEMPLATE_KEYS or the sweep grid to "
                "make each emitted config filename unique."
            )
        generated_stems.add(config_stem)

        config_path = stage.configs_dir / f"{config_stem}.yaml"
        generated_files.append(config_path)
        if config_path.exists():
            num_updated_files += 1
        else:
            num_written_files += 1
        _write_generated_config_file(cfg, config_path)

    print(
        f"[prepare_sweep_v2] Prepared {len(generated_files)} {stage.name} config file"
        f"{'' if len(generated_files) == 1 else 's'} in '{stage.configs_dir}' "
        f"({num_written_files} new, {num_updated_files} refreshed)."
    )
    return generated_files


def _generate_launch_script(stage: StageSpec, config_files: list[Path]) -> Path:
    launch_script_path = stage.configs_dir / stage.launch_output_file
    prepared_commands = 0

    for config_file in config_files:
        gradient_accumulation_steps = None
        if stage.accelerate_config_path is not None and stage.run_command is None:
            generated_cfg = OmegaConf.load(config_file)
            gradient_accumulation_steps = OmegaConf.select(
                generated_cfg,
                "train.gradient_accumulation_steps",
                default=None,
            )
            if gradient_accumulation_steps is not None:
                gradient_accumulation_steps = int(gradient_accumulation_steps)

        command = build_launch_command(
            config_file=_path_relative_to_cwd(config_file),
            num_gpus=stage.num_gpus,
            accelerate_config_path=stage.accelerate_config_path
            if stage.accelerate_config_path is not None
            else Path("accelerate_configs/deepspeed.yaml"),
            mixed_precision=stage.mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            run_command=stage.run_command,
        )
        append_to_output_file(command, launch_script_path)
        prepared_commands += 1

    print(
        f"[prepare_sweep_v2] Prepared launch script '{launch_script_path}' "
        f"for {prepared_commands} {stage.name} config(s), appending only new commands."
    )
    return launch_script_path


def _extract_config_name(command: str) -> str | None:
    marker = " --config-name "
    if marker not in command:
        return None
    return command.rsplit(marker, 1)[-1].split()[0]


def _pair_train_and_infer_commands(
    train_launch_script: Path,
    infer_launch_script: Path,
) -> list[str]:
    train_commands = read_commands(train_launch_script)
    infer_commands = read_commands(infer_launch_script)

    if len(train_commands) != len(infer_commands):
        raise SystemExit(
            "Error: train and infer launch scripts contain different numbers of "
            f"commands ({len(train_commands)} vs {len(infer_commands)})."
        )

    paired_commands: list[str] = []
    for idx, (train_command, infer_command) in enumerate(
        zip(train_commands, infer_commands, strict=True),
        start=1,
    ):
        train_config_name = _extract_config_name(train_command)
        infer_config_name = _extract_config_name(infer_command)
        if train_config_name != infer_config_name:
            raise SystemExit(
                "Error: train and infer commands do not align for sweep entry "
                f"{idx}: {train_config_name!r} != {infer_config_name!r}."
            )

        # Exit the whole job immediately if train or infer crashes so later
        # commands in the same split script are not started.
        paired_commands.append(f"{train_command} && {infer_command} || exit $?")

    return paired_commands


def _generate_combined_launch_script(
    spec: CombinedLaunchSpec,
    paired_commands: list[str],
) -> Path:
    launch_script_path = spec.output_dir / spec.launch_output_file
    for command in paired_commands:
        append_to_output_file(command, launch_script_path)

    print(
        f"[prepare_sweep_v2] Prepared combined train+infer launch script "
        f"'{launch_script_path}' for {len(paired_commands)} paired config(s), "
        "appending only new commands."
    )
    return launch_script_path


def _generate_runai_splits(
    input_file: Path,
    num_jobs: int,
    output_dir: Path,
) -> None:
    all_commands = read_commands(input_file)
    if not all_commands:
        raise SystemExit(f"No commands found in {input_file!s}")

    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = list_existing_jobs(output_dir)
    job_cmds_map, scheduled_cmds_flat = gather_existing_commands(jobs)
    scheduled_set = set(scheduled_cmds_flat)
    new_commands = [cmd for cmd in all_commands if cmd not in scheduled_set]

    if not jobs:
        chunks = even_chunks(all_commands, num_jobs)
        for idx, chunk in enumerate(chunks, 1):
            write_job_file(output_dir / f"job_{idx:02d}.sh", chunk)
        print(
            f"[prepare_sweep_v2] Wrote {len(chunks)} RunAI split file(s) "
            f"containing {len(all_commands)} command(s) to '{output_dir}'."
        )
        return

    if not new_commands:
        print(
            f"[prepare_sweep_v2] All commands are already present in '{output_dir}' "
            "nothing to do."
        )
        return

    cap_per_job = next((len(cmds) for _, cmds in job_cmds_map.items() if cmds), 1)
    num_new_jobs = math.ceil(len(new_commands) / cap_per_job)
    last_idx = max((idx for idx, _ in jobs), default=0)

    for i in range(1, num_new_jobs + 1):
        idx = last_idx + i
        chunk = new_commands[:cap_per_job]
        new_commands = new_commands[cap_per_job:]
        write_job_file(output_dir / f"job_{idx:02d}.sh", chunk)
        print(
            f"[prepare_sweep_v2] Created incremental RunAI split "
            f"'{output_dir / f'job_{idx:02d}.sh'}' with {len(chunk)} command(s)."
        )


def _build_stage_specs(
    output_root: Path,
    base_experiment_group: str,
    launch_output_file: str,
    splits_dir_name: str,
    train_num_gpus: int,
    infer_num_gpus: int,
    grade_num_gpus: int,
    train_accelerate_config: str,
    train_mixed_precision: str | None,
) -> tuple[StageSpec, StageSpec, StageSpec]:
    train_experiment_group = _stage_experiment_group(base_experiment_group, TRAIN_STAGE)
    infer_experiment_group = _stage_experiment_group(base_experiment_group, INFER_STAGE)
    grade_experiment_group = _stage_experiment_group(base_experiment_group, GRADE_STAGE)

    train_dir = output_root / train_experiment_group
    infer_dir = output_root / infer_experiment_group
    grade_dir = output_root / grade_experiment_group

    return (
        StageSpec(
            name=TRAIN_STAGE,
            experiment_group=train_experiment_group,
            configs_dir=train_dir,
            launch_output_file=launch_output_file,
            splits_output_dir=train_dir / splits_dir_name,
            num_gpus=train_num_gpus,
            accelerate_config_path=_resolve_accelerate_config_path(train_accelerate_config),
            mixed_precision=train_mixed_precision,
            run_command=None,
        ),
        StageSpec(
            name=INFER_STAGE,
            experiment_group=infer_experiment_group,
            configs_dir=infer_dir,
            launch_output_file=launch_output_file,
            splits_output_dir=infer_dir / splits_dir_name,
            num_gpus=infer_num_gpus,
            accelerate_config_path=None,
            mixed_precision=None,
            run_command=DEFAULT_INFER_RUN_COMMAND,
        ),
        StageSpec(
            name=GRADE_STAGE,
            experiment_group=grade_experiment_group,
            configs_dir=grade_dir,
            launch_output_file=launch_output_file,
            splits_output_dir=grade_dir / splits_dir_name,
            num_gpus=grade_num_gpus,
            accelerate_config_path=None,
            mixed_precision=None,
            run_command=DEFAULT_GRADE_RUN_COMMAND,
        ),
    )


def _build_validation_stage_specs(
    output_root: Path,
    base_experiment_group: str,
    launch_output_file: str,
    splits_dir_name: str,
    train_num_gpus: int,
    validation_accelerate_config: str,
    train_mixed_precision: str | None,
    variants: tuple[tuple[str, str], ...],
) -> tuple[StageSpec, ...]:
    base_group = _strip_stage_suffix(base_experiment_group)
    return tuple(
        StageSpec(
            name=stage_name,
            experiment_group=f"{base_group}_{stage_name}",
            configs_dir=output_root / f"{base_group}_{stage_name}",
            launch_output_file=launch_output_file,
            splits_output_dir=(output_root / f"{base_group}_{stage_name}") / splits_dir_name,
            num_gpus=train_num_gpus,
            accelerate_config_path=_resolve_accelerate_config_path(validation_accelerate_config),
            mixed_precision=train_mixed_precision,
            run_command=None,
            validation_data_source=data_source,
        )
        for stage_name, data_source in variants
    )


def _build_learn_validation_stage_specs(
    output_root: Path,
    base_experiment_group: str,
    launch_output_file: str,
    splits_dir_name: str,
    train_num_gpus: int,
    validation_accelerate_config: str,
    train_mixed_precision: str | None,
) -> tuple[StageSpec, ...]:
    return _build_validation_stage_specs(
        output_root=output_root,
        base_experiment_group=base_experiment_group,
        launch_output_file=launch_output_file,
        splits_dir_name=splits_dir_name,
        train_num_gpus=train_num_gpus,
        validation_accelerate_config=validation_accelerate_config,
        train_mixed_precision=train_mixed_precision,
        variants=LEARN_VALIDATION_VARIANTS,
    )


def _build_forget_validation_stage_specs(
    output_root: Path,
    base_experiment_group: str,
    launch_output_file: str,
    splits_dir_name: str,
    train_num_gpus: int,
    validation_accelerate_config: str,
    train_mixed_precision: str | None,
) -> tuple[StageSpec, ...]:
    return _build_validation_stage_specs(
        output_root=output_root,
        base_experiment_group=base_experiment_group,
        launch_output_file=launch_output_file,
        splits_dir_name=splits_dir_name,
        train_num_gpus=train_num_gpus,
        validation_accelerate_config=validation_accelerate_config,
        train_mixed_precision=train_mixed_precision,
        variants=FORGET_VALIDATION_VARIANTS,
    )


def _build_update_validation_stage_specs(
    output_root: Path,
    base_experiment_group: str,
    launch_output_file: str,
    splits_dir_name: str,
    train_num_gpus: int,
    validation_accelerate_config: str,
    train_mixed_precision: str | None,
) -> tuple[StageSpec, ...]:
    return _build_validation_stage_specs(
        output_root=output_root,
        base_experiment_group=base_experiment_group,
        launch_output_file=launch_output_file,
        splits_dir_name=splits_dir_name,
        train_num_gpus=train_num_gpus,
        validation_accelerate_config=validation_accelerate_config,
        train_mixed_precision=train_mixed_precision,
        variants=UPDATE_VALIDATION_VARIANTS,
    )


def _build_combined_launch_spec(
    output_root: Path,
    base_experiment_group: str,
    launch_output_file: str,
    splits_dir_name: str,
) -> CombinedLaunchSpec:
    combined_experiment_group = _strip_stage_suffix(base_experiment_group)
    output_dir = output_root / combined_experiment_group
    return CombinedLaunchSpec(
        experiment_group=combined_experiment_group,
        output_dir=output_dir,
        launch_output_file=launch_output_file,
        splits_output_dir=output_dir / splits_dir_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare train/infer/grade sweeps from a single sweep config by "
            "writing final stage configs, launch commands, and RunAI splits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sweep_file_path",
        required=True,
        help="Path to the single source sweep YAML file.",
    )
    parser.add_argument(
        "--config_dir",
        default="src/origins/configs",
        help="Root directory for resolving '/'-prefixed defaults references.",
    )
    parser.add_argument(
        "--default_config_path",
        default=None,
        help="Optional fallback base config if the source sweep has no defaults list.",
    )
    parser.add_argument(
        "--output_root",
        default="experiments",
        help="Root directory under which stage-specific experiment folders are created.",
    )
    parser.add_argument(
        "--launch_output_file",
        default="all_jobs.sh",
        help="Name of the launch script written into each stage folder.",
    )
    parser.add_argument(
        "--splits_dir_name",
        default="runai_splits",
        help="Directory name for RunAI split files inside each stage folder.",
    )

    parser.add_argument("--train_num_gpus", type=int, default=4, help="Number of GPUs per training job.")
    parser.add_argument("--infer_num_gpus", type=int, default=4, help="Number of GPUs per inference job.")
    parser.add_argument("--grade_num_gpus", type=int, default=1, help="Number of GPUs per grading job.")
    parser.add_argument(
        "--train_accelerate_config",
        default="train_launch",
        help="Name of the accelerate config in accelerate_configs/ for training launches.",
    )
    parser.add_argument(
        "--validation_accelerate_config",
        default="train_launch",
        help=(
            "Name of the accelerate config in accelerate_configs/ for validation-only "
            "launches. Defaults to 'train_launch' regardless of "
            "--train_accelerate_config."
        ),
    )
    parser.add_argument(
        "--train_mixed_precision",
        choices=["no", "fp16", "bf16"],
        default=None,
        help="Optional mixed-precision override for training launches.",
    )

    args = parser.parse_args()
    validation_accelerate_config = args.validation_accelerate_config

    sweep_file_path = Path(args.sweep_file_path)
    config_dir = Path(args.config_dir)
    default_config_path = (
        Path(args.default_config_path) if args.default_config_path is not None else None
    )
    output_root = Path(args.output_root)

    if not sweep_file_path.exists():
        raise SystemExit(f"Error: Sweep file not found: {sweep_file_path}")
    if not config_dir.is_dir():
        raise SystemExit(f"Error: Config directory not found: {config_dir}")
    if default_config_path is not None and not default_config_path.exists():
        raise SystemExit(f"Error: Default config not found: {default_config_path}")

    raw_sweep_cfg = OmegaConf.load(sweep_file_path)
    source = _load_shared_source_config(
        sweep_file_path=sweep_file_path,
        config_dir=config_dir,
        default_config_path=default_config_path,
    )
    shared_cfg = source.shared_cfg
    grid = source.grid
    base_experiment_group = _sync_experiment_group_with_filename(
        raw_sweep_cfg, sweep_file_path
    )
    OmegaConf.update(
        shared_cfg,
        "output.experiment_group",
        base_experiment_group,
        merge=False,
    )
    _validate_against_default_sweeps_schema(shared_cfg, config_dir)

    grid_size = _grid_size(grid)
    print(f"[prepare_sweep_v2] Sweep grid size: {grid_size}")

    train_stage, infer_stage, grade_stage = _build_stage_specs(
        output_root=output_root,
        base_experiment_group=base_experiment_group,
        launch_output_file=args.launch_output_file,
        splits_dir_name=args.splits_dir_name,
        train_num_gpus=args.train_num_gpus,
        infer_num_gpus=args.infer_num_gpus,
        grade_num_gpus=args.grade_num_gpus,
        train_accelerate_config=args.train_accelerate_config,
        train_mixed_precision=args.train_mixed_precision,
    )
    combined_launch = _build_combined_launch_spec(
        output_root=output_root,
        base_experiment_group=base_experiment_group,
        launch_output_file=args.launch_output_file,
        splits_dir_name=args.splits_dir_name,
    )

    train_base = _build_train_stage_base(
        shared_cfg,
        train_stage.experiment_group,
    )
    infer_base = _build_infer_stage_base(
        shared_cfg,
        infer_stage.experiment_group,
    )
    grade_base = _build_grade_stage_base(
        shared_cfg,
        grade_stage.experiment_group,
    )
    learn_validation_stages: tuple[StageSpec, ...] = ()
    if _should_build_learn_validation_stages(
        shared_cfg=shared_cfg,
        learn_derived_context=source.learn_derived_context,
    ):
        learn_validation_stages = _build_learn_validation_stage_specs(
            output_root=output_root,
            base_experiment_group=base_experiment_group,
            launch_output_file=args.launch_output_file,
            splits_dir_name=args.splits_dir_name,
            train_num_gpus=args.train_num_gpus,
            validation_accelerate_config=validation_accelerate_config,
            train_mixed_precision=args.train_mixed_precision,
        )

    forget_validation_stages: tuple[StageSpec, ...] = ()
    if _should_build_forget_validation_stages(source.learn_derived_context):
        forget_validation_stages = _build_forget_validation_stage_specs(
            output_root=output_root,
            base_experiment_group=base_experiment_group,
            launch_output_file=args.launch_output_file,
            splits_dir_name=args.splits_dir_name,
            train_num_gpus=args.train_num_gpus,
            validation_accelerate_config=validation_accelerate_config,
            train_mixed_precision=args.train_mixed_precision,
        )

    update_validation_stages: tuple[StageSpec, ...] = ()
    if _should_build_update_validation_stages(source.learn_derived_context):
        update_validation_stages = _build_update_validation_stage_specs(
            output_root=output_root,
            base_experiment_group=base_experiment_group,
            launch_output_file=args.launch_output_file,
            splits_dir_name=args.splits_dir_name,
            train_num_gpus=args.train_num_gpus,
            validation_accelerate_config=validation_accelerate_config,
            train_mixed_precision=args.train_mixed_precision,
        )

    launch_script_paths: dict[str, Path] = {}
    stage_jobs: list[tuple[StageSpec, DictConfig]] = [
        (train_stage, train_base),
        (infer_stage, infer_base),
        (grade_stage, grade_base),
    ]
    for stage in (
        *learn_validation_stages,
        *forget_validation_stages,
        *update_validation_stages,
    ):
        stage_jobs.append(
            (
                stage,
                _build_validation_stage_base(
                    shared_cfg=shared_cfg,
                    stage_experiment_group=stage.experiment_group,
                    validation_data_source=stage.validation_data_source or "target_learn",
                ),
            )
        )

    for stage, _ in stage_jobs:
        _reset_stage_outputs(stage)
    _reset_combined_outputs(combined_launch)

    for stage, stage_base_cfg in stage_jobs:
        config_files = _write_stage_configs(
            stage=stage,
            stage_base_cfg=stage_base_cfg,
            grid=grid,
            train_experiment_group=train_stage.experiment_group,
            infer_experiment_group=infer_stage.experiment_group,
            learn_derived_context=source.learn_derived_context,
        )
        launch_script_path = _generate_launch_script(stage, config_files)
        launch_script_paths[stage.name] = launch_script_path
        _generate_runai_splits(
            input_file=launch_script_path,
            num_jobs=grid_size,
            output_dir=stage.splits_output_dir,
        )

    paired_commands = _pair_train_and_infer_commands(
        train_launch_script=launch_script_paths[TRAIN_STAGE],
        infer_launch_script=launch_script_paths[INFER_STAGE],
    )
    combined_launch_script_path = _generate_combined_launch_script(
        combined_launch,
        paired_commands,
    )
    _generate_runai_splits(
        input_file=combined_launch_script_path,
        num_jobs=grid_size,
        output_dir=combined_launch.splits_output_dir,
    )

    print("[prepare_sweep_v2] All done, your stage sweeps are ready in", output_root)


if __name__ == "__main__":
    main()
