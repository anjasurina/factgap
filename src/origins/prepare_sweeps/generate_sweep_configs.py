"""
Pregenerate a set of Hydra-compatible YAML config files corresponding to the
Cartesian product of a user-defined sweep grid.  Each generated file contains
the full default configuration with the desired overrides applied.  The files
are written to a dedicated folder so that they can later be consumed by a
launcher.
"""

from __future__ import annotations
from itertools import product
from pathlib import Path
from copy import deepcopy
from typing import Any
import hashlib
import re
import sys
from omegaconf import OmegaConf, DictConfig, ListConfig
import argparse
import yaml

SENTINEL_VALUE = "overwriteme"
_TRAIN_CONFIG_KEY = "_train_config"
_TRAIN_INHERIT_KEYS = ["model.name", "model.attn_implementation", "input", "infer.inference_tasks_dir"]

_LEARN_CONFIG_KEY = "_learn_config"
_LEARN_CONFIG_LR_KEY = "_learn_config_lr"
_LEARN_CHECKPOINT_EPOCH_KEY = "_learn_checkpoint_epoch"

_RUN_COMMAND_KEY = "_run_command"


def apply_overrides(cfg: OmegaConf, overrides: dict[str, object]) -> None:
    """Apply a set of dotted-key overrides to a *mutable* OmegaConf object."""

    for dotted_key, value in overrides.items():
        # ``merge=False`` ensures that we do an *in-place* update without
        # merging dictionaries – which would ruin any existing default lists.
        OmegaConf.update(cfg, dotted_key, value, merge=False)


def _apply_phase_checkpoint_save_overrides(cfg: OmegaConf) -> None:
    """Force per-epoch checkpoint saves for update/forget-only training runs."""
    if OmegaConf.select(cfg, "train.upd.only_update_phase", default=False):
        OmegaConf.update(cfg, "train.upd.save_model_at_every_epoch", True, merge=False)
    if OmegaConf.select(cfg, "train.forget.only_forget_phase", default=False):
        OmegaConf.update(cfg, "train.forget.save_model_at_every_epoch", True, merge=False)


def _resolve_config_with_defaults(
    config_path: Path,
    config_dir: Path,
    _stack: frozenset[Path] | None = None,
) -> DictConfig:
    """Recursively resolve a config file's Hydra-style ``defaults`` list.

    Each entry in ``defaults`` is interpreted as a ``/``-prefixed path relative
    to *config_dir* (e.g. ``/default_sweeps`` -> ``config_dir/default_sweeps.yaml``).
    Referenced configs are loaded and merged left-to-right; the special
    ``_self_`` token determines where the current file's own keys appear in the
    merge order.

    Returns a single :class:`DictConfig` with all defaults composed and the
    ``defaults`` key itself removed.
    """

    if _stack is None:
        _stack = frozenset()

    abs_path = config_path.resolve()
    if abs_path in _stack:
        raise ValueError(
            f"Circular defaults chain detected: {config_path} is already "
            f"being resolved (chain so far: {_stack})"
        )
    _stack = _stack | {abs_path}

    try:
        raw_cfg = OmegaConf.load(config_path)
    except (yaml.YAMLError, Exception) as exc:
        raise RuntimeError(
            f"Failed to load '{config_path}' as valid YAML: {exc}"
        ) from exc

    defaults_raw = raw_cfg.pop("defaults", None)

    if defaults_raw is None:
        return raw_cfg

    defaults_list = OmegaConf.to_container(defaults_raw)
    if not isinstance(defaults_list, list):
        defaults_list = [defaults_list]

    layers_before_self: list[DictConfig] = []
    layers_after_self: list[DictConfig] = []
    target = layers_before_self

    for entry in defaults_list:
        if isinstance(entry, str) and entry.strip() == "_self_":
            target = layers_after_self
            continue

        if isinstance(entry, str):
            ref = entry.lstrip("/")
        elif isinstance(entry, dict):
            ref = "/".join(str(v) for v in entry.values())
        else:
            print(
                f"[WARNING] Ignoring unrecognised defaults entry {entry!r} in {config_path}",
                file=sys.stderr,
            )
            continue

        ref_file = config_dir / f"{ref}.yaml"
        if not ref_file.exists():
            raise FileNotFoundError(
                f"Defaults reference '{entry}' in {config_path} resolved to "
                f"'{ref_file}' but the file does not exist."
            )

        resolved = _resolve_config_with_defaults(ref_file, config_dir, _stack)
        target.append(resolved)

    result = OmegaConf.create({})
    for layer in layers_before_self:
        result = OmegaConf.merge(result, layer)
    result = OmegaConf.merge(result, raw_cfg)
    for layer in layers_after_self:
        result = OmegaConf.merge(result, layer)

    return result


def _find_sentinel_values(
    cfg: Any, sentinel: str = SENTINEL_VALUE, prefix: str = "",
) -> list[str]:
    """Return a list of dotted-key paths whose value equals *sentinel*.

    Converts the config to a plain Python container first (without resolving
    interpolations) so that Hydra-specific resolvers don't cause errors.
    """

    if isinstance(cfg, (DictConfig, ListConfig)):
        cfg = OmegaConf.to_container(cfg, resolve=False)

    hits: list[str] = []
    if isinstance(cfg, dict):
        for key, val in cfg.items():
            full_key = f"{prefix}.{key}" if prefix else key
            hits.extend(_find_sentinel_values(val, sentinel, full_key))
    elif isinstance(cfg, (list, tuple)):
        for idx, val in enumerate(cfg):
            full_key = f"{prefix}[{idx}]"
            hits.extend(_find_sentinel_values(val, sentinel, full_key))
    elif isinstance(cfg, str) and cfg == sentinel:
        hits.append(prefix)
    return hits


def _convert_scalar(value: Any) -> Any:
    """Best-effort conversion of *value* to int / float / bool / str.

    If *value* is not a string, it is returned unchanged.  When *value* is a
    string, the function attempts to interpret it as a boolean, integer, or
    floating-point number before ultimately falling back to the original
    string (with surrounding quotes stripped, if any).
    """

    # Short-circuit for non-string inputs (they may already be correctly typed).
    if not isinstance(value, str):
        return value

    text = value.strip()

    # Remove wrapping quotes (single or double) **once** if present so that
    # strings like "'foo'" or '"foo"' are handled gracefully.
    if (
        (text.startswith("'") and text.endswith("'"))
        or (text.startswith('"') and text.endswith('"'))
    ) and len(text) >= 2:
        text = text[1:-1].strip()

    # Bool
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False

    # Int or float
    try:
        if text.startswith("0") and len(text) > 1 and text[1].isdigit():
            # Preserve leading zeros (e.g. "0123") – treat as str
            raise ValueError
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text  # Fallback: keep as raw string


def _parse_grid_from_sweep(sweep_cfg: OmegaConf) -> dict[str, list[Any]]:
    """Extract ``hydra.sweeper.params`` into a Python dict of lists."""

    params = (sweep_cfg.get("hydra") or {}).get("sweeper", {}).get("params", {})

    grid: dict[str, list[Any]] = {}
    for key, raw in params.items():
        # ``raw`` is normally a comma-separated string – but tolerate list/tuple.
        if isinstance(raw, str):
            items = [itm.strip() for itm in raw.split(",") if itm.strip()]
        elif isinstance(raw, (list, tuple)):
            items = raw
        else:
            items = [raw]

        grid[key] = [_convert_scalar(item) for item in items]

    return grid


def _validate_grid(grid: dict[str, list[Any]]) -> None:
    """Run a few simple sanity checks on *grid*."""

    assert isinstance(grid, dict), "Grid must be a dictionary."
    for key, values in grid.items():
        assert isinstance(key, str) and key.strip(), f"Invalid grid key: {key!r}"
        assert isinstance(
            values, (list, tuple)
        ), f"Values for {key!r} must be a list or tuple."
        assert values, f"Values list for {key!r} is empty."

        # Ensure each item is of an expected primitive type so that we do not
        # silently pass complex objects down the line (which would break Hydra
        # later on).
        allowed_types = (str, int, float, bool)
        for val in values:
            assert isinstance(
                val, allowed_types
            ), (
                f"Unsupported type {type(val).__name__!s} for value {val!r} "
                f"in grid entry {key!r}. Allowed types: {', '.join(t.__name__ for t in allowed_types)}."
            )


def _sanitize_filename(text: str) -> str:
    """Return *text* without characters that are unsuitable for filenames."""

    return re.sub(r"[^A-Za-z0-9._-]", "", text)


def _build_unique_id(overrides: dict[str, Any], output_dir: Path) -> Path:
    """Return a deterministic path for the current *overrides*.

    The stem is composed from the trailing key segment and value of each
    override (order-preserving). Unsafe characters are stripped and very long
    filenames are shortened using an MD5 digest to stay below typical
    filesystem limits.
    """

    parts = [
        _sanitize_filename(f"{key.split('.')[-1]}{str(val).replace('.', 'p')}")
        for key, val in overrides.items()
    ]

    stem = "_".join(parts)

    if len(stem) > 180:
        digest = hashlib.md5(stem.encode()).hexdigest()[:10]
        stem = f"{stem[:170]}_{digest}"

    return stem


def _extract_grid(sweep_path: Path) -> dict[str, list[Any]]:
    """Load *sweep_path* (raw, without defaults resolution) and return the
    sweep grid from ``hydra.sweeper.params``.

    Returns an empty dict when no sweeper params are present (the grid may
    be supplied later via ``_train_config``).
    """

    try:
        sweep_cfg = OmegaConf.load(sweep_path)
    except (yaml.YAMLError, Exception) as exc:
        raise RuntimeError(
            "Failed to load sweep file as valid YAML. If your parameter values "
            "contain commas, ensure they are provided as a *single* YAML scalar "
            "string, for example: \n\n    model.name: \"foo, bar, baz\"\n\n"
            "or use YAML list syntax: \n\n    model.name: [foo, bar, baz]\n\n"
            "Original YAML error: " + str(exc)
        ) from exc

    grid = _parse_grid_from_sweep(sweep_cfg)
    if grid:
        _validate_grid(grid)
    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grid of YAML config files for a sweep.",
    )
    parser.add_argument(
        "--sweep_file_path",
        default="src/origins/configs/final_sweeps/experiments/gemma3_4b_batch_train.yaml",
        help="Path to the sweep YAML file.",
    )
    parser.add_argument(
        "--config_dir",
        default="src/origins/configs",
        help=(
            "Root directory used to resolve '/'-prefixed defaults references "
            "inside YAML configs (e.g. '/default_sweeps' -> config_dir/default_sweeps.yaml)."
        ),
    )
    parser.add_argument(
        "--default_config_path",
        default=None,
        help=(
            "(Optional, legacy) Path to a base config YAML file. "
            "Only used as a fallback when the sweep file has no 'defaults' list."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Path to the experiments directory.",
    )
    args = parser.parse_args()

    SWEEP_FILE = Path(args.sweep_file_path)
    CONFIG_DIR = Path("src/origins/configs") # Root directory used to resolve '/'-prefixed defaults references inside YAML configs
    OUTPUT_DIR = Path(args.output_dir)

    if not SWEEP_FILE.exists():
        sys.exit(f"Error: Sweep file not found: {SWEEP_FILE.resolve()}")
    if not CONFIG_DIR.is_dir():
        sys.exit(f"Error: Config directory not found: {CONFIG_DIR.resolve()}")

    # ------------------------------------------------------------------
    # 1) Extract the sweep grid (hydra.sweeper.params) from the raw file
    # ------------------------------------------------------------------
    grid = _extract_grid(SWEEP_FILE)
    keys = list(grid.keys())

    # ------------------------------------------------------------------
    # 2) Resolve the full base config by following the defaults chain
    # ------------------------------------------------------------------
    raw_sweep = OmegaConf.load(SWEEP_FILE)
    has_defaults = "defaults" in raw_sweep

    if has_defaults:
        print(f"[generate_sweep_configs] Resolving defaults chain from {SWEEP_FILE} …")
        base_cfg = _resolve_config_with_defaults(SWEEP_FILE, CONFIG_DIR)
    elif args.default_config_path is not None:
        fallback = Path(args.default_config_path)
        assert fallback.exists(), f"Base config not found: {fallback}"
        print(
            f"[generate_sweep_configs] No 'defaults' in sweep file; "
            f"falling back to --default_config_path={fallback}"
        )
        base_cfg = OmegaConf.load(fallback)
        base_cfg = OmegaConf.merge(base_cfg, raw_sweep)
    else:
        sys.exit(
            "Error: The sweep file has no 'defaults' list and "
            "--default_config_path was not provided."
        )

    # Remove the hydra.sweeper block – the grid has already been extracted
    hydra_section = base_cfg.get("hydra")
    if hydra_section is not None and hydra_section.get("sweeper") is not None:
        del hydra_section["sweeper"]

    # ------------------------------------------------------------------
    # 2b) If _train_config is present, inherit grid + fields from the
    #     referenced training config.
    # ------------------------------------------------------------------
    train_experiment_group: str | None = None
    infer_experiment_group: str | None = None

    if _TRAIN_CONFIG_KEY in base_cfg:
        train_ref = str(base_cfg.pop(_TRAIN_CONFIG_KEY))
        ref_path = train_ref.lstrip("/")
        train_config_path = CONFIG_DIR / f"{ref_path}.yaml"
        if not train_config_path.exists():
            sys.exit(
                f"Error: {_TRAIN_CONFIG_KEY} reference '{train_ref}' resolved to "
                f"'{train_config_path}' but the file does not exist."
            )

        resolved_train = _resolve_config_with_defaults(train_config_path, CONFIG_DIR)

        # Use the train config's sweep grid
        train_grid = _parse_grid_from_sweep(resolved_train)
        if not train_grid:
            train_grid = _parse_grid_from_sweep(OmegaConf.load(train_config_path))
        if train_grid:
            _validate_grid(train_grid)
            if grid:
                print(
                    f"[WARNING] Sweep file has its own grid which will be "
                    f"overridden by the grid from {_TRAIN_CONFIG_KEY}.",
                    file=sys.stderr,
                )
            grid = train_grid
            keys = list(grid.keys())
            print(f"[generate_sweep_configs] Using sweep grid from {_TRAIN_CONFIG_KEY}: {keys}")

        # Store the train experiment group for checkpoint path construction
        train_experiment_group = OmegaConf.select(
            resolved_train, "output.experiment_group", default=None,
        )
        if not train_experiment_group:
            sys.exit(
                f"Error: {_TRAIN_CONFIG_KEY} config has no output.experiment_group"
            )

        # Inherit specified fields from the resolved train config
        for dotted_key in _TRAIN_INHERIT_KEYS:
            val = OmegaConf.select(resolved_train, dotted_key, default=None)
            if val is not None:
                OmegaConf.update(base_cfg, dotted_key, deepcopy(val), merge=False)

        infer_experiment_group = train_experiment_group.replace("train", "infer")

        print(
            f"[generate_sweep_configs] Inherited {_TRAIN_INHERIT_KEYS} from "
            f"train config (experiment_group='{train_experiment_group}', "
            f"derived infer_experiment_group='{infer_experiment_group}')"
        )

    # ------------------------------------------------------------------
    # 2c) If _learn_config is present, inherit fields and auto-construct
    #     the update-training checkpoint path.
    # ------------------------------------------------------------------
    learn_experiment_group: str | None = None

    if _LEARN_CONFIG_KEY in base_cfg:
        learn_ref = str(base_cfg.pop(_LEARN_CONFIG_KEY))
        learn_config_lr = base_cfg.pop(_LEARN_CONFIG_LR_KEY, None)
        learn_checkpoint_epoch = base_cfg.pop(_LEARN_CHECKPOINT_EPOCH_KEY, None)

        if learn_config_lr is None:
            sys.exit(
                f"Error: {_LEARN_CONFIG_KEY} requires {_LEARN_CONFIG_LR_KEY} to be set."
            )
        if learn_checkpoint_epoch is None:
            sys.exit(
                f"Error: {_LEARN_CONFIG_KEY} requires {_LEARN_CHECKPOINT_EPOCH_KEY} to be set."
            )

        learn_config_lr = float(learn_config_lr)
        learn_checkpoint_epoch = int(learn_checkpoint_epoch)

        ref_path = learn_ref.lstrip("/")
        learn_config_path = CONFIG_DIR / f"{ref_path}.yaml"
        if not learn_config_path.exists():
            sys.exit(
                f"Error: {_LEARN_CONFIG_KEY} reference '{learn_ref}' resolved to "
                f"'{learn_config_path}' but the file does not exist."
            )

        resolved_learn = _resolve_config_with_defaults(learn_config_path, CONFIG_DIR)

        learn_experiment_group = OmegaConf.select(
            resolved_learn, "output.experiment_group", default=None,
        )
        if not learn_experiment_group:
            sys.exit(
                f"Error: {_LEARN_CONFIG_KEY} config has no output.experiment_group"
            )

        for dotted_key in _TRAIN_INHERIT_KEYS:
            val = OmegaConf.select(resolved_learn, dotted_key, default=None)
            if val is not None:
                OmegaConf.update(base_cfg, dotted_key, deepcopy(val), merge=False)

        learn_checkpoint_path = (
            f"outputs/{learn_experiment_group}"
            f"/learning_rate{learn_config_lr}"
            f"/model_after_epoch_{learn_checkpoint_epoch}"
        )
        OmegaConf.update(
            base_cfg, "train.upd.checkpoint_to_load_for_update_training",
            learn_checkpoint_path, merge=False,
        )

        OmegaConf.update(
            base_cfg, "train.num_train_epochs", learn_checkpoint_epoch, merge=False,
        )
        OmegaConf.update(
            base_cfg, "train.learning_rate", learn_config_lr, merge=False,
        )

        print(
            f"[generate_sweep_configs] Inherited fields from {_LEARN_CONFIG_KEY} "
            f"(experiment_group='{learn_experiment_group}', "
            f"lr={learn_config_lr}, epoch={learn_checkpoint_epoch}, "
            f"checkpoint='{learn_checkpoint_path}')"
        )

    if not grid:
        if train_experiment_group is None and learn_experiment_group is None:
            sys.exit(
                "Error: No sweep grid found. Either add hydra.sweeper.params "
                "to the sweep file, use _train_config to reference a train config, "
                "or use _learn_config to reference a learn config."
            )

    # ------------------------------------------------------------------
    # 3) Generate one config per grid point
    # ------------------------------------------------------------------
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    num_configs = 0
    for combo in product(*grid.values()):
        num_configs += 1

        cfg = deepcopy(base_cfg)
        overrides = dict(zip(keys, combo))

        print(overrides)
        apply_overrides(cfg, overrides)
        _apply_phase_checkpoint_save_overrides(cfg)

        unique_id = _build_unique_id(overrides, output_dir) or "default"
        cfg.output.experiment_name = unique_id

        # Auto-construct the checkpoint path from the train experiment group
        if train_experiment_group is not None:
            checkpoint_path = f"outputs/{train_experiment_group}/{unique_id}"
            OmegaConf.update(
                cfg, "infer.checkpoint_params.checkpoint_path", checkpoint_path,
            )

            grading_results_dir = f"outputs/{infer_experiment_group}/{unique_id}"
            OmegaConf.update(
                cfg, "grading.results_dir", grading_results_dir,
            )

        # Strip ``defaults`` and ``_train_config`` – must not appear in the
        # final config
        if "defaults" in cfg:
            del cfg["defaults"]
        if _TRAIN_CONFIG_KEY in cfg:
            del cfg[_TRAIN_CONFIG_KEY]
        for _key in (_LEARN_CONFIG_KEY, _LEARN_CONFIG_LR_KEY, _LEARN_CHECKPOINT_EPOCH_KEY, _RUN_COMMAND_KEY):
            if _key in cfg:
                del cfg[_key]

        # Warn about remaining sentinel values
        sentinels = _find_sentinel_values(cfg)
        if sentinels:
            print(
                f"[WARNING] Config '{unique_id}' still contains "
                f"{len(sentinels)} '{SENTINEL_VALUE}' sentinel(s): "
                + ", ".join(sentinels),
                file=sys.stderr,
            )

        OmegaConf.save(config=cfg, f=str(output_dir / f"{unique_id}.yaml"))

    msg = (
        f"[generate_sweep_configs] Wrote {num_configs} config file{'s' if num_configs != 1 else ''} "
        f"to {output_dir.resolve()}"
    )
    print(msg)
