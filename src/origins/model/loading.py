"""
Model loading helpers that handle the Llama-3.2-11B-Vision text-only training
case.

Loading `meta-llama/Llama-3.2-11B-Vision[-Instruct]` via `AutoModelForCausalLM`
instantiates only the text submodule and saves a checkpoint with
`model_type: "mllama_text_model"`, which is not registered in `AutoConfig` and
breaks subsequent loads (Transformers, vLLM). To avoid that, we load and save
the full multimodal class (`MllamaForConditionalGeneration`) and freeze the
vision tower so training stays text-only.

Scope: only the two 11B-Vision repo ids are routed through this path. Every
other model — including other mllama variants like 90B-Vision — falls through
to `AutoModelForCausalLM.from_pretrained` unchanged.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from transformers import AutoModelForCausalLM, PreTrainedModel

logger = logging.getLogger(__name__)


# ONLY these HuggingFace repo ids are routed through the multimodal-class load path.
_LLAMA_3_2_11B_VISION_NAMES: frozenset[str] = frozenset({
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-11B-Vision",
})

_MLLAMA_FULL_MODEL_TYPE = "mllama"
_MLLAMA_FULL_ARCHITECTURE = "MllamaForConditionalGeneration"


def _read_local_checkpoint_metadata(path: str) -> tuple[str | None, list[str]]:
    """
    Read `model_type` and `architectures` from a local checkpoint's
    `config.json`. Returns `(None, [])` if `path` is not a local directory or
    has no readable config.
    """
    if not isinstance(path, str) or not os.path.isdir(path):
        return None, []
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return None, []
    try:
        with open(cfg_path) as f:
            data = json.load(f)
    except Exception as exc:
        logger.debug(f"Could not read {cfg_path}: {exc}")
        return None, []
    return data.get("model_type"), list(data.get("architectures") or [])


def _should_use_mllama_full_class(name_or_path: str) -> bool:
    """
    Decide whether to load `name_or_path` as `MllamaForConditionalGeneration`.

    Returns True only when:
    - `name_or_path` is exactly the Llama-3.2-11B-Vision(-Instruct) repo id, OR
    - `name_or_path` is a local directory whose `config.json` was saved by us
      previously as a full mllama checkpoint
      (`model_type == "mllama"` and architectures contains
      `MllamaForConditionalGeneration`).

    Every other input — including any other mllama variant on HF — returns
    False, preserving the original `AutoModelForCausalLM` path.
    """
    if name_or_path in _LLAMA_3_2_11B_VISION_NAMES:
        return True

    model_type, architectures = _read_local_checkpoint_metadata(name_or_path)
    return (
        model_type == _MLLAMA_FULL_MODEL_TYPE
        and _MLLAMA_FULL_ARCHITECTURE in architectures
    )


def load_pretrained_lm(
    name_or_path: str,
    **from_pretrained_kwargs: Any,
) -> PreTrainedModel:
    """
    Drop-in replacement for `AutoModelForCausalLM.from_pretrained`.

    For Llama-3.2-11B-Vision and our own full-mllama checkpoints, loads as
    `MllamaForConditionalGeneration`. For everything else, behaves exactly
    like `AutoModelForCausalLM.from_pretrained`.
    """
    if _should_use_mllama_full_class(name_or_path):
        # Imported lazily so non-target runs don't pay the import cost.
        from transformers import MllamaForConditionalGeneration

        logger.info(
            f"Loading {name_or_path} as MllamaForConditionalGeneration "
            f"so saved checkpoints remain AutoConfig/vLLM-loadable."
        )
        return MllamaForConditionalGeneration.from_pretrained(
            name_or_path, **from_pretrained_kwargs
        )

    return AutoModelForCausalLM.from_pretrained(
        name_or_path, **from_pretrained_kwargs
    )


# Top-level submodule prefixes that hold non-text weights on
# `MllamaForConditionalGeneration`.
_MLLAMA_NON_TEXT_PREFIXES: tuple[str, ...] = (
    "vision_model",
    "multi_modal_projector",
)


# vLLM treats mllama as encoder-decoder and pre-reserves cross-attention KV
# cache sized for image tokens × max_num_seqs even when no images are passed,
# which OOMs on 80GB GPUs. `limit_mm_per_prompt={"image": 0}` would be the
# targeted fix but hits a vLLM 0.8.4 bug (KeyError: 'num_tiles' in the mllama
# processor's empty-image path). Capping `max_num_seqs` bounds the reservation
# instead.
_MLLAMA_TEXT_ONLY_VLLM_OVERRIDES: dict[str, Any] = {
    "max_num_seqs": 16,
}


def get_vllm_text_only_init_overrides(name_or_path: str) -> dict[str, Any]:
    """
    Extra kwargs to merge into `vllm.LLM(...)` for the given model.

    Non-empty only for Llama-3.2-11B-Vision; empty dict for everything else,
    so non-target `LLM(...)` calls are unchanged.
    """
    if not _should_use_mllama_full_class(name_or_path):
        return {}
    return dict(_MLLAMA_TEXT_ONLY_VLLM_OVERRIDES)


def freeze_non_text_modules(model: PreTrainedModel) -> int:
    """
    Freeze vision tower and multimodal projector on a
    `MllamaForConditionalGeneration`. No-op for any other model class.
    """
    try:
        from transformers import MllamaForConditionalGeneration
    except ImportError:
        return 0

    if not isinstance(model, MllamaForConditionalGeneration):
        return 0

    frozen_params = 0
    frozen_names: list[str] = []
    for name, param in model.named_parameters():
        if any(
            name.startswith(p + ".") or name == p
            for p in _MLLAMA_NON_TEXT_PREFIXES
        ):
            if param.requires_grad:
                param.requires_grad = False
                frozen_names.append(name)
            frozen_params += param.numel()

    logger.info(
        f"Froze {len(frozen_names)} tensors ({frozen_params/1e6:,.1f} M params) "
        f"under prefixes {_MLLAMA_NON_TEXT_PREFIXES} on "
        f"{type(model).__name__}."
    )
    if frozen_names:
        sample = "\n  ".join(frozen_names[:5])
        logger.info(f"First frozen tensors:\n  {sample}")

    return frozen_params
