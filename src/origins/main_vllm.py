"""
main_vllm.py

Runs multi-phase inference with vLLM over a set of generated prompts.
The script:
  1) Loads inference tasks and builds prompts.
  2) Builds one or more phase configs (often one checkpoint per phase).
  3) Runs vLLM generation per phase and saves outputs.
  4) Prints timing summaries for startup and each phase.

Key considerations:
- vLLM does not reliably re-initialize within the same Python process once
  CUDA has been used. To avoid hangs when running multiple checkpoints in one
  run, this script can isolate vLLM generation in a subprocess.
- Checkpoints may lack tokenizer/processor files (e.g., Gemma 3 requires
  preprocessor_config.json). We restore these from the base model if missing.
- Timing is tracked for model initialization, prompt preparation, inference,
  and post-processing to make performance profiling easy.
"""
import os
import traceback
import logging
from collections import defaultdict
import time
import shutil
from pathlib import Path
from dataclasses import dataclass
import multiprocessing as mp
import queue as queue_mod
from typing import Any, cast
import json

# third-party
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from safetensors import safe_open

# local
from origins.custom_classes import (
    InferenceTask,
    _prompt_category,
    PhaseConfig,
    Phase,
)
from origins.logging.print import AnsiColors, log_with_color
from origins.utils.seed import seed_everything
from origins.inference import load_inference_tasks
from origins.prompts.generate_prompts import generate_prompts_with_template
from origins.grading.tracker import ResponseTracker
from origins.model.overrides import set_up_model_config
from origins.model.loading import get_vllm_text_only_init_overrides


# Set before importing CUDA relevant libraries
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Setup standard python logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhaseTiming:
    """
    Stores timing breakdown for a single inference phase.
    """
    phase_name: str
    model_init_s: float = 0.0
    prompt_prep_s: float = 0.0
    inference_s: float = 0.0
    postprocess_s: float = 0.0

    @property
    def total_s(self) -> float:
        """Total wall-clock time for the phase."""
        return self.model_init_s + self.prompt_prep_s + self.inference_s + self.postprocess_s


def _get_parallel_sizes(model_path: str, num_gpus: int) -> tuple[int, int]:
    """
    Determines optimal tensor and pipeline parallel sizes based on the model's KV heads.
    vLLM requires: num_kv_heads % tensor_parallel_size == 0
    Total GPUs used = tensor_parallel_size * pipeline_parallel_size
    """
    import json

    try:
        config_file = Path(model_path) / "config.json"
        if not config_file.exists():
            return num_gpus, 1  # Fallback if we cannot read config

        with open(config_file) as f:
            config = json.load(f)

        # Get KV heads, fallback to regular attention heads, then fallback to num_gpus
        kv_heads = config.get("num_key_value_heads", config.get(
            "num_attention_heads", num_gpus))

        # (1) Check if the current number of GPUs works natively (pp_size = 1)
        if kv_heads % num_gpus == 0:
            return num_gpus, 1

        # (2) If not, find the largest tp_size that divides both GPUs and KV heads evenly
        for tp in range(num_gpus - 1, 0, -1):
            if num_gpus % tp == 0 and kv_heads % tp == 0:
                pp = num_gpus // tp                
                logger.info(
                    f"Auto-detected GQA constraint. Optimal change would be to TP={tp}, PP={pp} for {kv_heads} KV heads on {num_gpus} GPUs.")
                logger.info("Pipeline parallelism only supported on server** not locally...: wasting GPUs for TP fallback.")
                # pipeline parallelism fallback *only works on server** not locally...
                return tp, 1 # pp

    except Exception as e:
        logger.warning(f"Could not inspect model config for KV heads: {e}")    
    
    return num_gpus, 1


def _ensure_complete_checkpoint(checkpoint_path: str, base_model_name: str) -> None:
    """
    Ensure checkpoint directory is loadable by vLLM.

    Two things can go wrong with DeepSpeed-saved checkpoints:
    1. Missing tokenizer/processor files (copied from base model).
    2. tie_word_embeddings=False in config.json but no lm_head.weight on disk
       (DeepSpeed save_16bit_model drops it when weights are tied in memory).
       We patch the config to tie_word_embeddings=True so vLLM tie-loads lm_head
       from embed_tokens — this matches what's actually in memory post-training.
    """
    logging.warning(f"Ensuring checkpoint at {checkpoint_path} is complete for vLLM...")
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        return

    # --- (1) Restore missing tokenizer/processor files from the base model ---
    required_files = [
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for fname in required_files:
        dest = ckpt_path / fname
        if not dest.exists():
            log_with_color(
                f"Missing {fname} in checkpoint. Fetching from {base_model_name}...",
                logger, AnsiColors.OKCYAN,
            )
            try:
                cached = hf_hub_download(
                    repo_id=base_model_name, filename=fname)
                shutil.copy(cached, dest)
                log_with_color(
                    f"Restored {fname} -> {dest}", logger, AnsiColors.OKGREEN)
            except Exception as exc:
                logger.warning(f"Could not fetch {fname}: {exc}")

    # --- (2) Reconcile tie_word_embeddings with what's actually saved ---
    cfg_path = ckpt_path / "config.json"
    if not cfg_path.exists():
        return

    with open(cfg_path) as f:
        config = json.load(f)

    # If config already says tied, nothing to do.
    if config.get("tie_word_embeddings", False):
        return

    # Scan all weight files for lm_head.weight.
    has_lm_head = False

    # Gemma family always ties embeddings architecturally — skip weight scan entirely.
    # This avoids the slow torch.load path for .bin checkpoints without index files.
    arch = config.get("architectures", [""])[0].lower()
    if any(a in arch for a in ("gemma", "gemma2", "gemma3", "gemma4")):
        logger.info(f"Detected Gemma architecture ({arch}); assuming tied embeddings.")
        if not config.get("tie_word_embeddings", False):
            config["tie_word_embeddings"] = True
            with open(cfg_path, "w") as f:
                json.dump(config, f, indent=2)
        return

    has_lm_head = False

    # 1. Fast path: safetensors index file (O(1) JSON read)
    st_index = ckpt_path / "model.safetensors.index.json"
    bin_index = ckpt_path / "pytorch_model.bin.index.json"
    single_st = ckpt_path / "model.safetensors"
    single_bin = ckpt_path / "pytorch_model.bin"

    if st_index.exists():
        with open(st_index) as f:
            weight_map = json.load(f).get("weight_map", {})
        has_lm_head = any(k.endswith("lm_head.weight") for k in weight_map)

    # 2. Fast path: pytorch bin index file (O(1) JSON read)
    elif bin_index.exists():
        with open(bin_index) as f:
            weight_map = json.load(f).get("weight_map", {})
        has_lm_head = any(k.endswith("lm_head.weight") for k in weight_map)

    # 3. Medium path: single safetensors file (reads header only, not tensors)
    elif single_st.exists():
        try:
            with safe_open(single_st, framework="pt") as sf:
                has_lm_head = any(k.endswith("lm_head.weight") for k in sf.keys())
        except Exception as exc:
            logger.warning(f"Could not inspect {single_st}: {exc}")

    # 4. Slow path: glob all safetensors shards (header read per shard)
    elif any(ckpt_path.glob("*.safetensors")):
        logger.warning(
            "No safetensors index found, globbing all .safetensors files to check "
            "for lm_head.weight (this may be slow)..."
        )
        for st_file in ckpt_path.glob("*.safetensors"):
            try:
                with safe_open(st_file, framework="pt") as sf:
                    if any(k.endswith("lm_head.weight") for k in sf.keys()):
                        has_lm_head = True
                        break
            except Exception as exc:
                logger.warning(f"Could not inspect {st_file}: {exc}")

    # 5. Slowest path: single pytorch bin (must walk pickle stream)
    elif single_bin.exists():
        logger.warning(
            f"Only pytorch_model.bin found in {ckpt_path} without an index file. "
            "Walking pickle stream to find lm_head.weight — this may be slow."
        )
        state = torch.load(single_bin, map_location="meta", weights_only=True)
        has_lm_head = any(k.endswith("lm_head.weight") for k in state.keys())

    else:
        logger.warning(
            f"No weight files found in {ckpt_path}; cannot determine lm_head status."
        )

    # pytorch_model.bin (DeepSpeed save_16bit_model output)
    if not has_lm_head:
        bin_index = ckpt_path / "pytorch_model.bin.index.json"
        bin_single = ckpt_path / "pytorch_model.bin"
        if bin_index.exists():
            with open(bin_index) as f:
                weight_map = json.load(f).get("weight_map", {})
            has_lm_head = any(k.endswith("lm_head.weight") for k in weight_map)
        elif bin_single.exists():
            # Load keys only, not tensors, to avoid memory cost.
            state = torch.load(
                bin_single, map_location="meta", weights_only=True)
            has_lm_head = any(k.endswith("lm_head.weight")
                              for k in state.keys())

    if has_lm_head:
        return

    # Config says untied, but no lm_head on disk => weights are actually tied.
    log_with_color(
        f"Checkpoint has tie_word_embeddings=False but no lm_head.weight on disk. "
        f"Patching config to tie_word_embeddings=True for vLLM compatibility.",
        logger, AnsiColors.WARNING,
    )
    config["tie_word_embeddings"] = True
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)


def _get_vllm_sampling_params(cfg: DictConfig) -> SamplingParams:
    """
    Convert Hydra config into vLLM SamplingParams.

    Args:
        cfg: Hydra config containing inference settings.

    Returns:
        SamplingParams: vLLM sampling parameters object.
    """
    return SamplingParams(
        n=cfg.infer.num_return_sequences,
        temperature=cfg.infer.temperature,
        top_p=cfg.infer.top_p if hasattr(cfg.infer, "top_p") else 1.0,
        max_tokens=cfg.infer.max_new_tokens,
    )


def _format_chat_prompts(
    tokenizer,
    prompts: list[str],
    enable_thinking: bool = False,
) -> list[str]:
    messages = [[{"role": "user", "content": p}] for p in prompts]
    rendered = tokenizer.apply_chat_template(
        cast(list, messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return cast(list[str], rendered)


def _vllm_worker_entry(
    cfg_container: dict[str, Any],
    model_path: str,
    prompt_texts: list[str],
    q: mp.Queue,
) -> None:
    """
    Entry point for subprocess-based vLLM inference.

    Runs LLM init + generate in a child process to avoid CUDA re-init issues.
    Results + timings are returned through a multiprocessing queue.

    Args:
        cfg_container: Plain dict config (pickle-safe).
        model_path: Checkpoint path to load.
        prompt_texts: List of prompt strings to generate from.
        q: Multiprocessing queue for returning results and timings.

    Returns:
        None
    """
    try:
        cfg = OmegaConf.create(cfg_container)
        num_gpus = torch.cuda.device_count()
        # Check if number GPUs works with the model's KV heads,
        # and adjust pipeline parallelization accordingly
        tp_size, pp_size = _get_parallel_sizes(model_path, num_gpus)

        # Per-model overrides for text-only inference. Empty dict for every
        # model except Llama-3.2-11B-Vision; see `get_vllm_text_only_init_overrides`.
        extra_llm_kwargs = get_vllm_text_only_init_overrides(model_path)

        t0 = time.perf_counter()
        llm = LLM(
            model=model_path,
            tokenizer=cfg.model.name,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            dtype="bfloat16",
            gpu_memory_utilization=0.70,
            max_model_len=cfg.infer.max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            **extra_llm_kwargs,
        )
        model_init_s = time.perf_counter() - t0

        sampling_params = _get_vllm_sampling_params(cfg)

        t0 = time.perf_counter()
        tokenizer = llm.get_tokenizer()
        formatted_prompts = _format_chat_prompts(
            tokenizer, prompt_texts, enable_thinking=False
        )
        request_outputs = llm.generate(
            formatted_prompts, sampling_params, use_tqdm=True)

        inference_s = time.perf_counter() - t0

        results = [[o.text for o in out.outputs] for out in request_outputs]
        q.put({"results": results, "model_init_s": model_init_s,
              "inference_s": inference_s})
    except Exception as exc:
        # Print the full traceback so we can see exactly where the assert failed
        error_msg = traceback.format_exc()
        logger.error(f"Worker process failed with error:\n{error_msg}")
        q.put(exc)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _vllm_generate_in_subprocess(
    cfg_container: dict[str, Any],
    model_path: str,
    prompt_texts: list[str],
) -> tuple[list[list[str]], float, float]:
    """
    Run vLLM in a subprocess and collect results.

    Args:
        cfg_container: Plain dict config (pickle-safe).
        model_path: Checkpoint path to load.
        prompt_texts: List of prompt strings to generate from.

    Returns:
        tuple:
            - results: List of lists of generated strings (per prompt).
            - model_init_s: Time to initialize the model.
            - inference_s: Time to generate outputs. 
    """
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()

    p = ctx.Process(
        target=_vllm_worker_entry,
        args=(cfg_container, model_path, prompt_texts, q),
    )
    p.start()

    timeout_s = getattr(cfg_container.get(
        "infer", {}), "worker_timeout_s", 7200)
    try:
        payload = q.get(timeout=timeout_s)
    except queue_mod.Empty:
        p.terminate()
        raise RuntimeError("vLLM worker timed out")

    p.join()
    if isinstance(payload, Exception):
        raise payload

    return payload["results"], payload["model_init_s"], payload["inference_s"]


def run_vllm_inference(
    cfg: DictConfig,
    prompts: list[_prompt_category],
    phase_cfg: PhaseConfig,
    inference_tasks: list[InferenceTask],
    isolate_vllm: bool = False,
):
    """
    Run one inference phase and return timing details.

    Args:
        cfg: Hydra config.
        prompts: Prompt objects to generate from.
        phase_cfg: Phase configuration (checkpoint + epoch).
        inference_tasks: List of inference tasks for tracking.
        isolate_vllm: If True, run vLLM in a subprocess to avoid CUDA re-init issues.

    Returns:
        PhaseTiming: Timing summary for this phase.
    """
    if not prompts:
        logger.warning("No prompts provided for inference.")
        return PhaseTiming(phase_name=str(phase_cfg.starting_epoch or phase_cfg.phase))

    phase_timing = PhaseTiming(phase_name=str(
        phase_cfg.starting_epoch or phase_cfg.phase))
    tracker = ResponseTracker(cfg, prompts=prompts,
                              inference_tasks=inference_tasks)

    model_path = cfg.model.name
    if phase_cfg.start_from_checkpoint:
        model_path = phase_cfg.start_from_checkpoint
        _ensure_complete_checkpoint(model_path, cfg.model.name)
        log_with_color(
            f"Loading from checkpoint: {model_path}", logger, AnsiColors.OKBLUE)

    # Prepare prompts
    t0 = time.perf_counter()
    prompt_texts = [p.prompt_text for p in prompts]
    phase_timing.prompt_prep_s = time.perf_counter() - t0

    log_with_color(
        f"Generating responses for {len(prompt_texts)} prompts...", logger, AnsiColors.OKMAGENTA)

    if isolate_vllm:
        results, model_init_s, inference_s = _vllm_generate_in_subprocess(
            OmegaConf.to_container(cfg, resolve=True),
            model_path,
            prompt_texts,
        )
        phase_timing.model_init_s = model_init_s
        phase_timing.inference_s = inference_s
    else:
        num_gpus = torch.cuda.device_count()
        # Check if number GPUs works with the model's KV heads, and adjust
        # pipeline parallelization accordingly
        tp_size, pp_size = _get_parallel_sizes(model_path, num_gpus)

        # Per-model overrides for text-only inference. Empty dict for every
        # model except Llama-3.2-11B-Vision; see `get_vllm_text_only_init_overrides`.
        extra_llm_kwargs = get_vllm_text_only_init_overrides(model_path)

        t0 = time.perf_counter()
        llm = LLM(
            model=model_path,
            tokenizer=cfg.model.name,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            dtype="bfloat16",
            gpu_memory_utilization=0.70,
            max_model_len=cfg.infer.max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            **extra_llm_kwargs,
        )
        phase_timing.model_init_s = time.perf_counter() - t0

        sampling_params = _get_vllm_sampling_params(cfg)
        t0 = time.perf_counter()

        tokenizer = llm.get_tokenizer()
        formatted_prompts = _format_chat_prompts(
            tokenizer, prompt_texts, enable_thinking=False
        )
        request_outputs = llm.generate(
            formatted_prompts, sampling_params, use_tqdm=True)

        phase_timing.inference_s = time.perf_counter() - t0
        results = [[o.text for o in out.outputs] for out in request_outputs]

        if hasattr(llm, "shutdown"):
            llm.shutdown()
        del llm

    # Start timing post-processing
    t0 = time.perf_counter()

    # Map results to prompt objects (ResponseTracker expects this structure)
    results_map = defaultdict(list)
    for i, generated_texts in enumerate(results):
        results_map[prompts[i]] = generated_texts

    iteration = phase_cfg.starting_epoch or 0
    tracker.register_responses_only(
        responses=results_map, iteration=iteration)

    tracker.save_to_file(epoch_subdir=phase_cfg.starting_epoch)
    phase_timing.postprocess_s = time.perf_counter() - t0
    return phase_timing


def _construct_inference_phase_configs(cfg: DictConfig) -> list[PhaseConfig]:
    """
    Build phase configs for inference-only runs over checkpoint ranges.

    Args:
        cfg: Hydra config with checkpoint range settings.

    Returns:
        list[PhaseConfig]: Phase configs to run in order.
    """
    # Check if only doing inference on checkpoints.
    if not (cfg.infer.enable_inference and not cfg.train.enable_training):
        raise ValueError(
            "construct_phase_configs is only for inference-only scenarios."
        )

    if cfg.infer.checkpoint_params.checkpoint_path is None:
        raise ValueError(
            "When doing inference only, please use "
            "train.validation.checkpoint_params to specify the checkpoint path."
        )

    # If only doing inference on checkpoints, we need to load a different
    # checkpoint for each epoch in the specified range.
    base_checkpoint_path = cfg.infer.checkpoint_params.checkpoint_path
    start_epoch = cfg.infer.checkpoint_params.start_epoch
    end_epoch = cfg.infer.checkpoint_params.end_epoch
    infer_frequency = cfg.infer.frequency

    if start_epoch is None or end_epoch is None or infer_frequency < 1:
        raise ValueError(
            "Both start_epoch and end_epoch must be specified when doing "
            "inference only with checkpoint paths, "
            "and infer.frequency must be at least 1."
        )
    phases = []
    for epoch in range(start_epoch, end_epoch + 1, infer_frequency):
        checkpoint_path = str(
            Path(base_checkpoint_path) / f"model_after_epoch_{epoch}")
        if epoch == 0:
            if Path(checkpoint_path).exists():
                logger.info(
                    f"[EPOCH 0] Loading local checkpoint path {checkpoint_path}"
                )
            else:
                logger.warning(
                    f"[EPOCH 0] No local checkpoint found at "
                    f"{checkpoint_path}; loading base model {cfg.model.name}")
                checkpoint_path = cfg.model.name
        phase_cfg = PhaseConfig(
            phase=Phase.INFER_ONLY,
            starting_epoch=epoch,
            start_from_checkpoint=checkpoint_path,
            num_epochs=0,
            save_model_at_every_epoch=False,
            evaluate_on_train_begin=True,
            save_model_at_the_end=False,
        )
        phases.append(phase_cfg)
    return phases


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Orchestrates inference:
      1) Load tasks and prompts
      2) Build phase configs
      3) Run inference per phase
      4) Print a timing summary

    Args:
        cfg: Hydra config.

    Returns:
        None
    """
    set_up_model_config(cfg)
    seed_everything(cfg)

    startup_t0 = time.perf_counter()

    inference_tasks = load_inference_tasks(cfg=cfg)
    if not inference_tasks:
        logger.error("No inference tasks loaded.")
        return

    prompts = generate_prompts_with_template(
        inference_tasks=inference_tasks,
        include_reasoning=cfg.infer.include_reasoning,
        allow_unsure=cfg.infer.allow_unsure,
        base_model_prompt=cfg.infer.base_model_prompt,
        cfg=cfg,
    )

    phase_configs = _construct_inference_phase_configs(cfg)
    startup_s = time.perf_counter() - startup_t0

    isolate_vllm = len(phase_configs) > 1
    phase_timings: list[PhaseTiming] = []

    for phase_cfg in phase_configs:
        phase_timings.append(
            run_vllm_inference(
                cfg=cfg,
                prompts=prompts,
                phase_cfg=phase_cfg,
                inference_tasks=inference_tasks,
                isolate_vllm=isolate_vllm,
            )
        )

    logger.info("=== Timing Summary ===")
    logger.info(f"Startup: {startup_s:.2f}s")
    for pt in phase_timings:
        logger.info(
            f"Phase {pt.phase_name}: total {pt.total_s:.2f}s | "
            f"model_init {pt.model_init_s:.2f}s | "
            f"prompt_prep {pt.prompt_prep_s:.2f}s | "
            f"inference {pt.inference_s:.2f}s | "
            f"postprocess {pt.postprocess_s:.2f}s"
        )


if __name__ == "__main__":
    # vLLM V1 engine (v0.8+) can be unstable or have high memory overhead during init.
    # We force the legacy (V0) engine for stability on multi-GPU setups.
    os.environ["VLLM_USE_V1"] = "0"

    # If users still face P2P issues (hanging/crashing during NCCL init), uncomment:
    # os.environ["NCCL_P2P_DISABLE"] = "1"

    # Ensure standard PyTorch doesn't try to hog all memory before vLLM starts
    # vLLM manages memory aggressively.
    os.environ["TORCH_Empty_Cache"] = "1"

    _MAIN_START_TIME = time.perf_counter()
    try:
        main()
    finally:
        # Best-effort cleanup to silence NCCL/shm warnings
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    elapsed_s = time.perf_counter() - _MAIN_START_TIME
    elapsed_h = int(elapsed_s // 3600)
    elapsed_m = int((elapsed_s % 3600) // 60)
    elapsed_rem_s = elapsed_s % 60
    logger.info(
        f"\n\n{'='*40}\nTOTAL wall-clock runtime: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_rem_s:05.2f}\n{'='*40}"
    )
