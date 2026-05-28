# Set the model config overrides here.
from omegaconf import DictConfig


def set_up_model_config(cfg: DictConfig) -> DictConfig:
    
    # NEW CODE (The Fix):
    # Only set default if the user didn't specify one
    if cfg.model.attn_implementation == "overwriteme" or cfg.model.attn_implementation is None:
         cfg.model.attn_implementation = "eager"
    if cfg.model.max_batch_size.inference == "overwriteme" or cfg.model.max_batch_size.inference is None:
        cfg.model.max_batch_size.inference = 160
    
    if cfg.model.name in ["google/gemma-3-1b-it", "google/gemma-3-1b-pt"]:
        cfg.model.use_infer_cache = True
        cfg.model.max_batch_size.sft = 16
        cfg.model.max_batch_size.lora = 32
    else:
        # Defaults for all other models
        cfg.model.use_infer_cache = True
        cfg.model.max_batch_size.sft = 3
        cfg.model.max_batch_size.lora = 8
