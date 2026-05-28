import os
from omegaconf import open_dict, DictConfig
from accelerate.logging import get_logger
logger = get_logger(__name__)


def detect_accelerate(cfg: DictConfig):
    """
    Detects if the script is launched via 'accelerate launch' by checking env vars.
    Safely adds the 'accelerate' key to the config if it's missing.
    """
    
    # 1. Detection Logic: Check for environment variables set by Accelerate/Torchrun
    # LOCAL_RANK is the most reliable indicator that torch.distributed is active.
    is_accelerate_active = (
        os.environ.get("ACCELERATE_TORCH_LAUNCH") == "true" or 
        int(os.environ.get("LOCAL_RANK", -1)) != -1 or
        os.environ.get("DISTRIBUTED_TYPE") == "DEEPSPEED"
    )

    # 2. Safe Update: Unlock the config struct to verify/add the key
    with open_dict(cfg):
        cfg.accelerate = is_accelerate_active

    if is_accelerate_active:
        logger.info(f"Accelerate launch detected. Setting cfg.accelerate = {cfg.accelerate}")
    else:
        logger.info("No Accelerate launch detected.")