import os
import random
import numpy as np
import torch
from omegaconf import DictConfig


def generate_random_seed():
    """Generate a random seed."""
    return random.randint(0, 2**32 - 1)


# Update this function whenever you have a library that needs to be seeded.
def seed_everything(cfg: DictConfig):
    """Seed all random generators."""

    seed = cfg.seed_for_all

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    #  A lighter version of the above otherwise as not all algorithms have a deterministic implementation
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

