import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import random
import numpy as np

from utils.data import train

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Train model
    best_model_path = train(cfg)

    print(f"Training completed. Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main() 