from warnings import warn
import pytorch_lightning as pl
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    deterministic = True if seed else False
    if deterministic:
        pl.seed_everything(seed=seed, workers=True)

    return deterministic