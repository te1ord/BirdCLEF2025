import math
import numpy as np
import torch
import torch.nn as nn
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import FrequencyMasking, TimeMasking

from typing import Optional


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6, normalize_standart=True, normalize_minmax=True):
        super().__init__()
        self.eps = eps
        self.normalize_standart = normalize_standart
        self.normalize_minmax = normalize_minmax

    def forward(self, X):
        if self.normalize_standart:
            mean = X.mean((-2, -1), keepdim=True)
            std = X.std((-2, -1), keepdim=True)
            X = (X - mean) / (std + self.eps)
        if self.normalize_minmax:
            norm_max = torch.amax(X, dim=(-2, -1), keepdim=True)
            norm_min = torch.amin(X, dim=(-2, -1), keepdim=True)
            X = (X - norm_min) / (norm_max - norm_min + self.eps)
        return X 


class CustomMasking(nn.Module):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True):
        super().__init__()
        assert isinstance(mask_max_masks, int) and mask_max_masks > 0
        self.mask_max_masks = mask_max_masks
        self.mask_max_length = mask_max_length
        self.mask_module = None
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            output = x.clone()
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                n_applies = np.random.randint(low=1, high=self.mask_max_masks + 1)
                for _ in range(n_applies):
                    if self.inplace:
                        x[i : i + 1] = self.mask_module(x[i : i + 1])
                    else:
                        output[i : i + 1] = self.mask_module(output[i : i + 1])
        if self.inplace:
            return x
        else:
            return output


class CustomTimeMasking(CustomMasking):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True):
        super().__init__(mask_max_length=mask_max_length, mask_max_masks=mask_max_masks, p=p, inplace=inplace)
        self.mask_module = TimeMasking(time_mask_param=mask_max_length)


class CustomFreqMasking(CustomMasking):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True):
        super().__init__(mask_max_length=mask_max_length, mask_max_masks=mask_max_masks, p=p, inplace=inplace)
        self.mask_module = FrequencyMasking(freq_mask_param=mask_max_length)


class ChannelAgnosticAmplitudeToDB(nn.Module):
    def __init__(self, stype: str = "power", top_db: Optional[float] = None):
        super().__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() in [3, 4], f"Expected 3D or 4D tensor, but got {x.dim()}D tensor"

        add_fake_channel = False
        if x.dim() == 3:
            x = x.unsqueeze(1)
            add_fake_channel = True

        x_db = amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)

        if add_fake_channel:
            x_db = x_db.squeeze(1)
        return x_db