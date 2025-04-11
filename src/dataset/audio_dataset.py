import os
import math
import torch
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, List
import torch.nn as nn 
from audiomentations import Compose, AddGaussianNoise, TimeStretch
from torchaudio.transforms import MelSpectrogram
from src.augmentations.audio_augmentations import ChannelAgnosticAmplitudeToDB, NormalizeMelSpec
from src.augmentations.spec_augmentations import CustomFreqMasking, CustomTimeMasking

class AddChannelDimension(nn.Module):
    """Adds a channel dimension to the input tensor."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0)

class AudioDataset(Dataset):
    def __init__(
        self,
        input_df: pd.DataFrame,
        filenpath_col: str = "filename",
        target_col: str = "primary_label",
        sample_rate: int = 32000,
        target_duration: float = 5.0,
        normalize_audio: bool = True,
        audio_transforms: Optional[Compose] = None,
        spec_augment_config: Optional[Dict[str, Any]] = None,
        spec_params: Optional[Dict[str, Any]] = None,
    ):
        self.df = input_df.reset_index(drop=True)
        self.filenpath_col = filenpath_col
        self.target_col = target_col
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_samples = int(sample_rate * target_duration)
        self.normalize_audio = normalize_audio
        self.audio_transforms = audio_transforms

        # Initialize spectrogram extractor
        self.spectogram_extractor = nn.Sequential(
            MelSpectrogram(**spec_params),
            AddChannelDimension(),  # Add channel dimension after mel spectrogram
            ChannelAgnosticAmplitudeToDB(top_db=80.0),
            NormalizeMelSpec(normalize_standart=True, normalize_minmax=True),
        )

        # Initialize spec augmentations if provided
        if spec_augment_config is not None:
            self.spec_augment = []
            if "freq_mask" in spec_augment_config:
                self.spec_augment.append(CustomFreqMasking(**spec_augment_config["freq_mask"]))
            if "time_mask" in spec_augment_config:
                self.spec_augment.append(CustomTimeMasking(**spec_augment_config["time_mask"]))
            self.spec_augment = nn.Sequential(*self.spec_augment)
        else:
            self.spec_augment = None

        # Create label mapping
        self.classes = sorted(self.df[target_col].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def _prepare_sample(self, idx: int):
        # Load audio
        filepath = self.df[self.filenpath_col].iloc[idx]
        au, sr = librosa.load(filepath, sr=self.sample_rate)
        
        # Ensure correct sample rate
        assert sr == self.sample_rate, f"Expected sample rate {self.sample_rate}, got {sr}"

        # Extract center 5 seconds
        if len(au) < self.target_samples:
            # Pad if too short
            au = np.pad(au, (0, self.target_samples - len(au)), mode='constant')
        else:
            # Take center if too long
            start_idx = max(0, int(len(au) / 2 - self.target_samples / 2))
            end_idx = min(len(au), start_idx + self.target_samples)
            au = au[start_idx:end_idx]

        # Apply audio augmentations if in training mode
        if self.audio_transforms is not None and self.training:
            au = self.audio_transforms(samples=au, sample_rate=sr)

        # Normalize audio
        if self.normalize_audio:
            au = librosa.util.normalize(au)

        # Convert to spectrogram
        au_tensor = torch.from_numpy(au).float()
        spec = self.spectogram_extractor(au_tensor)

        # Apply spec augmentations if in training mode
        if self.spec_augment is not None and self.training:
            spec = self.spec_augment(spec)

        # Get target
        target_idx = self.class_to_idx[self.df[self.target_col].iloc[idx]]

        return spec, target_idx

    def __getitem__(self, idx: int):
        return self._prepare_sample(idx) 