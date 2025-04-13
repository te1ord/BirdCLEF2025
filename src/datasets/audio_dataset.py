import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import librosa
from audiomentations import Compose

from typing import Dict, Optional



class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_df: pd.DataFrame,
        filenpath_col: str = "filepath",
        target_col: str = "target",
        sample_rate: int = 32000,
        target_duration: float = 5.,
        audio_transforms: Optional[Compose] = None,
        normalize_audio: bool = True,
        mixup_audio: bool = True,
        mixup_params: Optional[Dict] = {"prob": 0.5, "alpha": 1.0}
    ):
        self.df = input_df.reset_index(drop=True)

        self.filenpath_col = filenpath_col
        self.target_col = target_col
        self.target_encoder = self._get_target_encoder()

        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_sample_count = int(sample_rate * target_duration)

        self.audio_transforms = audio_transforms
        self.normalize_audio = normalize_audio
        self.mixup_audio = mixup_audio

        if self.mixup_audio:
            assert mixup_params is not None, "If mixup_audio is True, mixup_params must not be None."
        self.mixup_params = mixup_params

        # TODO: all dataset caching


    def __len__(self):
        return len(self.df)
    

    def _get_target_encoder(self):
        target_encoder = OneHotEncoder()
        target_encoder.fit(
            X=self.df[self.target_col].values.reshape(-1, 1)
            )
        
        return target_encoder


    def _get_mixup_idx(self):
        # TODO: add weighted sampling

        mixup_idx = np.random.randint(0, self.__len__())

        return mixup_idx
    

    def _prepare_target(self, idx: int, sec_idx: Optional[int] = None):
        if not sec_idx:
            idxs = [idx]
            weights = np.array([1])

        else:
            idxs = [idx, sec_idx]

            if self.mixup_params.get("hard_target", True):
                weights = np.array([1, 1])
            else:
                alpha = self.mixup_params['alpha']
                weights = np.array([alpha, 1-alpha])


        labels = self.df.loc[idxs, self.target_col].values.reshape(-1, 1)
        encoded_labels = self.target_encoder.transform(labels).toarray()

        soft_labels = weights @ encoded_labels

        return soft_labels


    def _prepare_sample_piece(self, sample):
        # TODO: consider more fancy subsample selection

        sample_len = len(sample)
        # Extract center 5 seconds
        if sample_len < self.target_sample_count:
            # Pad if too short
            sample_piece = np.pad(sample, (0, self.target_sample_count - sample_len), mode='constant')
        else:
            # Take center if too long
            start_idx = max(0, int(sample_len / 2 - self.target_sample_count / 2))
            end_idx = min(sample_len, start_idx + self.target_sample_count)
            sample_piece = sample[start_idx:end_idx]

        return sample_piece
    

    def _get_sample(self, idx: int):
        # Load audio
        # TODO: consider torchaudio.load instead of librosa
        filepath = self.df[self.filenpath_col].iloc[idx]
        wave, sr = librosa.load(filepath, sr=None)
        
        # Ensure correct sample rate
        assert sr == self.sample_rate, f"Expected sample rate {self.sample_rate}, got {sr}"

        # We know that all samples contain only one channel
        assert len(wave.shape) == 1, "Expected one channel audio."

        # Extract center 5 seconds
        wave = self._prepare_sample_piece(wave)

        return wave
    

    def _prepare_sample(self, idx: int):
        wave = self._get_sample(idx)

        if self.mixup_audio and np.random.binomial(
            n=1, p=self.mixup_params["prob"]
        ):
            alpha = self.mixup_params['alpha']
            weights = np.array([alpha, 1-alpha])

            sec_idx = self._get_mixup_idx()
            sec_wave = self._get_sample(sec_idx)

            wave = weights @ np.array([wave, sec_wave])
            target = self._prepare_target(idx, sec_idx)

        else:
            target = self._prepare_target(idx)

        # Apply audio augmentations if in training mode
        if self.audio_transforms is not None and self.training:
            wave = self.audio_transforms(samples=wave, sample_rate=self.sample_rate)

        # Normalize audio
        if self.normalize_audio:
            wave = librosa.util.normalize(wave)

        return torch.from_numpy(wave).float(), torch.from_numpy(target).float()


    def __getitem__(self, idx: int):
        return self._prepare_sample(idx)