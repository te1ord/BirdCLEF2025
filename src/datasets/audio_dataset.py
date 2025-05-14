import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

import librosa
from audiomentations import Compose

from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import ast



class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_df: pd.DataFrame,
        filenpath_col: str,
        target_col: str,
        secondary_target_col: str,
        beta: float,
        class_names: List[str],
        sample_rate: int,
        target_duration: float,
        normalize_audio: bool,
        mixup_audio: bool,
        is_train: bool,
        mixup_params: Dict,
        audio_transforms: Optional[Compose],
        use_cache: bool,
        wave_piece: str,
        cache_n_samples: Optional[int] = 0,
    ):
        self.df = input_df.reset_index(drop=True)
        self.filenpath_col = filenpath_col
        self.target_col = target_col
        self.secondary_target_col = secondary_target_col
        
        self.df[secondary_target_col] = self.df[secondary_target_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
        )

        self.target_encoder = MultiLabelBinarizer(classes=class_names)
        self.target_encoder.fit([class_names])

        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_sample_count = int(sample_rate * target_duration)

        self.audio_transforms = audio_transforms
        self.normalize_audio = normalize_audio
        self.mixup_audio = mixup_audio and is_train

        if self.mixup_audio:
            assert mixup_params is not None, "If mixup_audio is True, mixup_params must not be None."
        
        self.mixup_params = mixup_params
        self.beta = beta

        self.is_train = is_train

        self.use_cache = use_cache
        if self.use_cache:
            self._cache_samples(top_n=cache_n_samples)

        self.wave_piece = wave_piece
        assert self.wave_piece in ('center','random')


    def __len__(self):
        return len(self.df)

    def _cache_samples(self, top_n: int):
        def load_wave(args):
            idx, cache = args
            return self._get_wave(idx) if cache else None

        # Cache only the most important samples
        idx_to_cache = self.df['weight'].sort_values(ascending=False).head(top_n).index
        should_cache = self.df.index.isin(idx_to_cache)
        args = list(zip(self.df.index, should_cache))

        self.df['wave'] = None
        with ThreadPoolExecutor() as executor:
            self.df['wave'] = list(tqdm(
                executor.map(load_wave, args),
                total=len(self.df),
                desc="Caching audio samples"
            ))


    def _get_mixup_idx(self):
        # TODO: add weighted sampling

        mixup_idx = np.random.randint(0, self.__len__())

        return mixup_idx
    

    def _get_labels(self, idx: int):
        prim = self.df.at[idx, self.target_col]
        secs = self.df.at[idx, self.secondary_target_col] or []
        return prim, list(secs)


    def _prepare_target(self, idx: int, sec_idx: Optional[int] = None):
        prim1, sec1 = self._get_labels(idx)
        alpha = self.mixup_params.get("alpha", 0.5)
        classes = list(self.target_encoder.classes_)
        K = len(classes)
        vec = np.zeros(K, dtype=float)

        # --- no mixup ---
        if sec_idx is None:
            i1 = classes.index(prim1)
            if self.mixup_params.get("hard_target", True):
                # hard: just multi-hot union of prim+secs
                vec[i1] = 1.0
                for s in sec1:
                    vec[classes.index(s)] = 1.0
            else:
                # soft:
                if sec1:
                    vec[i1] = 1.0 - self.beta
                    share = self.beta / len(sec1)
                    for s in sec1:
                        vec[classes.index(s)] = share
                else:
                    # no secondaries → full mass on primary
                    vec[i1] = 1.0

            return (vec > 0).astype(int) if self.mixup_params["hard_target"] else vec

        # --- mixup of two samples ---
        prim2, sec2 = self._get_labels(sec_idx)
        union_secs = set(sec1) | set(sec2)

        if union_secs:
            # there are secondaries → reserve β mass
            vec[classes.index(prim1)] = alpha * (1 - self.beta)
            vec[classes.index(prim2)] = (1 - alpha) * (1 - self.beta)
            share = self.beta / len(union_secs)
            for s in union_secs:
                vec[classes.index(s)] += share
        else:
            # no secondaries → β→0, exactly α & 1-α
            vec[classes.index(prim1)] = alpha
            vec[classes.index(prim2)] = 1 - alpha

        return (vec > 0).astype(int) if self.mixup_params.get("hard_target", True) else vec


    def _prepare_sample_piece(self, sample):
        # TODO: consider more fancy subsample selection

        sample_len = len(sample)

        # Extract center 5 seconds
        if sample_len < self.target_sample_count:
            # Pad if too short
            sample_piece = np.pad(sample, (0, self.target_sample_count - sample_len), mode='constant')
        
        else:

            if self.wave_piece == 'center':
                # Take center if too long
                start_idx = max(0, int(sample_len / 2 - self.target_sample_count / 2))
                end_idx = min(sample_len, start_idx + self.target_sample_count)
                sample_piece = sample[start_idx:end_idx]
            elif self.wave_piece == 'random':
                # pick a random start so that [start:start+target] is fully inside
                max_start = sample_len - self.target_sample_count
                start_idx = np.random.randint(0, max_start + 1)
                end_idx = start_idx + self.target_sample_count
                sample_piece = sample[start_idx:end_idx]
            else:
                raise ValueError(f"Invalid wave_piece: {self.wave_piece!r}")

        return sample_piece
    

    def _get_wave(self, idx: int):
        # Load audio
        # TODO: consider torchaudio.load instead of librosa
        filepath = self.df[self.filenpath_col].iloc[idx]
        wave, sr = librosa.load(filepath, sr=None)
        
        # wave = wave.astype(np.float32)

        # Ensure correct sample rate
        assert sr == self.sample_rate, f"Expected sample rate {self.sample_rate}, got {sr}"

        # We know that all samples contain only one channel
        assert len(wave.shape) == 1, f"Expected one channel audio, got wave.shape={wave.shape}."

        return wave
    

    def _get_sample(self, idx: int):
        if self.use_cache:
            wave = self.df['wave'].iloc[idx]
            if wave is None:
                wave = self._get_wave(idx)
        else:
            wave = self._get_wave(idx)

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

        wave = wave.astype(np.float32) # convert to float32 for audiomentations

        # Apply audio augmentations if in is_train mode
        if self.audio_transforms is not None and self.is_train:
            wave = self.audio_transforms(samples=wave, sample_rate=self.sample_rate)

        # Normalize audio
        if self.normalize_audio:
            wave = librosa.util.normalize(wave)

        # TODO: make float target for soft labels
        return torch.from_numpy(wave).float(), torch.from_numpy(target).float()


    def __getitem__(self, idx: int):
        return self._prepare_sample(idx)