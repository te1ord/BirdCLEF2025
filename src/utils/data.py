import os
import torch
import pandas as pd
from typing import Dict, Any, Optional, Callable
from torch.utils.data import DataLoader
from audiomentations import Compose
from sklearn.model_selection import KFold
from src.datasets.audio_dataset import AudioDataset


def prepare_data(
    df: pd.DataFrame,
    audio_dir: str,
    filenpath_col: str,
    target_col: str,
    sample_rate: int,
    duration: float,
    audio_transforms: Optional[Compose] = None,
    normalize_audio: bool = True,
    mixup_audio: bool = True,
    mixup_params: Optional[Dict] = {"prob": 0.5, "alpha": 1.0}
) -> AudioDataset:
    
    df['filepath'] = df[filenpath_col].apply(lambda path: os.path.join(audio_dir, path))

    return AudioDataset(
        input_df=df,
        audio_dir=audio_dir,
        filenpath_col="filepath",
        target_col=target_col,
        sample_rate=sample_rate,
        target_duration=duration,
        audio_transforms=audio_transforms,
        normalize_audio=normalize_audio,
        mixup_audio=mixup_audio,
        mixup_params=mixup_params
    )


def create_datasets(
    df: pd.DataFrame,
    val_fold: int,
    audio_dir: str,
    filenpath_col: str,
    target_col: str,
    sample_rate: int,
    duration: float,
    audio_transforms: Optional[Compose] = None,
    normalize_audio: bool = True,
    mixup_audio: bool = True,
    mixup_params: Optional[Dict] = {"prob": 0.5, "alpha": 1.0}
) -> tuple[AudioDataset, AudioDataset]:
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]

    train_dataset = prepare_data(
        df=train_df,
        audio_dir=audio_dir,
        filenpath_col=filenpath_col,
        target_col=target_col,
        sample_rate=sample_rate,
        target_duration=duration,
        audio_transforms=audio_transforms,
        normalize_audio=normalize_audio,
        mixup_audio=mixup_audio,
        mixup_params=mixup_params
    )

    val_dataset = prepare_data(
        df=val_df,
        audio_dir=audio_dir,
        filenpath_col=filenpath_col,
        target_col=target_col,
        sample_rate=sample_rate,
        target_duration=duration,
        audio_transforms=audio_transforms,
        normalize_audio=normalize_audio,
        mixup_audio=mixup_audio,
        mixup_params=mixup_params
    )

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: AudioDataset,
    val_dataset: AudioDataset,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader