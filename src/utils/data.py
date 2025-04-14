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
    target_duration: float,
    is_train: bool = True,
    audio_transforms: Optional[Compose] = None,
    normalize_audio: bool = True,
    mixup_audio: bool = True,
    mixup_params: Optional[Dict] = {"prob": 0.5, "alpha": 1.0},
) -> AudioDataset:
    
    df['filepath'] = df[filenpath_col].apply(lambda path: os.path.join(audio_dir, path))

    return AudioDataset(
        input_df=df,
        filenpath_col="filepath",
        target_col=target_col,
        sample_rate=sample_rate,
        target_duration=target_duration,
        audio_transforms=audio_transforms,
        normalize_audio=normalize_audio,
        mixup_audio=mixup_audio,
        mixup_params=mixup_params,
        training=is_train
    )


def create_datasets(
    df: pd.DataFrame,
    val_fold: int,
    audio_dir: str,
    **kwargs: Dict[str, Any]
) -> tuple[AudioDataset, AudioDataset]:
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]

    train_dataset = prepare_data(
        df=train_df,
        audio_dir=audio_dir,
        **kwargs.get("train_args", {})
    )

    val_dataset = prepare_data(
        df=val_df,
        audio_dir=audio_dir,
        **kwargs.get("val_args", {})
    )

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: AudioDataset,
    val_dataset: AudioDataset,
    **kwargs: Dict[str, Any]
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, **kwargs.get("train_args", {}))
    val_loader = DataLoader(val_dataset, **kwargs.get("val_args", {}))

    return train_loader, val_loader