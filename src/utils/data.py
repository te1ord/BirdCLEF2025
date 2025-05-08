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
    **kwargs: Dict[str, Any]
) -> AudioDataset:
    
    df['filepath'] = df[filenpath_col].apply(lambda path: os.path.join(audio_dir, path))

    return AudioDataset(
        input_df=df,
        filenpath_col="filepath",
        **kwargs
    )


def create_datasets(
    df: pd.DataFrame,
    val_fold: int,
    audio_dir: str,
    sample_submission_path: str,
    **kwargs: Dict[str, Any]
) -> tuple[AudioDataset, AudioDataset]:

    # TODO: write it somehow different
    target_col = kwargs['train_args']['target_col']

    class_freq = df[target_col].value_counts(normalize=True)
    sampling_weights = class_freq.pow(-kwargs['sampling_temperature'])

    df['weight'] = df[target_col].apply(lambda x: sampling_weights[x])

    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]

    # Get class names
    ss_df = pd.read_csv(sample_submission_path)
    class_names = sorted(ss_df.columns[1:].tolist())

    train_dataset = prepare_data(
        df=train_df,
        audio_dir=audio_dir,
        **kwargs.get("train_args", {}),
        audio_transforms = kwargs['audio_transforms'],
        class_names=class_names,
    )

    val_dataset = prepare_data(
        df=val_df,
        audio_dir=audio_dir,
        **kwargs.get("val_args", {}),
        audio_transforms = None,
        class_names=class_names,
    )

    return train_dataset, val_dataset, class_freq


def create_dataloaders(
    train_dataset: AudioDataset,
    val_dataset: AudioDataset,
    **kwargs: Dict[str, Any]
) -> tuple[DataLoader, DataLoader]:
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_dataset.df['weight'], len(train_dataset)
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **kwargs.get("train_args", {})
    )

    val_loader = DataLoader(val_dataset, **kwargs.get("val_args", {}))

    return train_loader, val_loader