import os
import torch
import pandas as pd
from typing import Dict, Any, Optional, Callable, Union
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.datasets.audio_dataset import AudioDataset
from src.models import SpecCNNClassifier
from src.trainer import LitTrainer, AudioForward
import lightning

def prepare_data(
    df: pd.DataFrame,
    audio_dir: str,
    sample_rate: int,
    duration: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    f_min: int,
    f_max: int,
    audio_augmentations: Optional[Callable] = None,
    spec_augmentations: Optional[Callable] = None,
    train: bool = True,
) -> AudioDataset:
    return AudioDataset(
        df=df,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        audio_augmentations=audio_augmentations,
        spec_augmentations=spec_augmentations,
        train=train,
    )

def create_datasets(
    df: pd.DataFrame,
    audio_dir: str,
    sample_rate: int,
    duration: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    f_min: int,
    f_max: int,
    audio_augmentations: Optional[Callable] = None,
    spec_augmentations: Optional[Callable] = None,
    train: bool = True,
) -> tuple[AudioDataset, AudioDataset]:
    train_df = df[df["fold"] != 0]
    val_df = df[df["fold"] == 0]

    train_dataset = prepare_data(
        df=train_df,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        audio_augmentations=audio_augmentations,
        spec_augmentations=spec_augmentations,
        train=train,
    )

    val_dataset = prepare_data(
        df=val_df,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        audio_augmentations=None,
        spec_augmentations=None,
        train=False,
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

def train(
    df: pd.DataFrame,
    audio_dir: str,
    sample_rate: int,
    duration: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    f_min: int,
    f_max: int,
    audio_augmentations: Optional[Callable] = None,
    spec_augmentations: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    accelerator: str = "gpu",
    devices: int = 1,
    precision: str = "16-mixed",
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    val_check_interval: float = 1.0,
    log_every_n_steps: int = 1,
    enable_checkpointing: bool = True,
    enable_progress_bar: bool = True,
    enable_model_summary: bool = True,
    overfit_batches: int = 0,
    deterministic: bool = True,
    benchmark: bool = False,
    reload_dataloaders_every_n_epochs: int = 0,
    default_root_dir: str = "checkpoints",
    logger: Optional[Any] = None,
    callbacks: Optional[list] = None,
) -> None:
    train_dataset, val_dataset = create_datasets(
        df=df,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        audio_augmentations=audio_augmentations,
        spec_augmentations=spec_augmentations,
        train=True,
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = SpecCNNClassifier(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        num_classes=len(df["primary_label"].unique()),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
    )

    trainer = LitTrainer(
        model=model,
        forward=AudioForward(
            loss_function=torch.nn.CrossEntropyLoss(),
            output_key="logits",
            input_key="targets",
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_params={
            "interval": "epoch",
            "frequency": 1,
        },
        batch_key="specs",
        metric_input_key="targets",
        metric_output_key="predictions",
        val_metrics=None,
        train_metrics=None,
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
        overfit_batches=overfit_batches,
        deterministic=deterministic,
        benchmark=benchmark,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        default_root_dir=default_root_dir,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(
        model=trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    ) 