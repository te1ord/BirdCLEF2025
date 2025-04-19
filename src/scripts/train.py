import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
from typing import Any, Optional, Callable
from audiomentations import Compose
from src.models.spec_cnn import SpecCNNClassifier
from src.utils.data import create_datasets, create_dataloaders
from src.trainers import AudioForward, LitTrainer
from src.augmentations.audio_augmentations import KEY2AUDIO_AUGMENTATION

import wandb
import lightning
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
import torchmetrics
from torchmetrics import MetricCollection
from src.metrics import KEY2METRICS
from src.utils.other import seed_everything

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    
    df = pd.read_csv(cfg.data.paths.train_csv)

    # Parse audio transforms
    audio_trasforms = [
        KEY2AUDIO_AUGMENTATION[transform.name](**transform.params)
        for transform in cfg.augmentation.audio.audio_transforms
        ]
    
    # parse metrics
    metrics = [
        KEY2METRICS[name](**cfg.training.metrics.params)
        for name in cfg.training.metrics.names
        ]
    
    
    train_dataset, val_dataset = create_datasets(
        df=df,
        audio_dir=cfg.data.paths.audio_dir,
        **cfg.data.dataset_args,
        audio_transforms=Compose(audio_trasforms),
        mixup_audio=cfg.augmentation.audio.mixup_audio,
        mixup_params=dict(cfg.augmentation.audio.mixup_params)
        )
    
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **cfg.data.dataloader_args
    )

    # Calculate total number of steps for scheduler
    total_steps = len(train_loader) * cfg.training.trainer.n_epochs

    model = SpecCNNClassifier(
        **cfg.model,
        spec_augment_config=dict(cfg.augmentation.spectrogram),
    )

    # TODO: Implemet normal config for this
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **cfg.training.optimizer.params
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        **cfg.training.scheduler.params
    )

    lightning_model = LitTrainer(
        model=model,
        forward=AudioForward(
            loss_function=torch.nn.BCEWithLogitsLoss(),
            output_key="logits",
            input_key="targets",
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_params= {**cfg.training.scheduler.config},
        batch_key="specs",
        metric_input_key="targets_1d",
        metric_output_key="predictions",
        val_metrics= MetricCollection(metrics, compute_groups=False),
        train_metrics=MetricCollection(metrics, compute_groups=False),
    )


    checkpoint_callback_params=dict(
        save_last=True,
        auto_insert_metric_name=True,
        save_weights_only=True,
        save_on_train_epoch_end=True,
        filename="{epoch}-{step}-{valid_MultilabelAUROC:.3f}",
    )

    all_callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.experiment_name, "checkpoints"),
            save_top_k=cfg.callbacks.n_checkpoints_to_save,
            mode=cfg.callbacks.metric_mode,
            monitor=cfg.callbacks.main_metric,
            **checkpoint_callback_params,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    wandb_logger = pl_loggers.WandbLogger(
        save_dir=cfg.experiment_name,
        **cfg.wandb,
    )

    os.environ['WANDB_LOG_MODEL'] = 'checkpoint'

    if torch.cuda.is_available():
        accelerator = 'cuda'
    elif torch.backends.mps.is_available():
        accelerator = 'mps'
        print("MPS device found.")
    else:
        accelerator = 'cpu'

    trainer = lightning.Trainer(
        accelerator=accelerator,
        devices='auto',
        precision=cfg.training.trainer.precision_mode,
        strategy=cfg.training.trainer.train_strategy,
        max_epochs=cfg.training.trainer.n_epochs,
        logger=wandb_logger,
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        val_check_interval=cfg.training.trainer.get('val_check_interval'),
        callbacks=all_callbacks,
        # **trainer_params,
    )
    trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    wandb.finish()


if __name__ == "__main__":
    main()