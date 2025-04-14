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


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    df = pd.read_csv(cfg.data.train_csv)

    # Parse audio transforms
    audio_trasforms = [
        KEY2AUDIO_AUGMENTATION[transform.name](**transform.params)
        for transform in cfg.augmentation.audio.audio_trasforms
        ]
    
    train_dataset, val_dataset = create_datasets(
        df=df,
        audio_dir=cfg.data.audio_dir,
        **cfg.data.dataset_args,
        audio_trasforms=Compose(audio_trasforms),
        mixup_audio=cfg.augmentation.audio.mixup_audio,
        mixup_params=dict(cfg.augmentation.audio.mixup_params)
        )

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **cfg.data.dataloader_args
    )


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
        **cfg.training.scheduler.params
    )

    trainer = LitTrainer(
        model=model,
        forward=AudioForward(
            loss_function=torch.nn.BCEWithLogitsLoss(),
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


    # all_callbacks = [
    #     ModelCheckpoint(
    #         dirpath=os.path.join(exp_name, "checkpoints"),
    #         save_top_k=n_checkpoints_to_save,
    #         mode=metric_mode,
    #         monitor=main_metric,
    #         **checkpoint_callback_params,
    #     ),
    #     LearningRateMonitor(logging_interval="step"),
    # ]

    # wandb_logger = pl_loggers.WandbLogger(
    #     save_dir=exp_name,
    #     name=exp_name,
    #     **wandb_logger_params,
    # )
    # trainer = lightning.Trainer(
    #     devices=-1,
    #     precision=precision_mode,
    #     strategy=train_strategy,
    #     max_epochs=n_epochs,
    #     logger=wandb_logger,
    #     log_every_n_steps=log_every_n_steps,
    #     callbacks=all_callbacks,
    #     **trainer_params,
    # )
    # trainer.fit(model=lightning_model, train_dataloaders=loaders["train"], val_dataloaders=loaders["valid"])
    # wandb.finish()