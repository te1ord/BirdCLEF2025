import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
from audiomentations import Compose
from src.models.spec_cnn import SpecCNNClassifier
from src.utils.data import create_datasets, create_dataloaders
from src.trainers import AudioForward, LitTrainer
from src.augmentations.audio_augmentations import KEY2AUDIO_AUGMENTATION

import wandb
import lightning
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from torchmetrics import MetricCollection
from src.metrics import KEY2METRICS
from src.losses import KEY2LOSSES
from src.utils.other import seed_everything



@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    df = pd.read_csv(cfg.data.paths.train_csv)

    # Parse audio transforms
    audio_transforms = []
    if hasattr(cfg.augmentation.audio, "audio_transforms") and cfg.augmentation.audio.audio_transforms:
        audio_transforms = [
            KEY2AUDIO_AUGMENTATION[t.name](**t.params)
            for t in cfg.augmentation.audio.audio_transforms
        ]

    # Parse metrics
    metrics = [
        KEY2METRICS[name](**cfg.training.metrics.params)
        for name in cfg.training.metrics.names
    ]

    train_dataset, val_dataset = create_datasets(
        df=df,
        audio_dir=cfg.data.paths.audio_dir,
        **cfg.data.dataset_args,
        audio_transforms=Compose(audio_transforms),
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **cfg.data.dataloader_args
    )


    model = SpecCNNClassifier(
        **cfg.model,
        spec_augment_config=dict(cfg.augmentation.spectrogram) if hasattr(cfg.augmentation, "spectrogram") else None ,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **cfg.training.optimizer.params
    )

    total_steps = len(train_loader) * cfg.training.trainer.n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        **cfg.training.scheduler.params
    )

    lightning_model = LitTrainer(
        model=model,
        forward=AudioForward(
            loss_function=KEY2LOSSES[cfg.training.forward.loss_function](),
            output_key=cfg.training.forward.output_key,
            input_key=cfg.training.forward.input_key,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_params={**cfg.training.scheduler.config},
        batch_key=cfg.training.forward.batch_key,
        metric_input_key=cfg.training.metrics.input_key,
        metric_output_key=cfg.training.metrics.output_key,
        val_metrics=MetricCollection(metrics, compute_groups=False),
        train_metrics=MetricCollection(metrics, compute_groups=False),
    )

    # checkpoint callback
    checkpoint_callback_params = dict(
        save_last=cfg.callbacks.save_last,
        auto_insert_metric_name=cfg.callbacks.auto_insert_metric_name,
        save_weights_only=cfg.callbacks.save_weights_only,
        save_on_train_epoch_end=cfg.callbacks.save_on_train_epoch_end,
        filename="{epoch}-{step}-{valid_MultilabelAUROC:.3f}",
    )
    
    ckpt_dir = os.path.join("logdirs", cfg.experiment_name, "checkpoints")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=cfg.callbacks.n_checkpoints_to_save,
        mode=cfg.callbacks.metric_mode,
        monitor=cfg.callbacks.main_metric,
        **checkpoint_callback_params,
    )

    all_callbacks = [
        ckpt_cb,
        LearningRateMonitor(logging_interval="step"),
    ]

    wandb_logger = pl_loggers.WandbLogger(
        save_dir=ckpt_dir,
        **cfg.wandb,
    )

    wandb_logger.log_hyperparams(
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    if torch.cuda.is_available():
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        print("MPS device found.")
    else:
        accelerator = "cpu"

    trainer = lightning.Trainer(
        accelerator=accelerator,
        devices="auto",
        precision=cfg.training.trainer.precision_mode,
        strategy=cfg.training.trainer.train_strategy,
        max_epochs=cfg.training.trainer.n_epochs,
        logger=wandb_logger,
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        val_check_interval=cfg.training.trainer.get("val_check_interval"),
        callbacks=all_callbacks,
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
