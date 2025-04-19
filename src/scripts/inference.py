import os
import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path

from src.inference.inference import Inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    model_path = Path(cfg.inference.checkpoint_path)
    test_soundscape_path = Path(cfg.inference.test_soundscape_path)
    
    
    df = pd.read_csv(cfg.data.paths.train_csv)
    class_labels = sorted(df.primary_label.unique())
    #sorted(os.listdir(cfg.data.train_audio_path))
    
    print(f"LEN LABELS:{len(class_labels)}")
    
    model_config = {
        "backbone": cfg.model.backbone,
        "n_classes": cfg.model.n_classes,
        "classifier_dropout": cfg.model.classifier_dropout,
        "top_db": cfg.model.top_db,
        "spec_params": dict(cfg.model.spec_params),
        "normalize_config": dict(cfg.model.normalize_config),
        "pretrained": cfg.model.pretrained,
    }
    
    inference = Inference(
        model_path=str(model_path),
        class_labels=class_labels,
        sample_rate=cfg.data.dataset_args.train_args.sample_rate,
        target_duration=cfg.data.dataset_args.train_args.target_duration,
        device=cfg.device,
        batch_size = cfg.inference.batch_size,
        model_config=model_config,
    )
    
    predictions = inference.predict_directory(str(test_soundscape_path))
    
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_path = Path(cfg.output_dir) / "submission.csv"
    
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print("\nFirst few predictions:")
    print(predictions.head())


if __name__ == "__main__":
    main() 