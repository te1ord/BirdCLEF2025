import os
import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from src.inference.inference import Inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    test_soundscape_path = Path(cfg.inference.test_soundscape_path)
    
    ss_df = pd.read_csv(cfg.data.paths.sample_submission)
    class_labels = sorted(ss_df.columns[1:].tolist())
    
    # Set up inference configuration
    inference_config = {
        'ensemble': cfg.inference.ensemble,
        'device': cfg.inference.device,
        'batch_size': cfg.inference.batch_size,
        'class_labels': class_labels,
        'sample_rate': cfg.data.dataset_args.train_args.sample_rate,
        'target_duration': cfg.data.dataset_args.train_args.target_duration,
    }
    
    for ii, model in enumerate(cfg.inference.ensemble.models):
        path = Path(model.path)
        cfg.inference.ensemble.models[ii].path = path / 'model.ckpt'

        model_config = OmegaConf.load(path / 'config.yaml')
        model_config = {
            "backbone": model_config.model.backbone,
            "n_classes": model_config.model.n_classes,
            "classifier_dropout": model_config.model.classifier_dropout,
            "top_db": model_config.model.top_db,
            "spec_params": dict(model_config.model.spec_params),
            "normalize_config": dict(model_config.model.normalize_config),
            "pretrained": model_config.model.pretrained,
            "spec_augment_config": None,
            "timm_kwargs": model_config.model.timm_kwargs,
            "out_indices": model_config.model.out_indices,
            "pool_type": model_config.model.pool_type,
            "in_chans": model_config.model.in_chans  
        }
        cfg.inference.ensemble.models[ii].model_cfg = model_config


    quantization_config = {
        'quantization_type': cfg.inference.quantization.quantization_type,
        'per_channel': cfg.inference.quantization.per_channel,
        'calibration_data_path': cfg.inference.quantization.calibration_data_path,
        'n_calibration_samples': cfg.inference.quantization.n_calibration_samples,
        'onnx_dir': cfg.inference.quantization.onnx_dir

    }
    smoothing_config = {
        'temporal_smoothing_type': cfg.inference.temporal_smoothing.temporal_smoothing_type,
        'temporal_smoothing_params': cfg.inference.temporal_smoothing.temporal_smoothing_params
    }

    inference = Inference(
        inference_cfg = inference_config,
        quantization_cfg = quantization_config,
        smoothing_cfg = smoothing_config
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
    
