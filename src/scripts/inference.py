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
    
    ss_df = pd.read_csv(cfg.data.paths.sample_submission)
    class_labels = sorted(ss_df.columns[1:].tolist())
    
    # Set up inference configuration
    inference_config = {
        'model_path': str(model_path),
        'device': cfg.inference.device,
        'batch_size': cfg.inference.batch_size,
        'class_labels': class_labels,
        'sample_rate': cfg.data.dataset_args.train_args.sample_rate,
        'target_duration': cfg.data.dataset_args.train_args.target_duration,
    }
    
    # Add ensemble configuration if present
    if hasattr(cfg.inference, 'ensemble'):
        inference_config['ensemble'] = dict(cfg.inference.ensemble)
        
        if inference_config['ensemble']['enabled']:
            print(f"Using ensemble of {len(inference_config['ensemble']['models'])} models with weights: {[m.weight for m in cfg.inference.ensemble.models]}")

    model_config = {
        "backbone": cfg.model.backbone,
        "n_classes": cfg.model.n_classes,
        "classifier_dropout": cfg.model.classifier_dropout,
        "top_db": cfg.model.top_db,
        "spec_params": dict(cfg.model.spec_params),
        "normalize_config": dict(cfg.model.normalize_config),
        "pretrained": cfg.model.pretrained,
        "spec_augment_config": None,
        "timm_kwargs": cfg.model.timm_kwargs,
        "out_indices": cfg.model.out_indices,
        "pool_type": cfg.model.pool_type      
    }

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
        model_cfg=model_config,
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
    
