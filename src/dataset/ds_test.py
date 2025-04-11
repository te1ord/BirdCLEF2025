import os
import math
import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import uuid
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# Assuming you have these modules:
from src.dataset.audio_dataset import AudioDataset


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def test_dataset(cfg: DictConfig):
    # Initialize logger for this module and log the configuration details.
    logger = logging.getLogger(__name__)
    
    # Log the entire Hydra configuration for context.
    logger.info("Full configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    # Log the augmentation parameters.
    logger.info("Audio Augmentation Parameters:\n%s", OmegaConf.to_yaml(cfg.augmentation.audio))
    logger.info("Spectrogram Augmentation Parameters:\n%s", OmegaConf.to_yaml(cfg.augmentation.spectrogram))
    
    # Generate a unique run identifier.
    run_id = uuid.uuid4().hex
    logger.info("Unique run ID: %s", run_id)
    
    # 1) Create the output directory if it doesn't exist.
    os.makedirs(cfg.data.paths.output_dir, exist_ok=True)
    
    # 2) Load data
    #    We only load the first `cfg.data.test.num_samples` rows for demonstration.
    train_csv_path = cfg.data.paths.train_csv
    train_df = pd.read_csv(train_csv_path).head(cfg.data.test.num_samples)
    
    # 3) Adjust file paths in the 'filename' column so that they become absolute or joined paths.
    file_col = cfg.data.filenpath_col
    audio_dir = cfg.data.paths.train_audio_dir
    train_df[file_col] = train_df[file_col].apply(lambda x: os.path.join(audio_dir, x))
    
    print(f"Loaded {len(train_df)} samples from {train_csv_path}")
    print(train_df.head())
    
    # 4) Audio augmentations from config
    audio_transforms = Compose([
        AddGaussianNoise(**cfg.augmentation.audio.gaussian_noise),
        TimeStretch(**cfg.augmentation.audio.time_stretch),
        PitchShift(**cfg.augmentation.audio.pitch_shift),
    ])
    
    # 5) Spectrogram augmentations from config
    spec_augment_config = {
        "freq_mask": cfg.augmentation.spectrogram.freq_mask,
        "time_mask": cfg.augmentation.spectrogram.time_mask,
    }
    
    # 6) Build dictionary for spectrogram parameters:
    spec_params = {
        "sample_rate": cfg.data.sample_rate,
        "n_fft": cfg.data.n_fft,
        "hop_length": cfg.data.hop_length,
        "n_mels": cfg.data.n_mels,
        "f_min": cfg.data.f_min,
        "f_max": cfg.data.f_max,
        # "top_db": cfg.data.top_db
    }
    
    # 7) Instantiate the dataset
    dataset = AudioDataset(
        input_df=train_df,
        filenpath_col=cfg.data.filenpath_col,
        target_col=cfg.data.target_col,
        sample_rate=cfg.data.sample_rate,
        target_duration=cfg.data.target_duration,
        normalize_audio=cfg.data.normalize_audio,
        audio_transforms=audio_transforms,
        spec_augment_config=spec_augment_config,
        spec_params=spec_params,  # pass the dict from above
    )
    
    # 8) Collect some waveforms & spectrograms for debugging / plotting
    waveforms = []
    specs = []
    class_names = []
    
    for idx in range(len(train_df)):
        # Get file path & label.
        filepath = dataset.df[dataset.filenpath_col].iloc[idx]
        label = dataset.df[dataset.target_col].iloc[idx]
        
        # Load raw audio.
        au, sr = librosa.load(filepath, sr=dataset.sample_rate)
        
        # Ensure length = target_samples.
        if len(au) < dataset.target_samples:
            au = np.pad(au, (0, dataset.target_samples - len(au)), mode='constant')
        else:
            # Center extraction if too long.
            start_idx = max(0, len(au) // 2 - dataset.target_samples // 2)
            end_idx = start_idx + dataset.target_samples
            au = au[start_idx:end_idx]
        
        waveforms.append(au)
        class_names.append(label)
        
        # Convert to tensor & get spectrogram.
        au_tensor = torch.from_numpy(au).float()
        spec_tensor = dataset.spectogram_extractor(au_tensor)
        spec_np = spec_tensor.squeeze().numpy()
        specs.append(spec_np)
    
    # 9) Plot waveforms & spectrograms.
    def plot_wave_spectrograms(
        waveforms, sample_rate, class_names, num_cols=2, specs=None, save_path=None
    ):
        num_files = len(waveforms)
        num_rows = math.ceil(num_files / num_cols) * 2  # 2 rows per sample (spectrogram + waveform)
        
        fig, axs = plt.subplots(
            num_rows, num_cols,
            figsize=(num_cols * cfg.data.test.figsize_width, num_rows * cfg.data.test.figsize_height)
        )
        
        # Ensure axs is 2D even if there's only one row.
        if num_rows == 2 and num_cols == 1:
            axs = np.reshape(axs, (num_rows, num_cols))
        
        for idx, (waveform, class_name) in enumerate(zip(waveforms, class_names)):
            # The spectrogram is in row i and the waveform is in row i+1.
            i, j = (idx // num_cols) * 2, idx % num_cols
            
            # Plot spectrogram.
            if specs is not None:
                axs[i][j].imshow(specs[idx], aspect='auto', origin='lower')
            else:
                freqs, times, sxx = signal.spectrogram(waveform, sample_rate)
                axs[i][j].pcolormesh(times, freqs, 10 * np.log10(sxx + 1e-15))
            axs[i][j].set_title(f"{class_name}", fontsize=10)
            axs[i][j].axis("off")
            
            # Plot waveform.
            axs[i+1][j].plot(waveform)
            axs[i+1][j].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        plt.close()
    
    # 10) Save the plot with the unique run identifier in the filename.
    output_path = os.path.join(cfg.data.paths.output_dir, f"test_samples_{run_id}.png")
    plot_wave_spectrograms(
        waveforms, 
        dataset.sample_rate, 
        class_names, 
        num_cols=cfg.data.test.num_cols,
        specs=specs,
        save_path=output_path
    )


if __name__ == "__main__":
    test_dataset()
