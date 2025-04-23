import os
import torch
import librosa
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from itertools import chain

from src.models.spec_cnn import SpecCNNClassifier


class Inference:
    def __init__(
        self,
        model_path: str,
        class_labels: List[str],
        sample_rate: int,
        target_duration: float,
        device: str,
        batch_size: int,
        model_config: Optional[dict],
    ):
        self.model_path = model_path
        self.class_labels = class_labels
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.device = device
        self.batch_size = batch_size
        self.model_config = model_config or {}
        
        self.model = self._load_model()
        self.model.eval()

        self.logger = []
        
    def _load_model(self) -> SpecCNNClassifier:
        """Load the trained model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.model_config['pretrained'] = False
        model = SpecCNNClassifier(
            **self.model_config,
            device=self.device
        )
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
            # tmp fix of loss_func in state dict
            state_dict = {k: v for k, v in state_dict.items() if "loss_function" not in k}

            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        model = model.to(self.device)
        return model
    
    def _prepare_audio_chunk(self, audio: np.ndarray) -> torch.Tensor:
        """Prepare audio chunk for inference."""
        target_length = int(self.sample_rate * self.target_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            start_idx = max(0, int(len(audio) / 2 - target_length / 2))
            audio = audio[start_idx:start_idx + target_length]
            
        audio = torch.from_numpy(audio).float()
        return audio
    
    def _get_audio_chunks(self, audio_path: str):
        """Get all chunks from an audio file with their metadata."""
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            soundscape_id = Path(audio_path).stem
            
            chunk_size = int(self.sample_rate * self.target_duration)
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                yield (chunk, soundscape_id, i // chunk_size)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return []
    
    def _process_batch(self, batch: List[Tuple[np.ndarray, str, int]]) -> pd.DataFrame:
        """Process a batch of audio chunks and return predictions."""
        audio_chunks, soundscape_ids, chunk_indices = zip(*batch)

        # print(f"Soundscape ID, Chunk index = {list(zip(soundscape_ids, chunk_indices))}")
        self.logger.append(list(zip(soundscape_ids, chunk_indices)))
        
        batch_tensor = torch.stack([self._prepare_audio_chunk(chunk) for chunk in audio_chunks])
        batch_tensor = batch_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(batch_tensor)
            logits = output["logits"]
            scores = torch.sigmoid(logits).cpu().numpy()
            #torch.softmax(logits, dim=-1).cpu().numpy()
            
        
        predictions = pd.DataFrame(columns=['row_id'] + self.class_labels)
        
        for i, (soundscape_id, chunk_idx) in enumerate(zip(soundscape_ids, chunk_indices)):
            row_id = f"{soundscape_id}_{(chunk_idx + 1) * int(self.target_duration)}"
            new_row = pd.DataFrame([[row_id] + list(scores[i])], 
                                 columns=['row_id'] + self.class_labels)
            predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
            
        return predictions
    
    def predict_directory(self, directory_path: str) -> pd.DataFrame:
        """Make predictions for all audio files in a directory."""
        all_predictions = pd.DataFrame(columns=['row_id'] + self.class_labels)
        
        audio_files = [os.path.join(directory_path, afile) 
                            for afile in sorted(os.listdir(directory_path)) 
                            if afile.endswith('.ogg')]
        # print(audio_files)

        # test
        # audio_files = audio_files[:16]

        batch = []
        
        for audio_file in tqdm(audio_files, desc="Loading audio files"):
            
            for chunk in self._get_audio_chunks(audio_file):
                batch.append(chunk)

                if len(batch) == self.batch_size:
                    preds = self._process_batch(batch)

                    all_predictions = pd.concat([all_predictions, preds], 
                                      axis=0, ignore_index=True)
                    batch = []
        
        # finish remaining batches
        if batch:
            preds = self._process_batch(batch)
            all_predictions = pd.concat([all_predictions, preds], 
                                      axis=0, ignore_index=True)
        
        for i in self.logger:
            print(i)
            print()

        return all_predictions 