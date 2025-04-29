import random
import os
import torch
import librosa
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import gc
from src.models.spec_cnn import SpecCNNClassifier
from src.utils.temp_smoother import Smoothing
from src.utils.quantization import SpectrogramCalibrationDataReader, CNNWrapper

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader,
)

from onnxruntime.quantization.shape_inference import quant_pre_process

class Inference:
    def __init__(
        self,
        inference_cfg,
        model_cfg,
        quantization_cfg,
        smoothing_cfg

    ):
        self.inference_cfg = inference_cfg
        self.model_cfg = model_cfg
        self.quantization_cfg = quantization_cfg
        self.smoothing_cfg = smoothing_cfg

        self.model = self._load_model().eval()


        if self.quantization_cfg['quantization_type'] != "none":
            self.spectogram_extractor = self.model.spectogram_extractor
            self._export_and_quantize()
            del self.model
            torch.cuda.empty_cache()
            self.model = None
            gc.collect()
        else:
            self.onnx_session = None

        self.logger = []

    def _export_and_quantize(self):

        # cnn wrapper to quantize only model (not spec extractor)
        wrapper = CNNWrapper(self.model).to(self.inference_cfg['device'] ).eval()

        # get dummy spec
        dummy_audio = torch.randn(1, int(self.inference_cfg['sample_rate'] * self.inference_cfg['target_duration']))
        with torch.no_grad():
            dummy_spec = self.spectogram_extractor(dummy_audio)

        # export to ONNX
        onnx_fp = os.path.join(self.quantization_cfg['onnx_dir'], "cnn_subnet.onnx")
        torch.onnx.export(
            wrapper, (dummy_spec,), onnx_fp,
            input_names=["specs"], output_names=["logits"],
            dynamic_axes={"specs":{0:"batch"}, "logits":{0:"batch"}},
            opset_version=13,
        )

        # pre-process the ONNX graph for better quantization
        optimized_fp = onnx_fp.replace(".onnx", "_optimized.onnx")
        quant_pre_process(onnx_fp, optimized_fp)
        quant_source = optimized_fp

        if self.quantization_cfg['quantization_type'] == "onnx":
            quant_fp = onnx_fp  
        
        elif self.quantization_cfg['quantization_type'] == "dynamic":
            
            quant_fp = onnx_fp.replace(".onnx", "_quant.onnx")
            quantize_dynamic(
                quant_source, quant_fp,
                weight_type=QuantType.QUInt8,
                per_channel=self.quantization_cfg['per_channel']
            )

        elif self.quantization_cfg['quantization_type'] == "static":
            if not self.quantization_cfg['calibration_data_path']:
                raise ValueError("Static quantization requires 'calibration_data_path' to a folder of audio files.")
            
            quant_fp = onnx_fp.replace(".onnx", "_quant.onnx")
            
            # build calibration specs list 
            audio_files = [
                os.path.join(self.quantization_cfg['calibration_data_path'], f)
                for f in os.listdir(self.quantization_cfg['calibration_data_path'])
                if f.endswith('.ogg') or f.endswith('.wav')
            ]
            random.shuffle(audio_files)

            specs_list = []
            for af in audio_files:
                audio, _ = librosa.load(af, sr=self.inference_cfg['sample_rate'])
                chunk = self._prepare_audio_chunk(audio)
                with torch.no_grad():
                    spec_t = self.spectogram_extractor(chunk.unsqueeze(0))
                specs_list.append(spec_t.cpu().numpy())
                if len(specs_list) >= self.quantization_cfg['n_calibration_samples']:
                    break

            reader = SpectrogramCalibrationDataReader(
                input_name="specs",
                specs_list=specs_list
            )
            quantize_static(
                quant_source, quant_fp,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
                per_channel=self.quantization_cfg['per_channel'],
            )
        else:
            raise ValueError(f"Unknown quantization_type={self.quantization_cfg['quantization_type']}")

        # load ORT session
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = os.cpu_count() or 1
        self.onnx_session = ort.InferenceSession(
            quant_fp, sess_opts, providers=["CPUExecutionProvider"]
        )
        self.onnx_in_name = self.onnx_session.get_inputs()[0].name
        self.onnx_out_name = self.onnx_session.get_outputs()[0].name

        # warm up
        self.onnx_session.run(
            [self.onnx_out_name],
            {self.onnx_in_name: dummy_spec.cpu().numpy().astype(np.float32)}
        )


    def _load_model(self) -> SpecCNNClassifier:
        """Load the trained model from checkpoint."""
        checkpoint = torch.load(self.inference_cfg['model_path'], map_location=self.inference_cfg['device'] )
        
        self.model_cfg['pretrained'] = False
        model = SpecCNNClassifier(
            **self.model_cfg,
            device=self.inference_cfg['device'] 
        )
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
            # tmp fix of loss_func in state dict
            state_dict = {k: v for k, v in state_dict.items() if "loss_function" not in k}

            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        model = model.to(self.inference_cfg['device'] )
        return model
    
    def _prepare_audio_chunk(self, audio: np.ndarray) -> torch.Tensor:
        """Prepare audio chunk for inference."""
        target_length = int(self.inference_cfg['sample_rate'] * self.inference_cfg['target_duration'])
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
            audio, _ = librosa.load(audio_path, sr=self.inference_cfg['sample_rate'])
            soundscape_id = Path(audio_path).stem
            
            chunk_size = int(self.inference_cfg['sample_rate'] * self.inference_cfg['target_duration'])
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                yield (chunk, soundscape_id, i // chunk_size)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return []
    
    def _process_batch(self, batch: List[Tuple[np.ndarray, str, int]]) -> pd.DataFrame:
        """Process a batch of audio chunks and return predictions."""
        audio_chunks, soundscape_ids, chunk_indices = zip(*batch)
        self.logger.append(list(zip(soundscape_ids, chunk_indices)))

        batch_tensor = torch.stack([self._prepare_audio_chunk(chunk) for chunk in audio_chunks]).to(self.inference_cfg['device'] )
        

        if self.quantization_cfg['quantization_type'] != "none":
            specs = self.spectogram_extractor(batch_tensor)  # [B, n_mels, T]
            specs_np = specs.cpu().numpy().astype(np.float32)
            logits_np = self.onnx_session.run(
                [self.onnx_out_name],
                {self.onnx_in_name: specs_np}
            )[0]
            scores = torch.sigmoid(torch.from_numpy(logits_np)).numpy()

        else:
            with torch.no_grad():
                output = self.model(batch_tensor)
                logits = output["logits"]
                scores = torch.sigmoid(logits).cpu().numpy()


        predictions = pd.DataFrame(columns=['row_id'] + self.inference_cfg['class_labels'])
        for i, (sid, idx) in enumerate(zip(soundscape_ids, chunk_indices)):
            row_id = f"{sid}_{(idx + 1) * int(self.inference_cfg['target_duration'])}"
            row = [row_id] + scores[i].tolist()
            predictions.loc[i] = row

        return predictions

    def predict_directory(self, directory_path: str) -> pd.DataFrame:
        """Make predictions for all audio files in a directory."""
        all_predictions = pd.DataFrame(columns=['row_id'] + self.inference_cfg['class_labels'])
        
        audio_files = [os.path.join(directory_path, afile) 
                            for afile in sorted(os.listdir(directory_path)) 
                            if afile.endswith('.ogg')]
        # print(audio_files)

        # test
        # audio_files = audio_files[:8]

        batch = []
        
        for audio_file in tqdm(audio_files, desc="Loading audio files"):
            
            for chunk in self._get_audio_chunks(audio_file):
                batch.append(chunk)

                if len(batch) == self.inference_cfg['batch_size'] :
                    preds = self._process_batch(batch)

                    all_predictions = pd.concat([all_predictions, preds], 
                                      axis=0, ignore_index=True)
                    batch = []

        
        # finish remaining batches
        if batch:
            preds = self._process_batch(batch)
            all_predictions = pd.concat([all_predictions, preds], 
                                      axis=0, ignore_index=True)
        
        #  temporal smoothing
        if self.smoothing_cfg['temporal_smoothing_type'] is not None:

            print(f"Using temporal smoothing {self.smoothing_cfg['temporal_smoothing_type']}")
            

            smoother = Smoothing(
                method=self.smoothing_cfg['temporal_smoothing_type'],
                params=self.smoothing_cfg['temporal_smoothing_params'][self.smoothing_cfg['temporal_smoothing_type']],
                class_labels=self.inference_cfg['class_labels']
            )

            all_predictions = smoother.apply(all_predictions)
        
        
        for i in self.logger:
            print(i)
            print()

        return all_predictions 
    
    