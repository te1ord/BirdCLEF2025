import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
from onnxruntime.quantization import (
    CalibrationDataReader,
)

class SpectrogramCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name: str, specs_list: List[np.ndarray]):
        self.input_name = input_name
        self.specs_list = specs_list
        self._iter = iter(self.specs_list)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        try:
            specs = next(self._iter)
            return {self.input_name: specs.astype(np.float32)}
        except StopIteration:
            return None

    def rewind(self):
        self._iter = iter(self.specs_list)


class CNNWrapper(torch.nn.Module):
    def __init__(self, speccnn: torch.nn.Module):
        super().__init__()
        # grab just the parts we want to quantize
        self.backbone   = speccnn.backbone
        self.pools      = speccnn.pool       # ModuleList of GeMGlobal
        self.classifier = speccnn.classifier
        self.out_indices = speccnn.out_indices

    def forward(self, specs: torch.Tensor):
        """
        specs: [B, n_mels, T]  (already run through MelSpectrogram+DB+Norm)
        """
        feats = self.backbone(specs.unsqueeze(1))

        if self.out_indices is None:
            feats = [feats[-1]]

        pooled = [ g(f) for f, g in zip(feats, self.pools) ]
        emb = torch.cat(pooled, dim=1)

        return self.classifier(emb)
