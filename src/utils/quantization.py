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
        # grab the backbone & head as before
        self.backbone    = speccnn.backbone
        self.pools       = speccnn.pool
        self.classifier  = speccnn.classifier
        self.out_indices = speccnn.out_indices
        self.n_specs = speccnn.n_specs

    def forward(self, specs: torch.Tensor):
        """
        specs: [B, n_mels, T]  (already MelSpectrogram + DB + Norm)
        """

        # 1) turn into [B, 1, F, T]
        x = specs.unsqueeze(1)

        # 2) fake-RGB if needed → [B, n_specs, F, T]
        if self.n_specs > 1:
            x = x.expand(-1, self.n_specs, -1, -1).contiguous()

        # 3) rest is unchanged
        feats = self.backbone(x)
        if self.out_indices is None:
            feats = [feats[-1]]

        pooled = [g(f) for f, g in zip(feats, self.pools)]
        emb = torch.cat(pooled, dim=1)
        return self.classifier(emb)
