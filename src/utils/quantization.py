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
    def __init__(self, mdl):
        super().__init__()
        self.backbone = mdl.backbone
        self.pool = mdl.pool
        self.classifier = mdl.classifier

    def forward(self, specs):
        emb = self.backbone(specs.unsqueeze(1))[-1]
        emb = self.pool(emb).reshape(emb.size(0), -1)
        return self.classifier(emb)