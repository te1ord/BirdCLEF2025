import torch.nn as nn
import torch

class GeMGlobal(nn.Module):
    def __init__(self, p: float = 3., eps: float = 1e-6):
        super().__init__()
        self.p    = nn.Parameter(torch.ones(1) * p)
        self.eps  = eps

        # this becomes ONNX GlobalAveragePool
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        # x: [B, C, H, W]
        # 1) clamp & power
        x = x.clamp(min=self.eps).pow(self.p)
        # 2) global average → [B, C, 1, 1]
        x = self.pool(x)
        # 3) root
        x = x.pow(1.0 / self.p)
        # 4) flatten to [B, C]
        return x.view(x.size(0), x.size(1))