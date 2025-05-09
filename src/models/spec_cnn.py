import numpy as np
import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional, List
from torchaudio.transforms import MelSpectrogram
from src.augmentations import ChannelAgnosticAmplitudeToDB, NormalizeMelSpec, CustomFreqMasking, CustomTimeMasking
from .blocks import GeMGlobal

class SpecCNNClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        device: str,
        n_classes: int,
        classifier_dropout: float,
        spec_params: Dict[str, Any],
        top_db: float,
        normalize_config: Dict[str, bool],
        pretrained: bool,
        pool_type: str,
        out_indices: List[int],
        in_chans: int,
        timm_kwargs: Optional[Dict],
        spec_augment_config: Optional[Dict[str, Any]]
        
    ):
        super().__init__()
        timm_kwargs = {} if timm_kwargs == "None" else timm_kwargs
        self.out_indices = None if out_indices == "None" else tuple(out_indices)
        self.n_specs = in_chans

        self.device = device

        self.spectogram_extractor = nn.Sequential(
            MelSpectrogram(**spec_params),
            ChannelAgnosticAmplitudeToDB(top_db=top_db),
            NormalizeMelSpec(**normalize_config),
        )
        
        # augments
        if spec_augment_config is not None:
            self.spec_augment = []
            if "freq_mask" in spec_augment_config:
                self.spec_augment.append(CustomFreqMasking(**spec_augment_config["freq_mask"]))
            if "time_mask" in spec_augment_config:
                self.spec_augment.append(CustomTimeMasking(**spec_augment_config["time_mask"]))
            self.spec_augment = nn.Sequential(*self.spec_augment)
        else:
            self.spec_augment = None

        # model
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=self.n_specs,
            exportable=True,
            out_indices=self.out_indices,
            **timm_kwargs,
        )

        print(self.backbone.feature_info.channels())
        
        feature_dims = self.backbone.feature_info.channels() if self.out_indices is not None else [self.backbone.feature_info.channels()[-1]]
        print(f"feature dims: {feature_dims}")

        # pooling
        pools: List[nn.Module] = []
        if pool_type.lower() == "gem":
            pools = [GeMGlobal() for _ in feature_dims]
        elif pool_type.lower() == "adavg":
            pools = [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(start_dim=1)
                )
                for _ in feature_dims
            ]
        else:
            raise ValueError(f"Unsupported pool_type={pool_type!r}; choose 'gem' or 'avg'")
        
        self.pool = nn.ModuleList(pools)

        self.emb_dim = sum(feature_dims)

        # head
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.emb_dim, n_classes),
        )
        
        self.to(self.device)

    def forward(self, input, return_spec_feature=False, return_cnn_emb=False):

        # specs
        specs = self.spectogram_extractor(input)

        # multi channel mode support
        specs = specs.unsqueeze(1).expand(-1, self.n_specs, -1, -1).contiguous()

        if self.spec_augment is not None and self.training:
            specs = self.spec_augment(specs)
        if return_spec_feature:
            return specs

        # features - list of stages            
        features = self.backbone(specs)

        if self.out_indices is None:
            features = [features[-1]]

        pooled = [p(fmap) for fmap, p in zip(features, self.pool)]
        
        emb = torch.cat(pooled, dim=1)

        if return_cnn_emb:
            return emb

        logits = self.classifier(emb)

        return {"logits": logits} 