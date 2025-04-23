import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional
from torchaudio.transforms import MelSpectrogram
from src.augmentations import ChannelAgnosticAmplitudeToDB, NormalizeMelSpec, CustomFreqMasking, CustomTimeMasking

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
        timm_kwargs: Optional[Dict],
        spec_augment_config: Optional[Dict[str, Any]],
    ):
        super().__init__()
        timm_kwargs = {} if timm_kwargs == "None" else timm_kwargs
        self.device = device

        self.spectogram_extractor = nn.Sequential(
            MelSpectrogram(**spec_params),
            ChannelAgnosticAmplitudeToDB(top_db=top_db),
            NormalizeMelSpec(**normalize_config),
        )
        
        if spec_augment_config is not None:
            self.spec_augment = []
            if "freq_mask" in spec_augment_config:
                self.spec_augment.append(CustomFreqMasking(**spec_augment_config["freq_mask"]))
            if "time_mask" in spec_augment_config:
                self.spec_augment.append(CustomTimeMasking(**spec_augment_config["time_mask"]))
            self.spec_augment = nn.Sequential(*self.spec_augment)
        else:
            self.spec_augment = None

        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=1,
            exportable=True,
            **timm_kwargs,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.backbone.feature_info.channels()[-1], n_classes),
        )
        
        self.to(self.device)

    def forward(self, input, return_spec_feature=False, return_cnn_emb=False):
        specs = self.spectogram_extractor(input)
        if self.spec_augment is not None and self.training:
            specs = self.spec_augment(specs)
        if return_spec_feature:
            return specs

        # Add Channel Dimension
        emb = self.backbone(specs.unsqueeze(1))[-1]
        if return_cnn_emb:
            return emb

        bs, ch, h, w = emb.shape
        emb = self.pool(emb)
        emb = emb.view(bs, ch)

        logits = self.classifier(emb)

        return {"logits": logits} 