import torch
import torchvision


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float,
        gamma: float,
        reduction: str,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class FocalLossBCE(torch.nn.Module):
    def __init__(
        self,
        alpha: float,
        gamma: float,
        reduction: str,
        bce_weight: float,
        focal_weight: float,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(inputs, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss