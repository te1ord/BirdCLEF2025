import torch

KEY2LOSSES = {
    "bce" : torch.nn.BCEWithLogitsLoss,
    'ce': torch.nn.CrossEntropyLoss,
}

