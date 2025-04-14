import torchmetrics

KEY2METRICS = {
    "f1" : torchmetrics.F1Score,
    'recall':torchmetrics.Recall,
    'precision':torchmetrics.Precision,
    'accuracy':torchmetrics.Accuracy,
    'rocauc': torchmetrics.AUROC

}

