# BirdCLEF 2025 - Kaggle Competition

This repository contains our experimental framework for the **BirdCLEF 2025** Kaggle competition. The goal of this project was to explore various modeling strategies and audio preprocessing techniques to detect bird species from audio recordings.


# Solution Overview

## Exploratory Data Analysis (EDA)
  Detailed metadata analysis was performed in [eda.ipynb](https://github.com/te1ord/BirdCLEF2025/blob/main/notebooks/eda.ipynb) and [dataset_test.ipynb](https://github.com/te1ord/BirdCLEF2025/blob/main/notebooks/dataset_test.ipynb) to understand data distribution and gain insights about potential validation splits.

**Key Discoveries:**
- Highly imbalanced data distribution
- RocAUC target metric with macro-averaging, which weights all classes equally
- Potential data leakage with identical recordings appearing in multiple collections
- Potential data leakage from multiple audio splits from single large recordings by the same author
- Potential use of Rating column for sample weighting
- Potential use of secondary labels for multilabel optimization
- Presence of [human voices](https://www.kaggle.com/competitions/birdclef-2025/discussion/568886) in the CSA collection (affecting many minority classes)

### Data Preprocessing
Considering previous findings, a stratified split was implemented to correctly evaluate performance on minority classes while **removing near-duplicate samples** (e.g., similar recordings from different collections) to avoid potential leakage.

The split logic is documented in [cv_split.ipynb](https://github.com/te1ord/BirdCLEF2025/blob/main/notebooks/cv_split.ipynb).

**TODO:** Explore stratified split by target grouped by author to further minimize chances of leakage.

## Framework Architecture
The framework design facilitates easy iteration to test multiple hypotheses with robust tracking and comparison, requiring minimal changes to the source code.

Built:
- A modular training pipeline that encapsulates each functional element within its module
- A configuration manager with Hydra for flexible and clean experimentation
- Weights & Biases (wandb) logger that tracks resolved Hydra config and model checkpoints
- An "MLOps" pipeline for inference on Kaggle

## Modeling Approach

Our multilabel classification pipeline includes these key components:

- Custom dataset class with mixup augmentation operating on raw audio level, forming training examples with corresponding soft/hard labels
- Dataloader sampler weights to increase sampling frequency for minority classes
- Custom model extracting mel spectrograms (on GPU) and passing them to a backbone with linear head
- Optimization using AdamW combined with cosine learning rate scheduler

## Experimental Setup

### Evaluation Strategy

To ensure robust evaluation, fold 0 was manually configured to include all classes. Following common practice for Kaggle competitions, we tracked only the target RocAUC metric during experiments.

After several iterations, it became clear that local validation was ineffective. Community discussions revealed that all participants faced the same issue: validation RocAUC scores of ~0.99 resulting in leaderboard (LB) scores ranging from 0.7 to 0.9, indicating a significant domain shift in the hidden test set.

Experiments confirmed no consistent correlation between local validation and public leaderboard scores, with relationships ranging from positive to negative across different hyperparameters.

This inconsistency led us to use the public leaderboard as our validation method, submitting checkpoints from the same epochs to evaluate each experiment and determine the right direction.

**Note:** Setting up alternative metrics for class imbalance (like F1/PR-AUC) may help estimate classifier performance for different error types. However, without error analysis, this wouldn't help optimize the target RocAUC metric, which doesn't distinguish between different error types.

### Loss Function Experiments

Loss functions decsision in combination with soft/hard labels required some experiments.

Despite BCE loss being common for multilabel classification, its appropriateness was questioned when using mixup augmentation.

Here are two scenarios:

1. **Hard Labels**
   - With hard labels per sample (e.g., [1, 1, 0, ..., N]), BCE loss is conceptually appropriate as it pushes each class probability toward 1 for positive classes and 0 for negative classes
   - In contrast, CE loss with softmax tries to distribute probability mass between positive classes (p1 = p2 = 1/2), as softmax requires all probabilities to sum to 1, which isn't optimal when we want to maximize probabilities for all positive classes to improve RocAUC

2. **Soft Labels**
   - With soft labels, both BCE and CE converge toward the same minimum where predicted probabilities match the soft label distribution
   - However, BCE's class-independence seems less effective as it lacks the regularization constraint of summing to 1 present in CE loss
   - This class competition is crucial for recognizing appropriate features in mixed audio samples, as it pushes the model toward a balanced representation of features from both mixed sources

These considerations led to testing two configurations:
1. Soft labels + softmax + CE
2. Hard labels + sigmoid + BCE

**Experimental Results:**

<!-- **Soft Labels:**
- BCE: LB 0.759
- CE: LB 0.775

**Hard Labels:**
- BCE: LB 0.766
- CE: LB 0.774 -->

| Label Type | Loss Function | LB Score |
|------------|---------------|----------|
| Soft       | BCE           | 0.759    |
| Soft       | CE            | 0.775    |
| Hard       | BCE           | 0.766    |
| Hard       | CE            | 0.774    |


 Hypothesis was generally confirmed, though we unexpectedly found CE with hard labels performing well at 0.774. I hypothesize this occurred because larger gradients (due to the loss function's larger scale: 1 * log p + 1 * log p is larger than 0.3 * log p + 0.7 * log p) produce more aggressive updates. After softmax, this nearly zeroes out uncertain classes, resulting in fewer false positives and improved LB RocAUC scores. This suggests that ensembling models optimized with CE + hard labels with other less strict classifiers may increase robustness.

To better optimize for minority classes, two approaches  were tested  beyond dataloader weights:

1. **Class Weights**
   - Class weights for our best model decreased performance from 0.777 to 0.766

2. **Focal Loss**
   - Both Focal Loss and FocalBCE loss were tested with soft labels, yielding poorer results compared to previous experiments

<!-- - Focal: LB 0.741
- FocalBCE: LB 0.759 -->

| Label Type | Loss Function | LB Score |
|------------|---------------|----------|
| Soft       | Focal         | 0.741    |
| Soft       | FocalBCE      | 0.759    |
| Hard       | Focal         | -    |
| Hard       | FocalBCE      | -    |

## Initial Hyperparameters Experiments

1. **Mel Spectrogram Parameters**
   - Increasing n_fft from 1024 to 2048 improved our best score (CE + soft labels) to LB 0.777
   - **Note:** Many discussions highlighted that mel spectrogram parameters can dramatically affect performance

2. **Dataset Parameters**
   - Removing audio augmentations worsened performance despite community claims to the contrary (e.g., hard BCE dropped from 0.766 to 0.737 LB score)
   - Testing random wave segments (instead of centered ones) with our best configuration decreased LB score to 0.770
   - **Note** Maybe filtering human voice would impact random pieces approach as our center piece statistically does not include and human voice

3. **Modeling Parameters**
   - Increasing batch size to 256 (from 16) worsened performance to LB 0.764, potentially due to fewer optimization steps making training less stochastic (needed more epochs)
   - **Note** Further exploration of learning rate and batch size trade-offs is warranted


# Further experiments

## Inference Optimization

At this stage, inference optimization was required due to implementing larger backbones. The inference pipeline was enhanced with ONNX optimization. Dynamic and Static Quantization methods were tested but resulted in significant accuracy reduction while only increasing inference speed by approximately 1.5-2×. Simple conversion to the ONNX format resulted in a 2× speed boost (comparable to Static Quantization) without reducing score at all, unlike the significant decrease observed with Static Quantization, not to mention the even greater reduction with Dynamic Quantization.

Here are detailed experiments on a small 3.5M model:

| Method | LB Score | Inference Speed |
|--------|----------|----------------|
| Baseline | 0.777 | ~18 min |
| Dynamic Quantization (per-tensor) | 0.623 | ~12 min |
| Dynamic Quantization (per-channel) | 0.623 | ~12 min |
| Static Quantization (per-channel) | 0.716 | ~9 min |
| **ONNX conversion** | **0.777** | **~9 min** |

## ECA-NFNet-L0 + Spectogram Parameters

The next steps involved testing a larger backbone, ECA-NFNet-L0, which was widely used in previous years and reported to perform best among alternatives. This model has 21,838,636 parameters compared to the previous 3,594,812. We combined it with different spectrogram parameters (n_mels 256, f_min 50, f_max 14,000) and used a batch size of 32 with the same learning rate of 5e-4 and eta_min of 1e-6, but extended training to 20 epochs.

| Epoch | LB Score |
|-------|----------|
| 10 | 0.778 |
| 13 | 0.786 |
| 15 | 0.780 |
| 20 | **0.788** |

Despite the instability between epochs and the absence of strong correlation between leaderboard and local validation scores (13 is best on local val), this combined approach resulted in a performance boost. This motivated further experimentation with other hyperparameters for mel spectrograms and training configurations, as well as testing additional backbones.

## ECA-NFNet-L1 + Spectogram Parameters

We proceeded with ECA-NFNet-L1, which has 38,334,440 parameters. For this experiment, n_mels was reset to 128, while f_min and f_max were adjusted to 20 and 15,000 respectively for softer boundary values. Hop size was increased to test whether a model with less temporal noise would work better—a beneficial change as it also decreased the input size to the model, which is crucial when using larger architectures. Batch size was increased to 64 with the same learning rate, and training was extended to 37 epochs.

Additionally, this experiment used a mixup probability of 1, which has been frequently reported to boost performance.

| Epoch | LB Score |
|-------|----------|
| 10 | 0.782 |
| 15 | **0.808** |
| 20 | 0.789 |
| 25 | 0.796 |
| 30 | 0.798 |
| 35 | 0.801 |

## Current Results

  Numerous experiments were conducted testing a wide range of parameters as well as different loss functions, backbones, and data processing strategies:

![Experiment Results](assets/experiments.jpg)

These efforts resulted in the following leaderboard position as of April 27, 2025, out of 1,320 participants:

![Leaderboard Results](assets/lb_27-04-2025.png)

## Future Work

1. Continue experiments with mel spectrogram parameters
2. Further explore mixup probability and alpha parameters
3. Make up some post processing techniques which smooth predictons on neighboring chunks
3. Test alternative backbones—try ResNets and other architectures along with different pooling strategies and features from various layers
4. Continue exploring batch size and learning rate trade-offs; try different warmup strategies
5. Optimize class weights/sampling weights temperature (and potentially focal loss gamma)
6. Implement training from checkpoints to enable large-scale training for many epochs
7. Filter human voices from audio recordings and revisit random audio segment selection

**Major approaches necessary for silver/gold performance:**

1. Pre-train on data containing minority classes from Xeno-canto and other resources
2. Use pseudo-labeling on available train soundscapes (which are reportedly recorded in the same locations as the hidden test set) and fine-tune meta models on these recordings