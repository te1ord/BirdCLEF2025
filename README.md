# BirdCLEF 2025 - Kaggle Competition

This repository contains our experimental framework for the **BirdCLEF 2025** Kaggle competition. The goal of this project was to explore various modeling strategies and audio preprocessing techniques to detect bird species from audio recordings.

Our implementation draws inspiration from previous solutions like [VSydorskyy/BirdCLEF_2023_1st_place](https://github.com/VSydorskyy/BirdCLEF_2023_1st_place) and lecture materials from [Module 3, Lecture 1 of IASA&UCU Audio Processing Course](https://github.com/VSydorskyy/ucu_audio_processing_course/blob/main/Module_3/Lecture_1/Signal_Classification.ipynb).

## Solution Overview

### Exploratory Data Analysis (EDA)
We performed detailed metadata analysis in [eda.ipynb](https://github.com/te1ord/BirdCLEF2025/blob/main/notebooks/eda.ipynb) and [dataset_test.ipynb](https://github.com/te1ord/BirdCLEF2025/blob/main/notebooks/dataset_test.ipynb) to understand data distribution and gain insights about potential validation splits.

**Key Discoveries:**
- Highly imbalanced data distribution
- RocAUC target metric with macro-averaging, which weights all classes equally
- Potential data leakage with identical recordings appearing in multiple collections
- Potential data leakage from multiple audio splits from single large recordings by the same author
- Potential use of Rating column for sample weighting
- Potential use of secondary labels for multilabel optimization
- Presence of [human voices](https://www.kaggle.com/competitions/birdclef-2025/discussion/568886) in the CSA collection (affecting many minority classes)

### Data Preprocessing
Considering our findings, we implemented a stratified split to correctly evaluate performance on minority classes while **removing near-duplicate samples** (e.g., similar recordings from different collections) to avoid potential leakage.

The split logic is documented in [cv_split.ipynb](https://github.com/te1ord/BirdCLEF2025/blob/main/notebooks/cv_split.ipynb).

**TODO:** Explore stratified split by target grouped by author to further minimize chances of leakage.

### Framework Architecture
The framework design facilitates easy iteration to test multiple hypotheses with robust tracking and comparison, requiring minimal changes to the source code.

We built:
- A modular training pipeline that encapsulates each functional element within its module
- A configuration manager with Hydra for flexible and clean experimentation
- Weights & Biases (wandb) logger that tracks resolved Hydra config and model checkpoints
- An "MLOps" pipeline for inference on Kaggle

### Modeling Approach

Our multilabel classification pipeline includes these key components:

- Custom dataset class with mixup augmentation operating on raw audio level, forming training examples with corresponding soft/hard labels
- Dataloader sampler weights to increase sampling frequency for minority classes
- Custom model extracting mel spectrograms (on GPU) and passing them to an EfficientNet backbone with linear head
- Optimization using AdamW with learning rate 5e-4 and weight decay 1e-5, combined with cosine learning rate scheduler reducing to minimum 1e-6

### Experimental Setup

#### Evaluation Strategy

To ensure robust evaluation, fold 0 was manually configured to include all classes. Following common practice for Kaggle competitions, we tracked only the target RocAUC metric during experiments.

After several iterations, it became clear that local validation was ineffective. Community discussions revealed that all participants faced the same issue: validation RocAUC scores of ~0.99 resulting in leaderboard (LB) scores ranging from 0.7 to 0.9, indicating a significant domain shift in the hidden test set.

Experiments confirmed no consistent correlation between local validation and public leaderboard scores, with relationships ranging from positive to negative across different hyperparameters.

This inconsistency led us to use the public leaderboard as our validation method, submitting checkpoints from the same epochs (10) to evaluate each experiment and determine the right direction.

**Note:** Setting up alternative metrics for class imbalance (like F1/PR-AUC) may help estimate classifier performance for different error types. However, without error analysis, this wouldn't help optimize the target RocAUC metric, which doesn't distinguish between different error types.

#### Loss Function Experiments

We prioritized experimenting with loss functions in combination with soft/hard labels.
Despite BCE loss being common for multilabel classification, its appropriateness was questioned when using mixup augmentation.

We considered two scenarios:

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

**Soft Labels:**
- BCE: LB 0.759
- CE: LB 0.775

**Hard Labels:**
- BCE: LB 0.766
- CE: LB 0.774

Our hypothesis was generally confirmed, though we unexpectedly found CE with hard labels performing well at 0.774. We hypothesize this occurred because larger gradients (due to the loss function's larger scale: 1 * log p + 1 * log p is larger than 0.3 * log p + 0.7 * log p) produce more aggressive updates. After softmax, this nearly zeroes out uncertain classes, resulting in fewer false positives and improved LB RocAUC scores. This suggests that ensembling models optimized with CE + hard labels with other less strict classifiers may increase robustness.

To better optimize for minority classes, we tested two approaches beyond dataloader weights:

1. **Class Weights**
   - Class weights for our best model decreased performance from 0.777 to 0.766

2. **Focal Loss**
   - Both Focal Loss and FocalBCE loss were tested with soft labels, yielding poorer results compared to previous experiments

**Soft Labels:**
- Focal: LB 0.741
- FocalBCE: LB 0.759

#### Additional Experiments

1. **Mel Spectrogram Parameters**
   - Increasing n_fft from 1024 to 2048 improved our best score (CE + soft labels) to LB 0.777
   - **Note:** Many discussions highlighted that mel spectrogram parameters can dramatically affect performance

2. **Dataset Parameters**
   - Removing audio augmentations worsened performance despite community claims to the contrary (e.g., hard BCE dropped from 0.766 to 0.737 LB score)
   - Testing random wave segments (instead of centered ones) with our best configuration decreased LB score to 0.770

3. **Modeling Parameters**
   - Increasing batch size to 256 (from 16) worsened performance to LB 0.764, potentially due to fewer optimization steps making training less stochastic
   - Further exploration of learning rate and batch size trade-offs is warranted

### Current Results
![Experiment Results](assets/experiments.jpg)

## Future Work

1. Fine-tune mel spectrogram parameters (multiple discussions indicate significant potential improvements)
2. Experiment with mixup probability and alpha parameters (some competitors report better results with probability = 1)
3. Test alternative backbones - we used tf_efficientnet_b0.in1k, but many competitors use eca_nfnet_l0 along with different pooling strategies and features from various layers
4. Explore batch size and learning rate trade-offs
5. Optimize class weights/sampling weights temperature and focal loss gamma
6. Implement training from checkpoints to enable large-scale training on the most promising approaches
7. Filter human voices from audio recordings