# PR3DICTR

**Platform for Research in 3D Image Classification and sTandardised tRaining**

PR3DICTR is an open-access, modular deep learning framework for 3D medical image classification, built on [PyTorch](https://pytorch.org/) and [MONAI](https://monai.io/). It provides a standardized, config-driven pipeline for developing, training, and evaluating prediction models from volumetric medical imaging data — with as little as two lines of code.

> Developed by the AI in Radiotherapy group at the University Medical Center Groningen (UMCG).

---

## Overview

Three-dimensional medical imaging is increasingly central to clinical decision-making, yet building deep learning models from 3D data remains technically demanding and methodologically inconsistent across research groups. PR3DICTR addresses this by combining:

- **Simplicity** — config-file-driven workflows requiring minimal code
- **Modularity** — interchangeable components for architectures, data loading, training, and evaluation
- **Standardization** — consistent pipelines that promote reproducibility across experiments and teams

PR3DICTR supports any binary or event-based (survival/time-to-event) 3D classification task, and can integrate imaging and tabular (clinical) data in a unified model.

---

## Repository Structure

```
PR3DICTR/
├── src/                        # Core framework source code
├── main/                       # Main training and evaluation scripts
├── notebooks/                  # Example Jupyter notebooks
├── documentation/              # Extended documentation
├── env/                        # Environment and dependency files
├── publications/               # Code for projects published using PR3DICTR
│   └── Project_A/
├── Basic_xerostomia_model.py   # Minimal working example
└── __init__.py
```

---

## Features

- **Config-driven** — define your entire experiment (preprocessing, architecture, training, evaluation) in a single YAML config file; only two lines of code needed to run
- **Multi-modal imaging** — CT, PET, MRI, radiation dose distributions, segmentation masks, and any other volumetric input
- **Multiple architectures** — CNN, ResNet (10/18/34/50/101/152/200), DenseNet (121/169/201/264), EfficientNetV2, ConvNeXt, ViT, TransRP (hybrid CNN-ViT), MLP (tabular-only)
- **Tabular data fusion** — integrate clinical features into image models via configurable linear layers or ViT attention
- **Automated hyperparameter optimization** — [Optuna](https://optuna.org/) integration for efficient Bayesian search
- **K-fold cross-validation** — stratified, reproducible splits with configurable fold count
- **Rich augmentation** — random cropping, flipping, affine transforms, rotation, Gaussian noise, MixUp
- **Flexible data loading** — Standard, Cache, SmartCache, and Persistent dataset modes for different RAM/speed trade-offs
- **Comprehensive evaluation** — AUC, accuracy, C-index, F1, precision, recall, Brier score, and calibration metrics (ACE, ECE, MCE), with plots (ROC, calibration, confusion matrix, Kaplan-Meier)
- **Experiment tracking** — native [Weights & Biases](https://wandb.ai/) integration
- **Ensemble evaluation** — post-hoc test set evaluation per fold and as an ensemble

---

## Getting Started

### Prerequisites

Your data must be prepared in the following format before using PR3DICTR:

**Volumetric image data** stored as NumPy (`.npy`) files, with all volumes sharing the same dimensions, organized by patient ID:

```
data/
├── PatientID001/
│   ├── CT.npy
│   └── PET.npy
├── PatientID002/
│   ├── CT.npy
│   └── PET.npy
...
```

**A CSV file** with at minimum three columns:

| Column | Description |
|---|---|
| `PatientID` | Links imaging data to clinical records |
| `Split` | `train_val` or `test` |
| `<label>` | One or more outcome columns; use `-1` for missing |

Additional tabular features (e.g., age, sex) can be included as extra columns.


### Basic Usage

```python
import os

from src.config_presets.tools.get_config import get_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

from src.experiments.experimentHandler import experimentHandler
from src.evaluation.validate_on_test_set import validate_models_on_test_set


if __name__ == '__main__':

    # Setup
    log_level = parse_args()
    setup_logging(log_level)

    # Load the config
    config = get_config('Xerostomia_model')

    # Disable randomness
    set_random_seed(config['general']['seed'])
    

    # # MAIN: DL running class (with optional hyperparameter optimization)
    # this line is used to run the k-fold cross-validation
    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)


    # TEST THE ENSEMBLE OF MODELS ON THE TEST SET
    
    # # # # run the models on the test set
    trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    validate_models_on_test_set(config, trial_dir)

```

For a minimal working example, see [`Basic_xerostomia_model.py`](Basic_xerostomia_model.py).

---

## Configuration

PR3DICTR uses a **Base Config** that provides sensible defaults for all parameters. Your project-specific config only needs to override what you want to change. The config is passed through the pipeline as a dictionary and controls every aspect of model definition, training, and evaluation.

Key configurable options:

| Category | Options |
|---|---|
| **Image modalities** | CT, PET, MRI, radiation dose, segmentations, any other volumetric input |
| **Input preprocessing** | Value clipping & normalization, value mapping (segmentations), cropping |
| **Augmentation** | Random crop, flip, affine, rotation, Gaussian noise, MixUp |
| **Dataset mode** | Standard, Cache, SmartCache, Persistent |
| **Model architecture** | CNN, ResNet, DenseNet, EfficientNetV2, ConvNeXt, ViT, TransRP, MLP |
| **Loss functions** | BCE, Focal, Hill, ASL, NLL (survival) |
| **Optimizers** | Adam, AdamW, AdaBound, SGD |
| **LR schedulers** | Cosine, Step, Plateau, None (fixed) |
| **Evaluation metrics** | AUC, Accuracy, C-index, F1, Precision, Recall, Brier, ACE, ECE, MCE |
| **Visualizations** | Calibration plot, reliability plot, confusion matrix, ROC curve, Kaplan-Meier |


---

## Example Notebooks

The [`notebooks/01_LearningExamples`](notebooks/01_LearningExamples) directory contains examples using the publicly available [NSCLC-Radiomics dataset](https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI) (Aerts et al., 2014) for sex classification from chest CT:

- **`TCIA_Data_Pre-processing.ipynb`** — data preparation walkthrough
- **`01_NSCLC_Example.ipynb`** — full PR3DICTR training run using a ResNet-10



---

## Architecture

Models in PR3DICTR consist of two main components:

**Image Encoder** — extracts features from 3D volumetric inputs. Multiple modalities (e.g., CT + PET) are stacked along the channel dimension, yielding input tensors of shape `[B, C, H, W, D]`.

**Output Module** — integrates image features with optional tabular data through fully connected layers. Supports multi-label outputs, where each label gets its own independent output head. When no imaging is used, an MLP operates on tabular features alone.

---

## Publications

Results and code from studies using PR3DICTR can be found in the [`publications/`](publications/) directory. Currently includes:

- **Head & Neck multi-toxicity** — multi-endpoint NTCP modelling for radiotherapy

---

## Roadmap

- [ ] Multi-class classification support
- [ ] Uncertainty quantification
- [ ] Attention map visualizations for model interpretability
- [ ] Semi-automatic model card generation
- [ ] GUI for config setup

---

## Citation

If you use PR3DICTR in your research, please cite: [arxiv](https://arxiv.org/pdf/2604.03203)

```
MacRae, D. C., van der Hoek, L., van der Wal, R., de Vette, S. P., Neh, H., Ma, B., ... & van Dijk, L. V. (2026).
PR3DICTR: A modular AI framework for medical 3D image-based detection and outcome prediction. arXiv preprint arXiv:2604.03203.
```

---

## License

[License details to be added]

---

## Acknowledgements

Developed by the AI in Radiotherapy group at the **University Medical Center Groningen (UMCG)**. PR3DICTR has been used in production for developing deep learning-based normal tissue complication probability (NTCP) models for head and neck radiotherapy, including dysphagia, taste, radiation pneumonitis, and multi-toxicity prediction.
