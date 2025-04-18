# Fine-Tuning ResNet50 on iNaturalist — Part B

This project is **Part B** of the DA6401 Assignment 2, where we fine-tune a pretrained ResNet50 on a subset of the iNaturalist dataset. The objective is to compare its performance against a CNN trained from scratch (Part A) and explore different fine-tuning strategies.

---

## Setup

### Clone the Repository

```bash
git clone https://github.com/ak4off/da6401_assignment2.git
cd da6401_assignment2/partB
```
### Install Requirements

You can install the required packages individually:

```bash
pip install torch torchvision
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install wandb
pip install tqdm
```
## Code Structure

```
partB/
├── the_trainer.py         # Main training loop using ResNet50
├── the_evaluator.py       # Evaluates the saved model on test data
├── the_data_loader.py     # Prepares DataLoader with augmentation and transforms
├── the__sweep.py          # Launches wandb sweep for hyperparameter tuning
├── the_utils.py           # Helper functions (accuracy calculation, config parser, etc.)
├── the_visualize.py       # (Optional) Visualizations - confusion matrix, prediction grids
├── models/
│   └── resnet_finetune.py # Model definition and ResNet50 fine-tuning logic
├── config.yaml            # Hyperparameters and wandb sweep config
└── README.md              # This file
```

---

## Model: ResNet50

- Pretrained on ImageNet.
- Final classification head adapted to **10 classes**.
- Supports three freeze modes:
  - `0`: Feature extractor (freeze all layers)
  - `1`: Freeze backbone, unfreeze final block
  - `2`: Full fine-tuning (all layers trainable)
## Dataset

- Dataset used: `iNaturalist_12K`

## Dataset Structure

```
data/
├── train/
└── val/     # Used as test set
```

- 20% of the `train` data is used as a validation set.

To link your local dataset:

```bash
ln -s <path-to-inaturalist_12k> data
```
## Usage

### Training

Train the ResNet50 model:

```bash
python the_trainer.py --data_dir data/inaturalist_12K
```
Train with Weights & Biases logging:
```
python the_trainer.py --data_dir data/inaturalist_12K --wandb
```
Run a Hyperparameter Sweep
```
python the__sweep.py
```
## Logging and Results

- Train and validation accuracy/loss are logged per epoch.
- Final test accuracy is computed using the `val/` directory.
- All metrics are logged to **Weights & Biases**, including:
  - Accuracy and loss curves
  - Parallel coordinate plots
  - Parameter importance
## License

This project is licensed under the **MIT License**.

