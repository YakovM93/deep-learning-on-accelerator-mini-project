# Deep Learning Mini-Project
Submitted by:

| #       |              Name |             Id |             email |
|---------|-------------------|----------------|------------------ |
|Student 1|  Oz Diamond | 315099077 | oz.diamond@campus.technion.ac.il |
|Student 2| Yakov Mishayev| 309645737 | yakov-m@campus.technion.ac.il |


## Installation

Set up the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate mini_project
```

## Usage
From the root directory of the project, run the following commands:

### Basic Usage


To run all experiments (all modes on both datasets with the same run parameters (batch size, epoch numbers) used in the report:

```bash
python src/mini_project/main.py --mode all
```

For a quick test with reduced epochs:

```bash
python src/mini_project/main.py --mode all-test
```
Run a single experiment:

```bash
# Self-supervised learning on MNIST
python src/mini_project/main.py --mode self_supervised --mnist

# Classification-guided learning on CIFAR-10
python src/mini_project/main.py --mode classification_guided

# Contrastive learning on MNIST
python src/mini_project/main.py --mode contrastive --mnist
```




### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Training paradigm (`self_supervised`, `classification_guided`, `contrastive`, `all`, `all-test`) | `self_supervised` |
| `--mnist` | Use MNIST dataset (otherwise CIFAR-10) | `False` |
| `--data-path` | Path to dataset | `./data` |
| `--save-path` | Path to save models and results | `./trained-models/` |
| `--batch-size` | Batch size | `32` |
| `--latent-dim` | Latent dimension size | `128` |
| `--lr` | Learning rate | `0.001` |
| `--epochs-ae` | Epochs for autoencoder training | `35` |
| `--epochs-clf` | Epochs for classifier training | `15` |
| `--epochs-cg` | Epochs for classification-guided training | `20` |
| `--epochs-contrastive` | Epochs for contrastive learning | `60` |
| `--optimizer` | Optimizer (`adam`, `adamw`, `sgd`, `rmsprop`) | `adamw` |
| `--device` | Computing device | `cuda` if available, else `cpu` |
| `--seed` | Random seed for reproducibility | `0` |

## Training Paradigms

### 1. Self-Supervised Learning (AE + Classifier)

This two-phase approach:
1. First trains an autoencoder in an unsupervised manner to learn representations
2. Then freezes the encoder and trains a classifier on top of it

```bash
python src/mini_project/main.py --mode self_supervised --mnist
```

### 2. Classification-Guided Learning

End-to-end supervised training where representations are learned directly from the classification task:

```bash
python src/mini_project/main.py --mode classification_guided
```

### 3. Contrastive Learning

Learns representations by contrasting positive pairs against negative pairs, followed by classifier training:

```bash
python src/mini_project/main.py --mode contrastive --mnist
```

## Output and Results

The script produces:

- **Trained Models**: Saved encoder, decoder, and classifier models
- **Loss Curves**: Training and validation loss plots
- **Accuracy Plots**: Classification accuracy progression
- **t-SNE Visualizations**: For both image space and latent space
- **Reconstructions**: Visual comparison of original vs. reconstructed images
- **Interpolations**: Latent space interpolation between samples
- **Results File**: Text file with final performance metrics

Results are organized as follows:

```
results/
├── MNIST/
│   ├── AE_CL/            (Self-Supervised)
│   ├── CG/               (Classification-Guided)
│   └── CONTRASTIVE/      (Contrastive Learning)
└── CIFAR/
    ├── AE_CL/            (Self-Supervised)
    ├── CG/               (Classification-Guided)
    └── CONTRASTIVE/      (Contrastive Learning)
```
