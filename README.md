# Deep Learning Mini-Project
Submitted by:

| #       |              Name |             Id |             email |
|---------|-------------------|----------------|------------------ |
|Student 1|  Oz Diamond | 315099077 | oz.diamond@campus.technion.ac.il |
|Student 2| Yakov Mishayev| 309645737 | yakov-m@campus.technion.ac.il |

## Project Structure

- `src/mini_project/main.py`: Implementation of self-supervised autoencoder (1.2.1) and classification-guided training (1.2.2)
- `src/mini_project/main_contrastive.py`: Implementation of contrastive learning approach (1.2.3)
- `src/mini_project/models.py`: Model architectures for different training approaches
- `src/mini_project/utils.py`: Utility functions for visualization and evaluation
- `run.sh`: Batch script to run all experiments

## Installation

Set up the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate aes
```

## Usage
### Option 1: Run All Experiments with the Batch Script

The `run.sh` script will run all the experiments with predefined parameters:

```bash
# Run all experiments with full epochs
bash run.sh

# Run a quick test with minimal epochs to check functionality
bash run.sh --test
```

results will be save in `./results`
### Option 2: Individual Experiments

#### 1.2.1 Self-Supervised Autoencoder (AE)

Train an autoencoder in a self-supervised manner and then train a classifier on the frozen encoder:

```bash
# For MNIST
python src/mini_project/main.py --save-path ./results/MNIST/AE_CL --mnist --mode self_supervised --epochs-ae 30 --epochs-clf 30

# For CIFAR-10
python src/mini_project/main.py --save-path ./results/CIFAR/AE_CL --mode self_supervised --epochs-ae 50 --epochs-clf 180
```

#### 1.2.2 Classification-Guided Training (CG)

Train an encoder and classifier jointly:

```bash
# For MNIST
python src/mini_project/main.py --save-path ./results/MNIST/CG --mnist --mode classification_guided --epochs-cg 30

# For CIFAR-10
python src/mini_project/main.py --save-path ./results/CIFAR/CG --mode classification_guided --epochs-cg 60
```

#### 1.2.3 Contrastive Learning

Train an encoder using contrastive learning and then train a classifier on the frozen encoder:

```bash
# For MNIST
python src/mini_project/main_contrastive.py --save-path ./results/MNIST/CONTRASTIVE --mnist --epochs-contrastive 30 --epochs-clf 30

# For CIFAR-10
python src/mini_project/main_contrastive.py --save-path ./results/CIFAR/CONTRASTIVE --epochs-contrastive 50 --epochs-clf 180
```



## Command Line Arguments

### Common Arguments

- `--seed`: Random seed for reproducibility (default: 0)
- `--data-path`: Path to dataset directory (default: ./data)
- `--save-path`: Path to save trained models and results
- `--batch-size`: Batch size for training
- `--latent-dim`: Dimension of latent space (default: 128)
- `--lr`: Learning rate (default: 1e-3)
- `--mnist`: Use MNIST dataset (if not specified, CIFAR-10 is used)
- `--device`: Device to use (default: 'cuda' if available, otherwise 'cpu')
- `--optimizer`: Optimizer type (default: 'adamw', options: 'adam', 'adamw', 'sgd', 'rmsprop')

### main.py Specific Arguments

- `--mode`: Training mode ('self_supervised' for 1.2.1 or 'classification_guided' for 1.2.2)
- `--epochs-ae`: Number of epochs for autoencoder pretraining (default: 35)
- `--epochs-clf`: Number of epochs for classifier training (default: 15)
- `--epochs-cg`: Number of epochs for classification guided training (default: 20)

### main_contrastive.py Specific Arguments

- `--epochs-contrastive`: Number of epochs for contrastive pretraining (default: 60)
- `--epochs-clf`: Number of epochs for classifier training after contrastive pretraining (default: 180)

## Output

Each experiment will create:
- Training and validation loss curves
- t-SNE visualizations of the latent space
- Reconstruction visualizations (for self-supervised and contrastive approaches)
- Latent space interpolations (for self-supervised approach)
- Test metrics in a text file

Results are saved in the directory specified by `--save-path`.
