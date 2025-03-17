
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset
import argparse
import random
import os
import datetime
from models.resnet_contrastive import ContrastiveResNetBasedLatentModel
from utils import (
    plot_tsne,
    save_checkpoint,
    load_checkpoint
)

NUM_CLASSES = 10

class ContrastivePairDataset(Dataset):
    """
    Dataset wrapper that returns pairs of augmented views of the same image for contrastive learning.
    """
    def __init__(self, dataset, transform1, transform2=None):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2 if transform2 is not None else transform1

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # Create two different augmented views of the same image
        view1 = self.transform1(img)
        view2 = self.transform2(img)
        return view1, view2, label

    def __len__(self):
        return len(self.dataset)


def freeze_seeds(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="./data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False, help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--val-split', default=0.1, type=float, help='Fraction of training data to use for validation')
    parser.add_argument('--override-checkpoint', action='store_true', default=False, help='Override checkpoint and start training from scratch')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from checkpoint')
    parser.add_argument('--checkpoint-dir', default='checkpoints_contrastive', type=str, help='Directory to save checkpoints')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature parameter for contrastive loss')
    parser.add_argument('--proj-head-type', default='mlp', type=str, choices=['linear', 'mlp'], help='Type of projection head for contrastive learning')
    return parser.parse_args()


def nt_xent_loss(z_i, z_j, temperature=0.5, device='cuda'):
    """
    Compute NT-Xent loss (SimCLR contrastive loss)

    Args:
        z_i: First batch of encoded samples (N, latent_dim)
        z_j: Second batch of encoded samples (N, latent_dim)
        temperature: Temperature scaling parameter
        device: Device to use

    Returns:
        NT-Xent loss value
    """
    batch_size = z_i.shape[0]
    # Concatenate all embeddings, creating a 2N x latent_dim tensor
    representations = torch.cat([z_i, z_j], dim=0)

    # Compute similarity matrix - dot product between all combinations of embeddings
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                           representations.unsqueeze(0),
                                           dim=2)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    similarity_matrix = similarity_matrix.masked_fill_(mask, -float('inf'))

    # The positives are in the diagonal of the similarity matrix
    # between z_i and z_j (i.e., the pair of augmented versions of the same sample)
    pos_sim = torch.cat([
        similarity_matrix[i, batch_size+i].reshape(1)
        for i in range(batch_size)
    ] + [
        similarity_matrix[batch_size+i, i].reshape(1)
        for i in range(batch_size)
    ])

    # Scale by temperature
    pos_sim = pos_sim / temperature

    # Compute the N+N-1 negatives for each anchor
    negatives = similarity_matrix[~mask].reshape(2*batch_size, -1)

    # The logits are concatenated positives and negatives
    logits = torch.cat([pos_sim.reshape(-1, 1), negatives / temperature], dim=1)

    # The positive is always at position 0, so use that as the target
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


def train_contrastive(model, train_loader, val_loader, device, epochs=20, lr=1e-3,
                      temperature=0.5, checkpoint_dir='checkpoints_contrastive',
                      resume=False, checkpoint_freq=5):
    """
    Train the ResNet-based model using contrastive learning with checkpointing.

    Args:
        model: ContrastiveResNetBasedLatentModel
        train_loader: DataLoader providing pairs of augmented views
        val_loader: DataLoader for validation
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        lr: Learning rate
        temperature: Temperature parameter for contrastive loss
        checkpoint_dir: Directory to save checkpoints
        resume: Whether to resume from checkpoint
        checkpoint_freq: Frequency (in epochs) to save checkpoints

    Returns:
        Trained model and training history
    """
    model = model.to(device)

    # Set model to only train encoder (including projection head)
    optimizer = optim.Adam(model.encoder.parameters(), lr=lr)

    # Initialize history lists
    train_losses = []
    val_losses = []

    # Checkpoint filename
    checkpoint_file = f"{checkpoint_dir}/contrastive_checkpoint.pth"

    # Resume from checkpoint if requested
    start_epoch = 0
    if resume and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        print("No checkpoint found or resume not requested, starting from scratch")

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0

        for views1, views2, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            views1, views2 = views1.to(device), views2.to(device)

            # Get contrastive features
            z_i = model(views1, mode='contrastive')
            z_j = model(views2, mode='contrastive')

            # Compute contrastive loss
            loss = nt_xent_loss(z_i, z_j, temperature=temperature, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for views1, views2, _ in val_loader:
                views1, views2 = views1.to(device), views2.to(device)

                # Get contrastive features
                z_i = model(views1, mode='contrastive')
                z_j = model(views2, mode='contrastive')

                # Compute contrastive loss
                loss = nt_xent_loss(z_i, z_j, temperature=temperature, device=device)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}")

    # Create and save the loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Contrastive Learning Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('NT-Xent Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/contrastive/contrastive_loss.png')
    plt.close()

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def train_classifier_with_frozen_encoder(model, train_loader, val_loader, device, epochs=20, lr=1e-3,
                                       checkpoint_dir='checkpoints_contrastive', resume=False, checkpoint_freq=5):
    """
    Train the classifier using the frozen contrastive encoder with checkpointing.

    Args:
        model: ContrastiveResNetBasedLatentModel with pre-trained encoder
        train_loader: DataLoader for training data (standard, not contrastive pairs)
        val_loader: DataLoader for validation data
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
        resume: Whether to resume from checkpoint
        checkpoint_freq: Frequency (in epochs) to save checkpoints

    Returns:
        Trained model and training history
    """
    model = model.to(device)

    # Freeze encoder, train only the classifier
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize history lists
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Checkpoint filename
    checkpoint_file = f"{checkpoint_dir}/classifier_checkpoint.pth"

    # Resume from checkpoint if requested
    start_epoch = 0
    if resume and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_accuracies = checkpoint.get('train_accuracies', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        print("No checkpoint found or resume not requested, starting from scratch")

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, targets = data.to(device), targets.to(device)

            # Forward pass (classifier mode)
            outputs = model(data, mode='classifier')

            # Calculate loss and backpropagate
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * (correct / total)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)

                # Forward pass (classifier mode)
                outputs = model(data, mode='classifier')

                # Calculate loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.2f}%")

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}")

    # Create and save the plots
    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Classifier Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/contrastive/classifier_training_loss.png')
    plt.close()

    # Plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Classifier Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/contrastive/classifier_training_accuracy.png')
    plt.close()

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }


def evaluate(model, test_loader, device):
    """
    Evaluate the model on the test set.

    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cuda' or 'cpu')

    Returns:
        Test accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            # Forward pass (classifier mode)
            outputs = model(data, mode='classifier')

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy


def create_report(args, contrastive_history, clf_history, test_accuracy):
    """
    Create a comprehensive report of the training process and results.

    Args:
        args: Command-line arguments
        contrastive_history: History of contrastive training
        clf_history: History of classifier training
        test_accuracy: Test accuracy

    Returns:
        Report as a string
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = [
        "=" * 80,
        f"CONTRASTIVE LEARNING MODEL TRAINING REPORT",
        f"Generated at: {now}",
        "=" * 80,
        "",
        "MODEL ARCHITECTURE",
        "-" * 50,
        f"Model type: ContrastiveResNetBasedLatentModel",
        f"Projection head type: {args.proj_head_type}",
        f"Latent dimension: {args.latent_dim}",
        f"Temperature parameter: {args.temperature}",
        "",
        "TRAINING CONFIGURATION",
        "-" * 50,
        f"Dataset: {'MNIST' if args.mnist else 'CIFAR10'}",
        f"Batch size: {args.batch_size}",
        f"Epochs (per phase): {args.epochs}",
        f"Learning rate: {args.lr}",
        f"Device: {args.device}",
        f"Random seed: {args.seed}",
        "",
        "CONTRASTIVE LEARNING PHASE RESULTS",
        "-" * 50,
        f"Initial contrastive loss: {contrastive_history['train_losses'][0]:.6f}",
        f"Final contrastive loss: {contrastive_history['train_losses'][-1]:.6f}",
        f"Initial validation loss: {contrastive_history['val_losses'][0]:.6f}",
        f"Final validation loss: {contrastive_history['val_losses'][-1]:.6f}",
        f"Improvement: {(1 - contrastive_history['val_losses'][-1] / contrastive_history['val_losses'][0]) * 100:.2f}%",
        "",
        "CLASSIFIER TRAINING PHASE RESULTS",
        "-" * 50,
        f"Initial training accuracy: {clf_history['train_accuracies'][0]:.2f}%",
        f"Final training accuracy: {clf_history['train_accuracies'][-1]:.2f}%",
        f"Initial validation accuracy: {clf_history['val_accuracies'][0]:.2f}%",
        f"Final validation accuracy: {clf_history['val_accuracies'][-1]:.2f}%",
        f"Improvement: {clf_history['val_accuracies'][-1] - clf_history['val_accuracies'][0]:.2f} percentage points",
        "",
        "FINAL EVALUATION",
        "-" * 50,
        f"Test accuracy: {test_accuracy:.2f}%",
        "",
        "FILES AND LOCATIONS",
        "-" * 50,
        f"Checkpoint directory: {args.checkpoint_dir}",
        f"Results directory: results/contrastive/",
        f"Plots saved: contrastive_loss.png, classifier_training_loss.png, classifier_training_accuracy.png, tsne_visualization.png",
        f"Model saved: results/contrastive/contrastive_full_model.pth",
        "",
        "=" * 80
    ]

    return "\n".join(report)


if __name__ == "__main__":
    # Get arguments
    args = get_args()
    freeze_seeds(args.seed)

    # Create output directories for results
    os.makedirs('results/contrastive', exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set appropriate transforms based on the dataset
    if args.mnist:
        # Base transform for loading data
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # Repeat grayscale to 3 channels
        ])
        # Transform for contrastive learning requires data augmentation
        contrastive_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.5),
            transforms.RandomGrayscale(p=0.2)
        ])
        in_channels = 3  # Now MNIST will be 3 channels
    else:
        # Base transform for loading data
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # Transform for contrastive learning requires data augmentation
        contrastive_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.5),
            transforms.RandomGrayscale(p=0.2)
        ])
        in_channels = 3  # CIFAR10 has 3 channels

    # Load datasets - we need two versions, one for contrastive pairs and one for regular classification
    if args.mnist:
        # For contrastive learning, we need two different augmented views
        train_base = datasets.MNIST(root=args.data_path, train=True, download=True, transform=None)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=base_transform)
    else:
        # For contrastive learning, we need two different augmented views
        train_base = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=base_transform)

    # Split training data into train and validation sets
    val_size = int(len(train_base) * args.val_split)
    train_size = len(train_base) - val_size
    train_subset, val_subset = random_split(train_base, [train_size, val_size])

    # Create contrastive pairs datasets for training and validation
    train_contrastive = ContrastivePairDataset(train_subset, contrastive_transform)
    val_contrastive = ContrastivePairDataset(val_subset, contrastive_transform)

    # Also create regular datasets for classifier training/evaluation
    train_regular = datasets.MNIST(root=args.data_path, train=True, download=True, transform=base_transform) if args.mnist else \
                   datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=base_transform)
    train_regular_subset, val_regular_subset = random_split(train_regular, [train_size, val_size])

    # Create data loaders
    train_contrastive_loader = DataLoader(train_contrastive, batch_size=args.batch_size, shuffle=True)
    val_contrastive_loader = DataLoader(val_contrastive, batch_size=args.batch_size, shuffle=False)

    train_regular_loader = DataLoader(train_regular_subset, batch_size=args.batch_size, shuffle=True)
    val_regular_loader = DataLoader(val_regular_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*50}")
    print(f"Running Contrastive Learning on {'MNIST' if args.mnist else 'CIFAR10'}")
    print(f"{'='*50}\n")

    # Initialize ContrastiveResNetBasedLatentModel
    print(f"Creating ContrastiveResNetBasedLatentModel with latent dimension {args.latent_dim}")
    model = ContrastiveResNetBasedLatentModel(
        in_channels=in_channels,
        latent_dim=args.latent_dim,
        num_classes=NUM_CLASSES,
        proj_head_type=args.proj_head_type
    )

    # Define checkpoint filenames
    contrastive_checkpoint = f"{args.checkpoint_dir}/contrastive_checkpoint.pth"
    classifier_checkpoint = f"{args.checkpoint_dir}/classifier_checkpoint.pth"
    final_model_path = 'results/contrastive/contrastive_full_model.pth'

    # Check if we should override existing checkpoints
    if args.override_checkpoint:
        if os.path.exists(contrastive_checkpoint):
            os.remove(contrastive_checkpoint)
        if os.path.exists(classifier_checkpoint):
            os.remove(classifier_checkpoint)
        print("Existing checkpoints removed, starting training from scratch.")
        args.resume = False

    # Phase 1: Contrastive Learning
    print("\nPhase 1: Contrastive Learning")
    print("-" * 50)

    # If there's a saved checkpoint and we want to resume, load it
    resume_phase1 = args.resume and os.path.exists(contrastive_checkpoint)
    if resume_phase1:
        print(f"Resuming contrastive training from checkpoint: {contrastive_checkpoint}")

    model, contrastive_history = train_contrastive(
        model, train_contrastive_loader, val_contrastive_loader,
        args.device, epochs=args.epochs, lr=args.lr,
        temperature=args.temperature,
        checkpoint_dir=args.checkpoint_dir, resume=resume_phase1
    )

    # Save the trained model after contrastive learning phase
    torch.save(model.state_dict(), 'results/contrastive/contrastive_model.pth')

    # Phase 2: Train classifier with frozen encoder
    print("\nPhase 2: Supervised classifier training with frozen contrastive encoder")
    print("-" * 50)

    # If there's a saved checkpoint and we want to resume, load it
    resume_phase2 = args.resume and os.path.exists(classifier_checkpoint)
    if resume_phase2:
        print(f"Resuming classifier training from checkpoint: {classifier_checkpoint}")

    model, clf_history = train_classifier_with_frozen_encoder(
        model, train_regular_loader, val_regular_loader,
        args.device, epochs=args.epochs, lr=args.lr,
        checkpoint_dir=args.checkpoint_dir, resume=resume_phase2
    )

    # Save the trained model after classifier phase
    torch.save(model.state_dict(), final_model_path)

    # Evaluate on test set
    print("\nEvaluating contrastive model on test set...")
    test_accuracy = evaluate(model, test_loader, args.device)

    # Plot t-SNE visualization of latent space
    print("\nGenerating t-SNE visualizations for contrastive latent space...")

    # Using the encoder part of our model
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super(EncoderWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            return self.model(x, mode='latent')

    encoder_wrapper = EncoderWrapper(model)
    plot_tsne(encoder_wrapper, test_loader, args.device, output_path='results/contrastive/tsne_visualization.png')

    # Create and save comprehensive report
    report = create_report(args, contrastive_history, clf_history, test_accuracy)

    with open('results/contrastive/training_report.txt', 'w') as f:
        f.write(report)

    print(f"\nTraining complete!")
    print(f"Model saved to: {final_model_path}")
    print(f"Results and plots saved to: results/contrastive/")
    print(f"Comprehensive report saved to: results/contrastive/training_report.txt")
    print(f"{'='*50}\n")

    # Also print the report to console
    print(report)
