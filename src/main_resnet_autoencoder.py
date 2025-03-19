import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import argparse
import random
import os
from models.resnet_autoencoder import ResNetBasedLatentModel
from utils import (
    plot_tsne,
    save_checkpoint,
    load_checkpoint
)

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    parser.add_argument('--checkpoint-dir', default='checkpoints_resnet', type=str, help='Directory to save checkpoints')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    return parser.parse_args()


def train_self_supervised(model, train_loader, val_loader, device, epochs=50, lr=1e-3,
                          checkpoint_dir='checkpoints', resume=False, checkpoint_freq=5):
    """
    Train the ResNet-based model using self-supervised reconstruction loss with checkpointing.
    This version preserves the 2D image structure for both input and reconstructed images.

    Args:
        model: ResNetBasedLatentModel
        train_loader: DataLoader for training data
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

    # Set model to only train encoder and decoder
    model.freeze_classifier(True)
    model.freeze_encoder(False)
    model.freeze_decoder(False)

    optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=lr
    )
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()  # L1Loss is Mean Absolute Error

    # Initialize history lists
    train_losses = []
    val_losses = []
    train_mae_values = []
    val_mae_values = []

    # Checkpoint filename
    checkpoint_file = f"{checkpoint_dir}/autoencoder_checkpoint.pth"

    # Resume from checkpoint if requested
    start_epoch = 0
    if resume and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_mae_values = checkpoint.get('train_mae_values', [])
        val_mae_values = checkpoint.get('val_mae_values', [])
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        print("No checkpoint found or resume not requested, starting from scratch")

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_mae = 0

        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(device)

            # Forward pass (autoencoder mode)
            reconstructed, _ = model(data, mode='autoencoder')

            # NOTE: No flattening here - we use the tensor shapes as they are
            loss = criterion_mse(reconstructed, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate MAE (for monitoring only, not used in optimization)
            with torch.no_grad():
                mae = criterion_mae(reconstructed, data)
                train_mae += mae.item()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        train_losses.append(train_loss)
        train_mae_values.append(train_mae)

        # Validation phase
        model.eval()
        val_loss = 0
        val_mae = 0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)

                # Forward pass (autoencoder mode)
                reconstructed, _ = model(data, mode='autoencoder')

                # NOTE: No flattening here - we use the tensor shapes as they are
                loss = criterion_mse(reconstructed, data)
                val_loss += loss.item()

                # Calculate MAE
                mae = criterion_mae(reconstructed, data)
                val_mae += mae.item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_losses.append(val_loss)
        val_mae_values.append(val_mae)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss (MSE): {train_loss:.6f}, Train MAE: {train_mae:.6f}")
        print(f"Val Loss (MSE): {val_loss:.6f}, Val MAE: {val_mae:.6f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_mae_values': train_mae_values,
                'val_mae_values': val_mae_values
            }
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}")

    # Create and save the plots
    # Plot MSE loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss (MSE)')
    plt.plot(val_losses, label='Validation Loss (MSE)')
    plt.title('Autoencoder Training and Validation MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/autoencoder_mse_loss.png')
    plt.close()

    # Plot MAE
    plt.figure(figsize=(10, 5))
    plt.plot(train_mae_values, label='Training MAE')
    plt.plot(val_mae_values, label='Validation MAE')
    plt.title('Autoencoder Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/autoencoder_mae.png')
    plt.close()

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mae_values': train_mae_values,
        'val_mae_values': val_mae_values
    }


def train_classifier_with_frozen_encoder(model, train_loader, val_loader, device, epochs=20, lr=1e-3,
                                        checkpoint_dir='checkpoints', resume=False, checkpoint_freq=5):
    """
    Train the classifier using the frozen ResNet encoder with checkpointing.
    This function is specifically designed for the ResNetBasedLatentModel architecture.

    Args:
        model: ResNetBasedLatentModel with pre-trained encoder
        train_loader: DataLoader for training data
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
    model.freeze_encoder(True)
    model.freeze_decoder(True)
    model.freeze_classifier(False)

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
    plt.savefig('results/classifier_training_loss.png')
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
    plt.savefig('results/classifier_training_accuracy.png')
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


def visualize_reconstructions(model, test_loader, device, num_examples=10):
    """
    Visualize original images and their reconstructions.

    Args:
        model: Model
        test_loader: DataLoader for test data
        device: Device to use ('cuda' or 'cpu')
        num_examples: Number of examples to visualize
    """
    model.eval()

    # Get some test examples
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images[:num_examples].to(device)

    # Get reconstructions
    with torch.no_grad():
        reconstructed, _ = model(images, mode='autoencoder')

    # Convert to numpy for plotting
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    # Plot original and reconstructed images
    plt.figure(figsize=(20, 4))
    for i in range(num_examples):
        # Original
        ax = plt.subplot(2, num_examples, i + 1)
        if images.shape[1] == 1:  # Grayscale (e.g., MNIST)
            plt.imshow(images[i][0], cmap='gray')
        else:  # RGB (e.g., CIFAR)
            plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.title(f"Original: {labels[i]}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Reconstructed
        ax = plt.subplot(2, num_examples, num_examples + i + 1)
        if reconstructed.shape[1] == 1:  # Grayscale
            plt.imshow(reconstructed[i][0], cmap='gray')
        else:  # RGB
            plt.imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        plt.title(f"Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.suptitle('ResNet Autoencoder Reconstructions', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/resnet_reconstructions.png')
    plt.close()


if __name__ == "__main__":
    # Get arguments
    args = get_args()
    freeze_seeds(args.seed)

    # Create output directories for results
    os.makedirs('results', exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set appropriate transforms based on the dataset
    if args.mnist:
        transform = transforms.Compose([
            # Convert MNIST to 3 channels to match ResNet input
            transforms.ToTensor(),
            transforms.Resize((32, 32)),  # Resize to 32x32
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # Repeat grayscale to 3 channels
        ])
        in_channels = 3  # Now MNIST will be 3 channels
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        in_channels = 3  # CIFAR10 has 3 channels

    # Load dataset
    if args.mnist:
        train_full = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    else:
        train_full = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

    # Split training data into train and validation sets
    val_size = int(len(train_full) * args.val_split)
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*50}")
    print(f"Running ResNet-based self-supervised learning on {'MNIST' if args.mnist else 'CIFAR10'}")
    print(f"{'='*50}\n")

    # Initialize ResNet-based model
    print(f"Creating ResNet-based model with latent dimension {args.latent_dim}")
    model = ResNetBasedLatentModel(
        in_channels=in_channels,
        latent_dim=args.latent_dim,
        num_classes=NUM_CLASSES
    )

    # Define checkpoint filenames
    autoencoder_checkpoint = f"{args.checkpoint_dir}/resnet_autoencoder_checkpoint.pth"
    classifier_checkpoint = f"{args.checkpoint_dir}/resnet_classifier_checkpoint.pth"
    final_model_path = 'results/resnet_full_model.pth'

    # Check if we should override existing checkpoints
    if args.override_checkpoint:
        if os.path.exists(autoencoder_checkpoint):
            os.remove(autoencoder_checkpoint)
        if os.path.exists(classifier_checkpoint):
            os.remove(classifier_checkpoint)
        print("Existing checkpoints removed, starting training from scratch.")
        args.resume = False

    # Phase 1: Train autoencoder (self-supervised)
    print("\nPhase 1: Self-supervised ResNet autoencoder training")
    print("-" * 50)

    # If there's a saved checkpoint and we want to resume, load it
    resume_phase1 = args.resume and os.path.exists(autoencoder_checkpoint)
    if resume_phase1:
        print(f"Resuming ResNet autoencoder training from checkpoint: {autoencoder_checkpoint}")

    model, ae_history = train_self_supervised(
        model, train_loader, val_loader,
        args.device, epochs=args.epochs, lr=args.lr,
        checkpoint_dir=args.checkpoint_dir, resume=resume_phase1
    )

    # Print final MAE values
    print(f"Final training MAE: {ae_history['train_mae_values'][-1]:.6f}")
    print(f"Final validation MAE: {ae_history['val_mae_values'][-1]:.6f}")

    # Save the trained model after self-supervised phase
    torch.save(model.state_dict(), 'results/resnet_self_supervised_model.pth')

    # Visualize some reconstructions
    print("\nVisualizing ResNet reconstructions...")
    visualize_reconstructions(model, test_loader, args.device)

    # Phase 2: Train classifier with frozen encoder
    print("\nPhase 2: Supervised classifier training with frozen ResNet encoder")
    print("-" * 50)

    # If there's a saved checkpoint and we want to resume, load it
    resume_phase2 = args.resume and os.path.exists(classifier_checkpoint)
    if resume_phase2:
        print(f"Resuming ResNet classifier training from checkpoint: {classifier_checkpoint}")

    model, clf_history = train_classifier_with_frozen_encoder(
        model, train_loader, val_loader,
        args.device, epochs=args.epochs, lr=args.lr,
        checkpoint_dir=args.checkpoint_dir, resume=resume_phase2
    )

    # Save the trained model after classifier phase
    torch.save(model.state_dict(), final_model_path)

    # Evaluate on test set
    print("\nEvaluating ResNet model on test set...")
    test_accuracy = evaluate(model, test_loader, args.device)

    # Plot t-SNE visualization of latent space
    print("\nGenerating t-SNE visualizations for ResNet latent space...")
    # Using the encoder part of our model
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super(EncoderWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            return self.model(x, mode='latent')

    encoder_wrapper = EncoderWrapper(model)
    plot_tsne(encoder_wrapper, test_loader, args.device)

    print(f"\nSummary:")
    print(f"{'='*50}")
    print(f"Model: ResNet-based Autoencoder")
    print(f"Dataset: {'MNIST' if args.mnist else 'CIFAR10'}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Results saved in 'results' directory")
    print(f"Checkpoints saved in '{args.checkpoint_dir}' directory")
    print(f"{'='*50}\n")
