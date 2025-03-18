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
import datetime
import hashlib
import json
from models.cnn_autoencoder_cifar import ConvAutoencoder
from helpers.helper import visualize_reconstructions , generate_report
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
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--val-split', default=0.1, type=float, help='Fraction of training data to use for validation')
    parser.add_argument('--override-checkpoint', action='store_true', default=False, help='Override checkpoint and start training from scratch')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from checkpoint')
    parser.add_argument('--checkpoint-dir', default='checkpoints_cnn', type=str, help='Directory to save checkpoints')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--initial-features', default=64, type=int, help='Number of features in first conv layer')
    return parser.parse_args()


def train_self_supervised(model, train_loader, val_loader, device, epochs=20, lr=1e-3,
                          checkpoint_dir='checkpoints', resume=False, checkpoint_freq=5,
                          run_id=None, report_data=None):
    """
    Train the CNN autoencoder using self-supervised reconstruction loss with checkpointing.

    Args:
        model: ConvAutoencoder
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
        resume: Whether to resume from checkpoint
        checkpoint_freq: Frequency (in epochs) to save checkpoints
        run_id: Unique identifier for this run
        report_data: Dictionary to store reporting data

    Returns:
        Trained model and training history
    """
    if report_data is None:
        report_data = {}

    model = model.to(device)

    # Record starting time
    start_time = datetime.datetime.now()
    report_data['autoencoder_training_start_time'] = start_time.strftime("%Y-%m-%d %H:%M:%S")

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

    # For tracking best model
    best_val_loss = float('inf')
    best_epoch = -1

    # Checkpoint filename
    checkpoint_file = f"{checkpoint_dir}/autoencoder_checkpoint_{run_id}.pth"

    # Add to report
    report_data['autoencoder_checkpoint_path'] = checkpoint_file

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
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', -1)
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        print("No checkpoint found or resume not requested, starting from scratch")

    epoch_data = []

    for epoch in range(start_epoch, epochs):
        epoch_start_time = datetime.datetime.now()

        # Training phase
        model.train()
        train_loss = 0
        train_mae = 0

        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(device)

            # Forward pass (autoencoder mode)
            reconstructed, _ = model(data, mode='autoencoder')

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

                loss = criterion_mse(reconstructed, data)
                val_loss += loss.item()

                # Calculate MAE
                mae = criterion_mae(reconstructed, data)
                val_mae += mae.item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_losses.append(val_loss)
        val_mae_values.append(val_mae)

        epoch_end_time = datetime.datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # Save best model separately
            best_model_path = f"{checkpoint_dir}/autoencoder_best_{run_id}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae
            }, best_model_path)
            report_data['autoencoder_best_model_path'] = best_model_path

        print(f"Epoch {epoch+1}/{epochs}, Train Loss (MSE): {train_loss:.6f}, Train MAE: {train_mae:.6f}")
        print(f"Val Loss (MSE): {val_loss:.6f}, Val MAE: {val_mae:.6f}")

        # Record epoch statistics
        epoch_data.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'duration_seconds': epoch_duration
        })

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_mae_values': train_mae_values,
                'val_mae_values': val_mae_values,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch
            }
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}")

    end_time = datetime.datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    # Output directory for plots
    plot_dir = f"results/{run_id}"
    os.makedirs(plot_dir, exist_ok=True)

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
    mse_plot_path = f'{plot_dir}/autoencoder_mse_loss.png'
    plt.savefig(mse_plot_path)
    plt.close()

    # Add to report
    report_data['autoencoder_mse_plot_path'] = mse_plot_path

    # Plot MAE
    plt.figure(figsize=(10, 5))
    plt.plot(train_mae_values, label='Training MAE')
    plt.plot(val_mae_values, label='Validation MAE')
    plt.title('Autoencoder Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    mae_plot_path = f'{plot_dir}/autoencoder_mae.png'
    plt.savefig(mae_plot_path)
    plt.close()

    # Add to report
    report_data['autoencoder_mae_plot_path'] = mae_plot_path

    # Update report data with training statistics
    report_data['autoencoder_training_end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    report_data['autoencoder_training_duration_seconds'] = training_duration
    report_data['autoencoder_epochs'] = epochs
    report_data['autoencoder_best_epoch'] = best_epoch + 1
    report_data['autoencoder_best_val_loss'] = best_val_loss
    report_data['autoencoder_final_train_loss'] = train_losses[-1]
    report_data['autoencoder_final_val_loss'] = val_losses[-1]
    report_data['autoencoder_final_train_mae'] = train_mae_values[-1]
    report_data['autoencoder_final_val_mae'] = val_mae_values[-1]
    report_data['autoencoder_epoch_data'] = epoch_data

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mae_values': train_mae_values,
        'val_mae_values': val_mae_values,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }


def train_classifier_with_frozen_encoder(model, train_loader, val_loader, device, epochs=20, lr=1e-3,
                                        checkpoint_dir='checkpoints', resume=False, checkpoint_freq=5,
                                        run_id=None, report_data=None):
    """
    Train the classifier using the frozen CNN encoder with checkpointing.

    Args:
        model: ConvAutoencoder with pre-trained encoder
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
        resume: Whether to resume from checkpoint
        checkpoint_freq: Frequency (in epochs) to save checkpoints
        run_id: Unique identifier for this run
        report_data: Dictionary to store reporting data

    Returns:
        Trained model and training history
    """
    if report_data is None:
        report_data = {}

    model = model.to(device)

    # Record starting time
    start_time = datetime.datetime.now()
    report_data['classifier_training_start_time'] = start_time.strftime("%Y-%m-%d %H:%M:%S")

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

    # For tracking best model
    best_val_acc = 0.0
    best_epoch = -1

    # Checkpoint filename
    checkpoint_file = f"{checkpoint_dir}/classifier_checkpoint_{run_id}.pth"

    # Add to report
    report_data['classifier_checkpoint_path'] = checkpoint_file

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
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_epoch = checkpoint.get('best_epoch', -1)
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        print("No checkpoint found or resume not requested, starting from scratch")

    epoch_data = []

    for epoch in range(start_epoch, epochs):
        epoch_start_time = datetime.datetime.now()

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

        epoch_end_time = datetime.datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()

        # Check if this is the best model so far
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch

            # Save best model separately
            best_model_path = f"{checkpoint_dir}/classifier_best_{run_id}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss
            }, best_model_path)
            report_data['classifier_best_model_path'] = best_model_path

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.2f}%")

        # Record epoch statistics
        epoch_data.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'duration_seconds': epoch_duration
        })

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch
            }
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}")

    end_time = datetime.datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    # Output directory for plots
    plot_dir = f"results/{run_id}"
    os.makedirs(plot_dir, exist_ok=True)

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
    loss_plot_path = f'{plot_dir}/classifier_training_loss.png'
    plt.savefig(loss_plot_path)
    plt.close()

    # Add to report
    report_data['classifier_loss_plot_path'] = loss_plot_path

    # Plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Classifier Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    acc_plot_path = f'{plot_dir}/classifier_training_accuracy.png'
    plt.savefig(acc_plot_path)
    plt.close()

    # Add to report
    report_data['classifier_accuracy_plot_path'] = acc_plot_path

    # Update report data with training statistics
    report_data['classifier_training_end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    report_data['classifier_training_duration_seconds'] = training_duration
    report_data['classifier_epochs'] = epochs
    report_data['classifier_best_epoch'] = best_epoch + 1
    report_data['classifier_best_val_accuracy'] = best_val_acc
    report_data['classifier_final_train_loss'] = train_losses[-1]
    report_data['classifier_final_val_loss'] = val_losses[-1]
    report_data['classifier_final_train_accuracy'] = train_accuracies[-1]
    report_data['classifier_final_val_accuracy'] = val_accuracies[-1]
    report_data['classifier_epoch_data'] = epoch_data

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }

def evaluate(model, test_loader, device, report_data=None):
    """
    Evaluate the model on the test set.

    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cuda' or 'cpu')
        report_data: Dictionary to store reporting data

    Returns:
        Test accuracy
    """
    if report_data is None:
        report_data = {}

    model.eval()
    correct = 0
    total = 0

    # For calculating per-class accuracy
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    # For confusion matrix
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            # Forward pass (classifier mode)
            outputs = model(data, mode='classifier')

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Per class accuracy
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

            # Store for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            per_class_accuracy[i] = 100 * class_correct[i] / class_total[i]
        else:
            per_class_accuracy[i] = 0.0

    # Update report
    report_data['test_accuracy'] = accuracy
    report_data['per_class_accuracy'] = per_class_accuracy

    # If CIFAR-10, include class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Create a prettier class report for humans
    class_report = "\nClass Accuracy Report:\n"
    class_report += "=" * 40 + "\n"
    class_report += "{:<15} {:<15} {:<10}\n".format("Class Index", "Class Name", "Accuracy")
    class_report += "-" * 40 + "\n"

    for i in range(NUM_CLASSES):
        class_report += "{:<15} {:<15} {:.2f}%\n".format(
            i, class_names[i], per_class_accuracy[i]
        )

    report_data['class_accuracy_report'] = class_report
    print(class_report)

    return accuracy
if __name__ == "__main__":
    # Get arguments
    args = get_args()
    freeze_seeds(args.seed)

    # Create output directories for results
    os.makedirs('results', exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Generate a unique run ID for this experiment
    config_hash = hashlib.md5(json.dumps({
        'latent_dim': args.latent_dim,
        'initial_features': args.initial_features,
        'seed': args.seed,
        'epochs': args.epochs,
        'lr': args.lr
    }, sort_keys=True).encode()).hexdigest()[:8]
    run_id = f"cnn_{args.latent_dim}_f{args.initial_features}_{config_hash}"

    # Create results directory for this run
    run_dir = f"results/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    # Initialize report data
    report_data = {
        'run_id': run_id,
        'config': {
            'latent_dim': args.latent_dim,
            'initial_features': args.initial_features,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'seed': args.seed,
            'device': args.device
        },
        'start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Set transforms for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    in_channels = 3  # CIFAR10 has 3 channels
    report_data['dataset'] = 'CIFAR10'

    # Load CIFAR-10 dataset
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
    print(f"Running CNN-based self-supervised learning on CIFAR10")
    print(f"Run ID: {run_id}")
    print(f"{'='*50}\n")

    # Initialize CNN-based model
    print(f"Creating CNN-based model with latent dimension {args.latent_dim}")
    model = ConvAutoencoder(
        in_channels=in_channels,  # CIFAR-10 images are 32x32x3
        latent_dim=args.latent_dim,
        num_classes=NUM_CLASSES,
        initial_features=args.initial_features
    )

    # Define checkpoint filenames
    autoencoder_checkpoint = f"{args.checkpoint_dir}/autoencoder_checkpoint_{run_id}.pth"
    classifier_checkpoint = f"{args.checkpoint_dir}/classifier_checkpoint_{run_id}.pth"
    final_model_path = f'{run_dir}/cnn_full_model.pth'

    # Record model architecture in report
    report_data['model_summary'] = str(model)
    report_data['model_parameters'] = sum(p.numel() for p in model.parameters())

    # Check if we should override existing checkpoints
    if args.override_checkpoint:
        if os.path.exists(autoencoder_checkpoint):
            os.remove(autoencoder_checkpoint)
        if os.path.exists(classifier_checkpoint):
            os.remove(classifier_checkpoint)
        print("Existing checkpoints removed, starting training from scratch.")
        args.resume = False

    # Phase 1: Train autoencoder (self-supervised)
    print("\nPhase 1: Self-supervised CNN autoencoder training")
    print("-" * 50)

    # If there's a saved checkpoint and we want to resume, load it
    resume_phase1 = args.resume and os.path.exists(autoencoder_checkpoint)
    if resume_phase1:
        print(f"Resuming CNN autoencoder training from checkpoint: {autoencoder_checkpoint}")

    model, ae_history = train_self_supervised(
        model, train_loader, val_loader,
        args.device, epochs=args.epochs, lr=args.lr,
        checkpoint_dir=args.checkpoint_dir, resume=resume_phase1,
        run_id=run_id, report_data=report_data
    )

    # Print final MAE values
    print(f"Final training MAE: {ae_history['train_mae_values'][-1]:.6f}")
    print(f"Final validation MAE: {ae_history['val_mae_values'][-1]:.6f}")

    # Save the trained model after self-supervised phase
    torch.save(model.state_dict(), f'{run_dir}/cnn_self_supervised_model.pth')

    # Visualize some reconstructions
    print("\nVisualizing CNN reconstructions...")
    visualize_reconstructions(model, test_loader, args.device, run_id=run_id, report_data=report_data)

    # Phase 2: Train classifier with frozen encoder
    print("\nPhase 2: Supervised classifier training with frozen CNN encoder")
    print("-" * 50)

    # If there's a saved checkpoint and we want to resume, load it
    resume_phase2 = args.resume and os.path.exists(classifier_checkpoint)
    if resume_phase2:
        print(f"Resuming CNN classifier training from checkpoint: {classifier_checkpoint}")

    model, clf_history = train_classifier_with_frozen_encoder(
        model, train_loader, val_loader,
        args.device, epochs=args.epochs, lr=args.lr,
        checkpoint_dir=args.checkpoint_dir, resume=resume_phase2,
        run_id=run_id, report_data=report_data
    )

    # Save the trained model after classifier phase
    torch.save(model.state_dict(), final_model_path)
    report_data['final_model_path'] = final_model_path

    # Evaluate on test set
    print("\nEvaluating CNN model on test set...")
    test_accuracy = evaluate(model, test_loader, args.device, report_data=report_data)

    # Plot t-SNE visualization of latent space
    print("\nGenerating t-SNE visualizations for CNN latent space...")
    # Using the encoder part of our model
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super(EncoderWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            return self.model(x, mode='latent')

    encoder_wrapper = EncoderWrapper(model)
    tsne_plot_path = plot_tsne(encoder_wrapper, test_loader, args.device)
    report_data['tsne_plot_path'] = tsne_plot_path

    # Record completion time
    report_data['end_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_duration = (datetime.datetime.strptime(report_data['end_time'], "%Y-%m-%d %H:%M:%S") -
                        datetime.datetime.strptime(report_data['start_time'], "%Y-%m-%d %H:%M:%S")).total_seconds()
    report_data['total_duration_seconds'] = total_duration

    # Generate report
    report_path = generate_report(args, run_id, report_data)
    print(f"Detailed report saved to: {report_path}")

    print(f"\nSummary:")
    print(f"{'='*50}")
    print(f"Model: CNN-based Autoencoder (F={args.initial_features})")
    print(f"Dataset: CIFAR10")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Results saved in '{run_dir}' directory")
    print(f"Checkpoints saved in '{args.checkpoint_dir}' directory")
    print(f"Run ID: {run_id}")
    print(f"{'='*50}\n")
