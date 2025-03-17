import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import datetime
import hashlib
import random

def generate_run_id():
    """Generate a unique run ID based on timestamp and random string."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_str = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return f"{timestamp}_{rand_str}"

def visualize_reconstructions(model, test_loader, device, num_examples=10, run_id=None, report_data=None):
    """
    Visualize original images and their reconstructions.

    Args:
        model: Model
        test_loader: DataLoader for test data
        device: Device to use ('cuda' or 'cpu')
        num_examples: Number of examples to visualize
        run_id: Unique identifier for this run
        report_data: Dictionary to store reporting data
    """
    if report_data is None:
        report_data = {}

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

    # For CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Plot original and reconstructed images
    plt.figure(figsize=(20, 4))
    for i in range(num_examples):
        # Original
        ax = plt.subplot(2, num_examples, i + 1)
        if images.shape[1] == 1:  # Grayscale
            plt.imshow(images[i][0], cmap='gray')
        else:  # RGB
            # Normalize to [0, 1] for proper display
            img = np.transpose(images[i], (1, 2, 0))
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            plt.imshow(np.clip(img, 0, 1))
        plt.title(f"Original: {class_names[labels[i]]}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Reconstructed
        ax = plt.subplot(2, num_examples, num_examples + i + 1)
        if reconstructed.shape[1] == 1:  # Grayscale
            plt.imshow(reconstructed[i][0], cmap='gray')
        else:  # RGB
            # Normalize to [0, 1] for proper display
            img = np.transpose(reconstructed[i], (1, 2, 0))
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            plt.imshow(np.clip(img, 0, 1))
        plt.title(f"Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.suptitle('CNN Autoencoder Reconstructions', fontsize=16)
    plt.tight_layout()

    # Save plot
    plot_dir = f"results/{run_id}"
    os.makedirs(plot_dir, exist_ok=True)
    recon_plot_path = f'{plot_dir}/cnn_reconstructions.png'
    plt.savefig(recon_plot_path)
    plt.close()

    # Add to report
    report_data['reconstruction_plot_path'] = recon_plot_path


def generate_report(args, run_id, report_data):
    """
    Generate a detailed report about the experiment.

    Args:
        args: Command line arguments
        run_id: Unique identifier for this run
        report_data: Dictionary containing training report data
    """
    # Create reports directory
    os.makedirs('reports', exist_ok=True)

    # Start building the report
    report = []
    report.append("=" * 80)
    report.append(f"CNN AUTOENCODER EXPERIMENT REPORT - {run_id}")
    report.append("=" * 80)
    report.append("")

    # Hardware and setup information
    report.append("SETUP INFORMATION")
    report.append("-" * 50)
    report.append(f"Device: {args.device}")
    report.append(f"Dataset: {'MNIST' if args.mnist else 'CIFAR10'}")
    report.append(f"Batch Size: {args.batch_size}")
    report.append(f"Latent Dimension: {args.latent_dim}")
    report.append(f"Random Seed: {args.seed}")
    report.append("")

    # Training parameters
    report.append("TRAINING PARAMETERS")
    report.append("-" * 50)
    report.append(f"Training Epochs: {args.epochs}")
    report.append(f"Learning Rate: {args.lr}")
    report.append(f"Validation Split: {args.val_split}")
    report.append("")

    # Results information
    report.append("RESULTS")
    report.append("-" * 50)

    # Add test accuracy if available
    if 'test_accuracy' in report_data:
        report.append(f"Test Accuracy: {report_data['test_accuracy']:.2f}%")

    # Add final loss values if available
    if 'final_train_loss' in report_data and 'final_val_loss' in report_data:
        report.append(f"Final Training Loss: {report_data['final_train_loss']:.6f}")
        report.append(f"Final Validation Loss: {report_data['final_val_loss']:.6f}")

    # Add MAE values if available
    if 'final_train_mae' in report_data and 'final_val_mae' in report_data:
        report.append(f"Final Training MAE: {report_data['final_train_mae']:.6f}")
        report.append(f"Final Validation MAE: {report_data['final_val_mae']:.6f}")

    report.append("")

    # Path to visualization plots
    report.append("VISUALIZATIONS")
    report.append("-" * 50)

    if 'reconstruction_plot_path' in report_data:
        report.append(f"Reconstruction Plot: {report_data['reconstruction_plot_path']}")

    if 'tsne_plot_path' in report_data:
        report.append(f"t-SNE Visualization: {report_data['tsne_plot_path']}")

    if 'loss_plot_path' in report_data:
        report.append(f"Training Loss Plot: {report_data['loss_plot_path']}")

    report.append("")

    # Model architecture information if available
    if 'model_summary' in report_data:
        report.append("MODEL ARCHITECTURE")
        report.append("-" * 50)
        report.append(report_data['model_summary'])
        report.append("")

    # Additional notes or comments
    if 'notes' in report_data:
        report.append("NOTES")
        report.append("-" * 50)
        report.append(report_data['notes'])
        report.append("")

    # Save the report to a file
    report_path = f"reports/{run_id}_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report generated and saved to {report_path}")

    return report_path
