import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import argparse
import random
import os
from models.model import(
    LatentModel,
    evaluate,
    visualize_reconstructions
)
from utils import(
    plot_tsne,
    train_self_supervised,
    train_classifier_with_frozen_encoder,
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
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str, help='Directory to save checkpoints')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    return parser.parse_args()

if __name__ == "__main__":
    # Get arguments
    args = get_args()
    freeze_seeds(args.seed)

    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set appropriate transforms based on the dataset
    if args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # Load dataset
    if args.mnist:
        train_full = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
        input_dim = 28 * 28  # MNIST image size
    else:
        train_full = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        input_dim = 32 * 32 * 3  # CIFAR10 image size

    # Split training data into train and validation sets
    val_size = int(len(train_full) * args.val_split)
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*50}")
    print(f"Running self-supervised learning on {'MNIST' if args.mnist else 'CIFAR10'}")
    print(f"{'='*50}\n")

    # Initialize model
    print(f"Creating model with input dimension {input_dim} and latent dimension {args.latent_dim}")
    model = LatentModel(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        num_classes=NUM_CLASSES,
        encoder_hidden_dims=[512, 256],
        decoder_hidden_dims=[256, 512],
        classifier_hidden_dims=[64]
    )

    # Define checkpoint filenames
    autoencoder_checkpoint = f"{args.checkpoint_dir}/autoencoder_checkpoint.pth"
    classifier_checkpoint = f"{args.checkpoint_dir}/classifier_checkpoint.pth"
    final_model_path = 'results/full_model.pth'

    # Check if we should override existing checkpoints
    if args.override_checkpoint:
        if os.path.exists(autoencoder_checkpoint):
            os.remove(autoencoder_checkpoint)
        if os.path.exists(classifier_checkpoint):
            os.remove(classifier_checkpoint)
        print("Existing checkpoints removed, starting training from scratch.")
        args.resume = False

    # Phase 1: Train autoencoder (self-supervised)
    print("\nPhase 1: Self-supervised autoencoder training")
    print("-" * 50)

    # If there's a saved checkpoint and we want to resume, load it
    resume_phase1 = args.resume and os.path.exists(autoencoder_checkpoint)
    if resume_phase1:
        print(f"Resuming autoencoder training from checkpoint: {autoencoder_checkpoint}")

    model, ae_history = train_self_supervised(
        model, train_loader, val_loader,
        args.device, epochs=args.epochs, lr=args.lr,
        checkpoint_dir=args.checkpoint_dir, resume=resume_phase1
    )

    # Print final MAE values
    print(f"Final training MAE: {ae_history['train_mae_values'][-1]:.6f}")
    print(f"Final validation MAE: {ae_history['val_mae_values'][-1]:.6f}")

    # Save the trained model after self-supervised phase
    torch.save(model.state_dict(), 'results/self_supervised_model.pth')

    # Visualize some reconstructions
    print("\nVisualizing reconstructions...")
    visualize_reconstructions(model, test_loader, args.device)

    # Phase 2: Train classifier with frozen encoder
    print("\nPhase 2: Supervised classifier training with frozen encoder")
    print("-" * 50)

    # If there's a saved checkpoint and we want to resume, load it
    resume_phase2 = args.resume and os.path.exists(classifier_checkpoint)
    if resume_phase2:
        print(f"Resuming classifier training from checkpoint: {classifier_checkpoint}")

    model, clf_history = train_classifier_with_frozen_encoder(
        model, train_loader, val_loader,
        args.device, epochs=args.epochs, lr=args.lr,
        checkpoint_dir=args.checkpoint_dir, resume=resume_phase2
    )

    # Save the trained model after classifier phase
    torch.save(model.state_dict(), final_model_path)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = evaluate(model, test_loader, args.device)

    # Plot t-SNE visualization of latent space
    print("\nGenerating t-SNE visualizations...")
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
    print(f"Dataset: {'MNIST' if args.mnist else 'CIFAR10'}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Results saved in 'results' directory")
    print(f"Checkpoints saved in '{args.checkpoint_dir}' directory")
    print(f"{'='*50}\n")
