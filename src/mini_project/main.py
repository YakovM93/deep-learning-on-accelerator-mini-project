import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

# Import your models from model.py
from models.model import (ConvAutoencoderMNIST, ConvAutoencoderCIFAR, LatentClassifier,
                          ConvEncoderClassifierMNIST, ConvEncoderClassifierCIFAR)
# Import utility functions from utils.py
from utils import plot_tsne

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for reproducibility')
    parser.add_argument('--data-path', default="./data", type=str, help='Path to dataset')
    parser.add_argument('--save-path', default='./trained-models', type=str, help='Path to save the trained models')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--latent-dim', default=128, type=int, help='Latent dimension')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--epochs-ae', default=35, type=int, help='Number of epochs for autoencoder pretraining')
    parser.add_argument('--epochs-clf', default=15, type=int, help='Number of epochs for classifier training (self-supervised mode)')
    parser.add_argument('--epochs-cg', default=20, type=int, help='Number of epochs for classification guided training')
    parser.add_argument('--mnist', action='store_true', default=False, help='Use MNIST if True, else CIFAR10')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--mode', default='self_supervised', choices=['self_supervised', 'classification_guided', 'evaluation_ae', 'evaluation_clf'],
                        help='Training mode: self_supervised (1.2.1) or classification_guided (1.2.2)')
    parser.add_argument('--pretrained_model', default=None, type=str, help='Path to pretrained model')
    # Optionally add a flag to switch optimizers or schedulers
    parser.add_argument('--optimizer', default='adamw', type=str, help='Optimizer: adam / adamw / sgd / rmsprop')
    return parser.parse_args()

# def reconstruction_loss(x_recon, x):
#     return nn.functional.l1_loss(x_recon, x)

def reconstruction_loss(x_recon, x):
    # Ensure both tensors have the same shape before computing loss
    if x_recon.shape != x.shape:
        # For MNIST, you might need to adjust the output shape
        # If x_recon has a different number of channels, reshape it to match x
        x_recon = x_recon.view(x.shape)
    return nn.functional.l1_loss(x_recon, x)

def train_autoencoder(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    for images, _ in dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        recon = model(images)
        loss = reconstruction_loss(recon, images)
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()

def evaluate_autoencoder(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            recon = model(images)
            loss = reconstruction_loss(recon, images)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def train_classifier(frozen_encoder, classifier, dataloader, optimizer, device, scheduler=None):
    frozen_encoder.eval()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            z = frozen_encoder.encode(images)
        logits = classifier(z)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(dataloader.dataset)

def evaluate_classifier(frozen_encoder, classifier, dataloader, device):
    frozen_encoder.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            z = frozen_encoder.encode(images)
            logits = classifier(z)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc

def train_classification_guided(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(dataloader.dataset)

def evaluate_classification_guided(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc

def visualize_reconstructions(model, test_loader, device, num_examples=10):
    """
    Visualize original images and their reconstructions.

    Args:
        model: LatentModel
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
        reconstructed = model(images)

    # Convert to numpy for plotting
    images = images.cpu().numpy()

    # Reshape reconstructed images
    if len(images.shape) > 2:
        # For image datasets like MNIST or CIFAR
        reconstructed = reconstructed.view(images.shape).cpu().numpy()
    else:
        # For flat data
        reconstructed = reconstructed.cpu().numpy().reshape(images.shape)

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
        if images.shape[1] == 1:  # Grayscale
            plt.imshow(reconstructed[i][0], cmap='gray')
        else:  # RGB
            plt.imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        plt.title(f"Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.suptitle('Autoencoder Reconstructions', fontsize=16)
    plt.tight_layout()
    plt.savefig('reconstructions.png')
    print(f"Saved reconstruction visualization to {os.path.abspath('reconstructions.png')}")
    plt.close()

def linear_interpolation(model, dataloader, device, steps=10, n_image_pairs=1, save_path=None):
    """
    Perform linear interpolation between pairs of images in the latent space.

    Args:
        model: The autoencoder model
        dataloader: DataLoader containing images
        device: Device to use (cuda or cpu)
        steps: Number of interpolation steps between images
        n_image_pairs: Number of image pairs to interpolate
        save_path: Path to save the visualization (if None, will show the plot)
    """
    print("starting linear interpolation")
    model.eval()
    images, labels = next(iter(dataloader))

    if len(images) < 2*n_image_pairs:
        print(f"Not enough images for {n_image_pairs} pairs. Need at least {2*n_image_pairs} images.")
        return

    fig, all_axes = plt.subplots(n_image_pairs, steps+2, figsize=(3*(steps+2), 3*n_image_pairs))

    for pair_idx in range(n_image_pairs):
        # Get two images
        idx1, idx2 = pair_idx*2, pair_idx*2+1
        img1 = images[idx1].unsqueeze(0).to(device)
        img2 = images[idx2].unsqueeze(0).to(device)
        label1, label2 = labels[idx1].item(), labels[idx2].item()

        with torch.no_grad():
            z1 = model.encode(img1)
            z2 = model.encode(img2)

        # Create interpolation steps
        alphas = np.linspace(0, 1, steps)
        interpolated_images = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            with torch.no_grad():
                x_interp = model.decode(z_interp)
            interpolated_images.append(x_interp.squeeze(0).cpu())

        # Get axes for this pair
        if n_image_pairs == 1:
            axes = all_axes
        else:
            axes = all_axes[pair_idx]

        # Plot the original images at the beginning and end
        img1_np = img1.squeeze(0).cpu()
        img2_np = img2.squeeze(0).cpu()

        if img1_np.shape[0] == 1:  # Grayscale
            axes[0].imshow(img1_np.squeeze(), cmap="gray")
            axes[-1].imshow(img2_np.squeeze(), cmap="gray")
        else:  # RGB
            axes[0].imshow(np.transpose(img1_np, (1, 2, 0)))
            axes[-1].imshow(np.transpose(img2_np, (1, 2, 0)))

        axes[0].set_title(f"Original\nClass: {label1}")
        axes[-1].set_title(f"Original\nClass: {label2}")
        axes[0].axis("off")
        axes[-1].axis("off")

        # Plot the interpolated images
        for i, img in enumerate(interpolated_images):
            if img.shape[0] == 1:  # Grayscale
                axes[i+1].imshow(img.squeeze(), cmap="gray")
            else:  # RGB
                axes[i+1].imshow(np.transpose(img, (1, 2, 0)))
            axes[i+1].set_title(f"α={alphas[i]:.2f}")
            axes[i+1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved reconstruction visualization to {os.path.abspath(save_path)}")
    plt.show()



def main():
    args = get_args()
    #debug
    if args.mnist:
        print("MNIST")
    device = args.device
    freeze_seeds(args.seed)

    # Data Augmentation for training
    if args.mnist:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        full_train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=test_transform)
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        full_train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

    val_size = 5000
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    # For val dataset, we can use test_transform to avoid augmentation in validation
    val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Helper function to create optim & scheduler
    def create_optimizer_scheduler(model_params, lr, opt_type='adamw', T_max=10):
        if opt_type.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=lr)
        elif opt_type.lower() == 'adamw':
            optimizer = optim.AdamW(model_params, lr=lr)
        elif opt_type.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model_params, lr=lr)
        elif opt_type.lower() == 'sgd':
            optimizer = optim.SGD(model_params, lr=lr, momentum=0.85)
        else:
            optimizer = optim.AdamW(model_params, lr=lr)

        # Example: CosineAnnealingLR over the number of epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return optimizer, scheduler

    # Mode Selection
    if args.mode == 'self_supervised':
        # Self-Supervised Autoencoder Training (1.2.1)
        if args.mnist:
            autoencoder = ConvAutoencoderMNIST(latent_dim=args.latent_dim).to(device)
        else:
            autoencoder = ConvAutoencoderCIFAR(latent_dim=args.latent_dim).to(device)

        optimizer_ae, scheduler_ae = create_optimizer_scheduler(autoencoder.parameters(), args.lr, args.optimizer, T_max=args.epochs_ae)
        print("Training Autoencoder (Self-Supervised)...")
        for epoch in range(args.epochs_ae):
            train_loss = train_autoencoder(autoencoder, train_loader, optimizer_ae, device, scheduler_ae)
            val_loss = evaluate_autoencoder(autoencoder, val_loader, device)
            print(f"[AE Epoch {epoch+1}/{args.epochs_ae}] Train MAE={train_loss:.4f} | Val MAE={val_loss:.4f}")

        test_mae = evaluate_autoencoder(autoencoder, test_loader, device)
        print(f"Test Reconstruction MAE: {test_mae:.4f}")

        # Save the trained autoencoder model

        autoencoder.save_model(f"{args.save_path}/autoencoder.pth")


        # Train Classifier on Frozen Encoder
        for param in autoencoder.parameters():
            param.requires_grad = False
        classifier = LatentClassifier(latent_dim=args.latent_dim, num_classes=10).to(device)

        optimizer_clf, scheduler_clf = create_optimizer_scheduler(classifier.parameters(), args.lr, args.optimizer, T_max=args.epochs_clf)
        print("Training Classifier on Frozen Encoder...")
        for epoch in range(args.epochs_clf):
            train_clf_loss = train_classifier(autoencoder, classifier, train_loader, optimizer_clf, device, scheduler_clf)
            val_clf_loss, val_clf_acc = evaluate_classifier(autoencoder, classifier, val_loader, device)
            print(f"[CLF Epoch {epoch+1}/{args.epochs_clf}] Train Loss={train_clf_loss:.4f} | Val Loss={val_clf_loss:.4f}, Acc={val_clf_acc*100:.2f}%")

        test_clf_loss, test_clf_acc = evaluate_classifier(autoencoder, classifier, test_loader, device)
        print(f"Test Classifier Loss={test_clf_loss:.4f}, Test Accuracy={test_clf_acc*100:.2f}%")

        # Save the trained classifier model
        classifier.save_model(f"{args.save_path}/classifier.pth")

        # Qualitative Evaluations
        print("Visualizing Reconstructions...")
        visualize_reconstructions(autoencoder, test_loader, device)
        print("Performing Linear Interpolation on two images...")
        linear_interpolation(autoencoder, test_loader, device, steps=10)

        print("Generating t-SNE plots for Self-Supervised Model...")
        plot_tsne(autoencoder.encode, test_loader, device, image_tsne_path='tsne_img_selfsup.png', latent_tsne_path='tsne_latent_selfsup.png')

    elif args.mode == 'classification_guided':
        # Classification-Guided Training (1.2.2)
        if args.mnist:
            model_cg = ConvEncoderClassifierMNIST(latent_dim=args.latent_dim, num_classes=10).to(device)
        else:
            model_cg = ConvEncoderClassifierCIFAR(latent_dim=args.latent_dim, num_classes=10).to(device)

        optimizer_cg, scheduler_cg = create_optimizer_scheduler(model_cg.parameters(), args.lr, args.optimizer, T_max=args.epochs_cg)
        print("Training Classification-Guided Model...")
        for epoch in range(args.epochs_cg):
            train_loss = train_classification_guided(model_cg, train_loader, optimizer_cg, device, scheduler_cg)
            val_loss, val_acc = evaluate_classification_guided(model_cg, val_loader, device)
            print(f"[Epoch {epoch+1}/{args.epochs_cg}] Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

        test_loss, test_acc = evaluate_classification_guided(model_cg, test_loader, device)
        print(f"Test Loss={test_loss:.4f}, Test Accuracy={test_acc*100:.2f}%")

        # Save the trained classification-guided model
        model_cg.save_model(f"{args.save_path}/classification_guided_model.pth")

        # t-SNE for Classification-Guided Model using its encoder
        def encode_fn(x):
            return model_cg.encode(x)
        print("Generating t-SNE plots for Classification-Guided Model...")
        plot_tsne(encode_fn, test_loader, device, image_tsne_path='tsne_img_cg.png', latent_tsne_path='tsne_latent_cg.png')
    elif args.mode == 'evaluation_ae':
        if not args.pretrained_model:
            raise ValueError("Pretrained model path is required for evaluation.")
        pth = torch.load(args.pretrained_model)
        if args.mnist:
            autoencoder = ConvAutoencoderMNIST(latent_dim=args.latent_dim).to(device)
        else:
            autoencoder = ConvAutoencoderCIFAR(latent_dim=args.latent_dim).to(device)
        autoencoder.load_state_dict(pth)
        # Qualitative Evaluations
        print("Visualizing Reconstructions...")
        visualize_reconstructions(autoencoder, test_loader, device)
        print("Performing Linear Interpolation on two images...")
        linear_interpolation(autoencoder, test_loader, device, steps=10)

        # print("Generating t-SNE plots for Self-Supervised Model...")
        # plot_tsne(autoencoder.encode, test_loader, device, image_tsne_path='tsne_img_selfsup.png', latent_tsne_path='tsne_latent_selfsup.png')

if __name__ == "__main__":
    main()
