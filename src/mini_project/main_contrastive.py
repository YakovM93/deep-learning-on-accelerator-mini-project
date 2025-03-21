"""
Task 1.2.3 Contrastive Learning:
    Apply a contrastive learning approach with a modified encoder architecture to train a latent classifier with a pretrained encoder
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

from models.contrastive_model import (
    ContrastiveEncoderCIFAR , ContrastiveEncoderMNIST, LatentClassifier
)
# Import utility functions from utils.py
from utils import plot_tsne, visualize_reconstructions, linear_interpolation

# ----- Contrastive Functions -----
def nt_xent_loss(z, temperature=0.5):
    z = nn.functional.normalize(z, dim=1)
    batch_size = z.shape[0] // 2
    sim_matrix = torch.matmul(z, z.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
    positives = torch.cat([torch.diag(sim_matrix, batch_size),
                           torch.diag(sim_matrix, -batch_size)])
    exp_sim = torch.exp(sim_matrix / temperature)
    denom = exp_sim.sum(dim=1)
    loss = -torch.log(torch.exp(positives / temperature) / denom)
    return loss.mean()

class TwoCropsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # Apply the transformation twice to get two different views
        return [self.transform(x), self.transform(x)]

# Define an identity collate function to avoid default merging for contrastive mode
def identity_collate(batch):
    return batch

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for reproducibility')
    parser.add_argument('--data-path', default="./data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=24, type=int, help='Batch size')
    parser.add_argument('--latent-dim', default=128, type=int, help='Latent dimension')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--epochs-clf', default=180, type=int, help='Number of epochs for classifier training')
    parser.add_argument('--epochs-contrastive', default=2, type=int, help='Number of epochs for contrastive training')
    parser.add_argument('--mnist', action='store_true', default=True, help='Use MNIST if True, else CIFAR10')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--optimizer', default='adamw', type=str, help='Optimizer: adam / adamw / sgd / rmsprop')
    # If you want backward compatibility with old command line
    parser.add_argument('--mode', default='contrastive', type=str, help='Legacy parameter - only contrastive mode is supported')
    return parser.parse_args()

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

def main():
    args = get_args()
    device = args.device
    freeze_seeds(args.seed)

    # Data Augmentation for training
    if args.mnist:
        base_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_transform = TwoCropsTransform(base_transform)
        full_train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=test_transform)
    else:
        base_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_transform = TwoCropsTransform(base_transform)
        full_train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

    val_size = 5000
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = test_transform

    # Data loaders for contrastive learning
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=identity_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=identity_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Helper function to create optimizer & scheduler
    def create_optimizer_scheduler(model_params, lr, opt_type='adamw', T_max=10):
        if opt_type.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=lr)
        elif opt_type.lower() == 'adamw':
            optimizer = optim.AdamW(model_params, lr=lr)
        elif opt_type.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model_params, lr=lr)
        elif opt_type.lower() == 'sgd':
            optimizer = optim.SGD(model_params, lr=lr, momentum=0.80)
        else:
            optimizer = optim.AdamW(model_params, lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return optimizer, scheduler

    # Initialize contrastive model
    if args.mnist:
        model_contrast = ContrastiveEncoderMNIST(latent_dim=args.latent_dim).to(device)
    else:
        model_contrast = ContrastiveEncoderCIFAR(latent_dim=args.latent_dim).to(device)
    optimizer_contrast, scheduler_contrast = create_optimizer_scheduler(model_contrast.parameters(), args.lr, args.optimizer, T_max=args.epochs_contrastive)
    print("Training Encoder with Contrastive Loss...")

    # Contrastive training loop
    for epoch in range(args.epochs_contrastive):
        total_loss = 0.0
        model_contrast.train()

        for batch_idx, batch in enumerate(train_loader):
            # Extract images and labels
            images_list = []
            labels_list = []

            for sample in batch:
                # Each sample is (image, label)
                image, label = sample
                images_list.append(image)
                labels_list.append(label)

            # Stack images
            images = torch.stack(images_list).to(device)
            labels = torch.tensor(labels_list).to(device)

            # For contrastive learning, we need two views
            # Since we don't have two separate views, split the batch in half
            batch_size = images.size(0)
            half_size = batch_size // 2

            view1 = images[:half_size]
            view2 = images[half_size:2*half_size]

            # If batch size is odd, adjust
            if view1.size(0) != view2.size(0):
                min_size = min(view1.size(0), view2.size(0))
                view1 = view1[:min_size]
                view2 = view2[:min_size]

            # Print shapes for debugging first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"View1 shape: {view1.shape}")
                print(f"View2 shape: {view2.shape}")

            # Process views through encoder
            optimizer_contrast.zero_grad()
            z1 = model_contrast.encode(view1)
            z2 = model_contrast.encode(view2)

            # Concatenate for contrastive loss
            z = torch.cat([z1, z2], dim=0)

            loss = nt_xent_loss(z, temperature=0.5)
            loss.backward()
            optimizer_contrast.step()

            # Update total loss
            total_loss += loss.item() * view1.size(0)

        if scheduler_contrast is not None:
            scheduler_contrast.step()

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model_contrast.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Extract images for validation
                images_list = []

                for sample in batch:
                    image, _ = sample
                    images_list.append(image)

                images = torch.stack(images_list).to(device)

                # Split into two views
                batch_size = images.size(0)
                half_size = batch_size // 2

                view1 = images[:half_size]
                view2 = images[half_size:2*half_size]

                # If batch size is odd, adjust
                if view1.size(0) != view2.size(0):
                    min_size = min(view1.size(0), view2.size(0))
                    view1 = view1[:min_size]
                    view2 = view2[:min_size]

                z1 = model_contrast.encode(view1)
                z2 = model_contrast.encode(view2)
                z = torch.cat([z1, z2], dim=0)

                loss = nt_xent_loss(z, temperature=0.5)
                total_val_loss += loss.item() * view1.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(f"[Contrastive Epoch {epoch+1}/{args.epochs_contrastive}] Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

    # Test contrastive loss
    total_test_loss = 0.0
    model_contrast.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            batch_size = images.size(0)
            images = images.to(device)

            # Create two copies for the contrastive framework
            z1 = model_contrast.encode(images)
            z2 = model_contrast.encode(images)  # Use the same images as second view

            z = torch.cat([z1, z2], dim=0)
            loss = nt_xent_loss(z, temperature=0.5)
            total_test_loss += loss.item() * batch_size

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    print(f"Test Contrastive Loss: {avg_test_loss:.4f}")

    # Create new standard data loaders for classifier training
    print("Creating new standard data loaders for classifier training...")

    # Create fresh datasets with regular transforms
    if args.mnist:
        standard_train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=base_transform)
        standard_test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=test_transform)
    else:
        standard_train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=base_transform)
        standard_test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

    # Split again to maintain same train/val split
    standard_train_subset, standard_val_subset = random_split(standard_train_dataset, [train_size, val_size])

    # Create standard data loaders (no TwoCropsTransform, no identity_collate)
    standard_train_loader = DataLoader(standard_train_subset, batch_size=args.batch_size, shuffle=True)
    standard_val_loader = DataLoader(standard_val_subset, batch_size=args.batch_size, shuffle=False)
    standard_test_loader = DataLoader(standard_test_dataset, batch_size=args.batch_size, shuffle=False)

    # Freeze the contrastively-trained encoder
    for param in model_contrast.parameters():
        param.requires_grad = False

    # Train classifier on frozen encoder
    classifier = LatentClassifier(latent_dim=args.latent_dim, num_classes=10).to(device)
    optimizer_clf, scheduler_clf = create_optimizer_scheduler(classifier.parameters(), args.lr, args.optimizer, T_max=args.epochs_clf)
    print("Training Classifier on Frozen Contrastive Encoder...")

    # Train classifier using standard data loaders
    for epoch in range(args.epochs_clf):
        train_clf_loss = train_classifier(model_contrast, classifier, standard_train_loader, optimizer_clf, device, scheduler_clf)
        val_clf_loss, val_clf_acc = evaluate_classifier(model_contrast, classifier, standard_val_loader, device)
        print(f"[Contrastive CLF Epoch {epoch+1}/{args.epochs_clf}] Train Loss={train_clf_loss:.4f} | Val Loss={val_clf_loss:.4f}, Acc={val_clf_acc*100:.2f}%")

    # Evaluate on test data
    test_clf_loss, test_clf_acc = evaluate_classifier(model_contrast, classifier, standard_test_loader, device)
    print(f"Contrastive Test Classifier Loss={test_clf_loss:.4f}, Test Accuracy={test_clf_acc*100:.2f}%")

    # Generate t-SNE visualization
    print("Generating t-SNE plots for Contrastively-Trained Encoder...")
    plot_tsne(model_contrast.encode, standard_test_loader, device, image_tsne_path='tsne_img_contrastive.png', latent_tsne_path='tsne_latent_contrastive.png')

if __name__ == "__main__":
    main()
