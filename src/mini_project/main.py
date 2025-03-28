
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

# Models from models.py
from models import (
    ConvAutoencoderMNIST, ConvAutoencoderCIFAR,
    ConvEncoderClassifierMNIST, ConvEncoderClassifierCIFAR,
    LatentClassifier, ContrastiveEncoderCIFAR, ContrastiveEncoderMNIST
)
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
    parser.add_argument('--save-path', default='./trained-models/', type=str, help='Path to save the trained models')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--latent-dim', default=128, type=int, help='Latent dimension')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--epochs-ae', default=35, type=int, help='Number of epochs for autoencoder pretraining')
    parser.add_argument('--epochs-clf', default=15, type=int, help='Number of epochs for classifier training')
    parser.add_argument('--epochs-cg', default=30, type=int, help='Number of epochs for classification guided training')
    parser.add_argument('--epochs-contrastive', default=60, type=int, help='Number of epochs for contrastive training')
    parser.add_argument('--mnist', action='store_true', default=False, help='Use MNIST if True, else CIFAR10')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--mode', default='self_supervised',
                        choices=['self_supervised', 'classification_guided', 'contrastive', 'all', 'all-test'],
                        help='Training mode')
    parser.add_argument('--optimizer', default='adamw', type=str, help='Optimizer: adam / adamw / sgd / rmsprop')
    return parser.parse_args()

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
        total_loss += loss.item() * images.size(0)
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(dataloader.dataset)

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

def evaluate_autoencoder_reconstruction(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list) and isinstance(batch[0], tuple):
                # This is a batch from identity_collate
                images = torch.stack([item[0] for item in batch]).to(device)
            else:
                # This is a standard batch (images, labels)
                images, _ = batch
                images = images.to(device)

            # Get reconstructions
            z = model.encode(images)
            reconstructions = model.decode(z)

            # Calculate reconstruction loss
            loss = reconstruction_loss(images, reconstructions)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples

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

def run_self_supervised(args, train_loader, val_loader, test_loader, device):
    print("Running Self-Supervised Training Mode")

    # Create model
    if args.mnist:
        autoencoder = ConvAutoencoderMNIST(latent_dim=args.latent_dim).to(device)
    else:
        autoencoder = ConvAutoencoderCIFAR(latent_dim=args.latent_dim).to(device)

    # Phase 1: Training Autoencoder (Self-Supervised)
    optimizer_ae, scheduler_ae = create_optimizer_scheduler(autoencoder.parameters(), args.lr, args.optimizer, T_max=args.epochs_ae)
    print("Training Autoencoder (Self-Supervised)...")

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs_ae):
        train_loss = train_autoencoder(autoencoder, train_loader, optimizer_ae, device, scheduler_ae)
        val_loss = evaluate_autoencoder(autoencoder, val_loader, device)
        print(f"[AE Epoch {epoch+1}/{args.epochs_ae}] Train MAE={train_loss:.4f} | Val MAE={val_loss:.4f}")

        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Plot training curves
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{args.save_path}/training_curves.png")
    print(f"saved loss training curves of autoencoder (Phase 1) to {os.path.abspath(args.save_path)}/training_curves.png")
    plt.close()

    test_mae = evaluate_autoencoder(autoencoder, test_loader, device)
    print(f"Test Reconstruction MAE: {test_mae:.4f}")

    # Write test results to file
    with open(f"{args.save_path}/test_results.txt", "w") as f:
        f.write(f"Test Reconstruction MAE: {test_mae:.4f}\n")

    # Save the trained autoencoder model
    autoencoder.save_model(f"{args.save_path}/autoencoder.pth")

    # Phase 2: Train Classifier on Frozen Encoder
    for param in autoencoder.parameters():
        param.requires_grad = False
    classifier = LatentClassifier(latent_dim=args.latent_dim, num_classes=10).to(device)

    optimizer_clf, scheduler_clf = create_optimizer_scheduler(classifier.parameters(), args.lr, args.optimizer, T_max=args.epochs_clf)
    print("Training Classifier on Frozen Encoder...")

    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.epochs_clf):
        train_clf_loss = train_classifier(autoencoder, classifier, train_loader, optimizer_clf, device, scheduler_clf)
        _, train_clf_acc = evaluate_classifier(autoencoder, classifier, train_loader, device)
        val_clf_loss, val_clf_acc = evaluate_classifier(autoencoder, classifier, val_loader, device)

        # Store metrics
        train_losses.append(train_clf_loss)
        train_accuracies.append(train_clf_acc)
        val_losses.append(val_clf_loss)
        val_accuracies.append(val_clf_acc)

        print(f"[CLF Epoch {epoch+1}/{args.epochs_clf}] Train Loss={train_clf_loss:.4f} | Val Loss={val_clf_loss:.4f}, Acc={val_clf_acc*100:.2f}%")

    # Plot losses
    plt.figure(figsize=(10,6))
    print(f"Plotting {len(train_losses)} train loss points and {len(val_losses)} validation loss points ")
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_path}/classifier_loss.png")
    print(f"saved classifier loss curves of autoencoder (Phase 2) to {os.path.abspath(args.save_path)}/classifier_loss.png")
    plt.close()

    # Plot accuracies
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_accuracies) + 1), [acc * 100 for acc in train_accuracies], label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), [acc * 100 for acc in val_accuracies], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_path}/classifier_accuracy.png")
    print(f"saved classifier accuracy curves of autoencoder (Phase 2) to {os.path.abspath(args.save_path)}/classifier_accuracy.png")
    plt.close()

    test_clf_loss, test_clf_acc = evaluate_classifier(autoencoder, classifier, test_loader, device)
    print(f"Test Classifier Loss={test_clf_loss:.4f}, Test Accuracy={test_clf_acc*100:.2f}%")

    # Write test classifier results to file
    with open(f"{args.save_path}/test_results.txt", "a") as f:
        f.write(f"Test Classifier Loss: {test_clf_loss:.4f}\n")
        f.write(f"Test Classifier Accuracy: {test_clf_acc*100:.2f}%\n")

    # Save the trained classifier model
    classifier.save_model(f"{args.save_path}/classifier.pth")

    # Qualitative Evaluations
    print("Visualizing Reconstructions...")
    visualize_reconstructions(autoencoder, test_loader, device, save_path=f"{args.save_path}/reconstructions.png")
    print("Performing Linear Interpolation on two images...")
    linear_interpolation(autoencoder, test_loader, device, steps=10, n_image_pairs=1, save_path=f"{args.save_path}/interpolation.png")

    print("Generating t-SNE plots for Self-Supervised Model...")
    plot_tsne(autoencoder.encode, test_loader, device, image_tsne_path=f"{args.save_path}/tsne_img_selfsup.png", latent_tsne_path=f"{args.save_path}/tsne_latent_selfsup.png")

def run_classification_guided(args, train_loader, val_loader, test_loader, device):
    print("Running Classification-Guided Training Mode")

    # Create model
    if args.mnist:
        model_cg = ConvEncoderClassifierMNIST(latent_dim=args.latent_dim, num_classes=10).to(device)
    else:
        model_cg = ConvEncoderClassifierCIFAR(latent_dim=args.latent_dim, num_classes=10).to(device)

    optimizer_cg, scheduler_cg = create_optimizer_scheduler(model_cg.parameters(), args.lr, args.optimizer, T_max=args.epochs_cg)
    print("Training Classification-Guided Model...")

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.epochs_cg):
        train_loss = train_classification_guided(model_cg, train_loader, optimizer_cg, device, scheduler_cg)
        val_loss, val_acc = evaluate_classification_guided(model_cg, val_loader, device)
        print(f"[Epoch {epoch+1}/{args.epochs_cg}] Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    # Plot losses
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss Over Time')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(range(1, len(val_accuracies) + 1), [acc * 100 for acc in val_accuracies], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{args.save_path}/classification_guided_training_curves.png")
    print(f"saved classification-guided training curves to {os.path.abspath(args.save_path)}/classification_guided_training_curves.png")
    plt.close()

    test_loss, test_acc = evaluate_classification_guided(model_cg, test_loader, device)
    print(f"Test Loss={test_loss:.4f}, Test Accuracy={test_acc*100:.2f}%")

    # Write test results to file
    with open(f"{args.save_path}/test_results.txt", "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")

    # Save the trained classification-guided model
    model_cg.save_model(f"{args.save_path}/classification_guided_model.pth")

    # t-SNE for Classification-Guided Model using its encoder
    def encode_fn(x):
        return model_cg.encode(x)
    print("Generating t-SNE plots for Classification-Guided Model...")
    plot_tsne(encode_fn, test_loader, device, image_tsne_path=f"{args.save_path}/tsne_img_cg.png", latent_tsne_path=f"{args.save_path}/tsne_latent_cg.png")

def run_contrastive(args, train_loader_contrastive, val_loader_contrastive, test_loader, standard_train_loader, standard_val_loader, standard_test_loader, device):
    print("Running Contrastive Learning Mode")

    # Initialize contrastive model
    if args.mnist:
        model_contrast = ContrastiveEncoderMNIST(latent_dim=args.latent_dim).to(device)
    else:
        model_contrast = ContrastiveEncoderCIFAR(latent_dim=args.latent_dim).to(device)

    optimizer_contrast, scheduler_contrast = create_optimizer_scheduler(model_contrast.parameters(), args.lr, args.optimizer, T_max=args.epochs_contrastive)
    print("Training Encoder with Contrastive Loss...")

    # Contrastive training loop
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_recon_losses = []  # Track reconstruction losses
    val_recon_losses = []    # Track reconstruction losses

    for epoch in range(args.epochs_contrastive):
        total_loss = 0.0
        total_recon_loss = 0.0  # For tracking reconstruction error
        model_contrast.train()
        total_train = 0

        for batch_idx, batch in enumerate(train_loader_contrastive):
            # Extract and process the batch - handling different formats safely
            try:
                if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], tuple):
                    # This is the format from identity_collate
                    images_list = []
                    for sample in batch:
                        if isinstance(sample[0], list):  # If using TwoCropsTransform
                            # Each sample[0] contains two views of the same image
                            for view in sample[0]:
                                images_list.append(view)
                        else:
                            images_list.append(sample[0])
                    images = torch.stack(images_list).to(device)
                else:
                    # Regular format
                    images, _ = batch
                    images = images.to(device)
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, list):
                    print(f"First element type: {type(batch[0])}")
                    if isinstance(batch[0], tuple):
                        print(f"First tuple element type: {type(batch[0][0])}")
                continue  # Skip problematic batches

            # For contrastive learning, we need two views
            # Split the batch in half to create two views
            batch_size = images.size(0)
            if batch_size < 2:
                continue  # Skip batches that are too small

            half_size = batch_size // 2
            view1 = images[:half_size]
            view2 = images[half_size:2*half_size]

            # If batch size is odd, adjust
            if view1.size(0) != view2.size(0):
                min_size = min(view1.size(0), view2.size(0))
                view1 = view1[:min_size]
                view2 = view2[:min_size]

            # Process views through encoder
            optimizer_contrast.zero_grad()
            z1 = model_contrast.encode(view1)
            z2 = model_contrast.encode(view2)

            # Concatenate for contrastive loss
            z = torch.cat([z1, z2], dim=0)

            loss = nt_xent_loss(z, temperature=0.5)
            loss.backward()
            optimizer_contrast.step()

            # Update total loss and sample count
            total_loss += loss.item() * view1.size(0)
            total_train += view1.size(0)

            # Measure reconstruction loss (without using it for training)
            with torch.no_grad():
                # Get reconstructions for view1
                reconstructions = model_contrast.decode(z1)
                recon_loss = reconstruction_loss(view1, reconstructions)
                total_recon_loss += recon_loss.item() * view1.size(0)

        if scheduler_contrast is not None:
            scheduler_contrast.step()

        # Skip epoch metrics if no batches were processed
        if total_train == 0:
            print(f"[Contrastive Epoch {epoch+1}/{args.epochs_contrastive}] No valid batches found, skipping metrics")
            continue

        avg_train_loss = total_loss / total_train
        avg_train_recon_loss = total_recon_loss / total_train

        # Validation
        model_contrast.eval()
        total_val_loss = 0.0
        total_val_recon_loss = 0.0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader_contrastive:
                # Extract images for validation - handle different formats
                try:
                    if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], tuple):
                        # This is the format from identity_collate
                        images_list = []
                        for sample in batch:
                            if isinstance(sample[0], list):  # If using TwoCropsTransform
                                # Each sample[0] contains two views of the same image
                                for view in sample[0]:
                                    images_list.append(view)
                            else:
                                images_list.append(sample[0])
                        images = torch.stack(images_list).to(device)
                    else:
                        # Regular format
                        images, _ = batch
                        images = images.to(device)
                except Exception as e:
                    print(f"Error processing validation batch: {e}")
                    continue  # Skip problematic batches

                # Skip if batch is too small
                batch_size = images.size(0)
                if batch_size < 2:
                    continue

                # Split into two views
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
                total_val += view1.size(0)

                # Calculate reconstruction loss
                reconstructions = model_contrast.decode(z1)
                recon_loss = reconstruction_loss(view1, reconstructions)
                total_val_recon_loss += recon_loss.item() * view1.size(0)

        # Skip validation metrics if no batches were processed
        if total_val == 0:
            print(f"[Contrastive Epoch {epoch+1}/{args.epochs_contrastive}] No valid validation batches found, skipping validation metrics")
            continue

        avg_val_loss = total_val_loss / total_val
        avg_val_recon_loss = total_val_recon_loss / total_val

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_recon_losses.append(avg_train_recon_loss)
        val_recon_losses.append(avg_val_recon_loss)

        print(f"[Contrastive Epoch {epoch+1}/{args.epochs_contrastive}] Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Train Recon MAE={avg_train_recon_loss:.4f} | Val Recon MAE={avg_val_recon_loss:.4f}")

    # Plot training curves
    plt.figure(figsize=(12,5))

    # Plot losses
    print(f"Plotting {len(train_losses)} and {len(val_losses)} validation losses points")
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrastive Learning Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{args.save_path}/contrastive_learning_losses_curves.png')
    print(f"saved Contrastive Learning Losses Curves in {os.path.abspath(f'{args.save_path}/contrastive_learning_losses_curves.png')}")
    plt.close()

    # Plot reconstruction MAE
    plt.figure(figsize=(12,5))
    plt.plot(range(1, len(train_recon_losses)+1), train_recon_losses, label='Train Reconstruction MAE')
    plt.plot(range(1, len(val_recon_losses)+1), val_recon_losses, label='Validation Reconstruction MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Contrastive Learning Reconstruction Error (MAE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{args.save_path}/contrastive_reconstruction_mae_curves.png')
    print(f"saved Contrastive Learning Reconstruction MAE Curves in {os.path.abspath(f'{args.save_path}/contrastive_reconstruction_mae_curves.png')}")
    plt.close()

    # Test contrastive loss
    total_test_loss = 0.0
    total_test_recon_loss = 0.0
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

            # Calculate reconstruction loss
            reconstructions = model_contrast.decode(z1)
            recon_loss = reconstruction_loss(images, reconstructions)
            total_test_recon_loss += recon_loss.item() * batch_size

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    avg_test_recon_loss = total_test_recon_loss / len(test_loader.dataset)
    print(f"Test Contrastive Loss: {avg_test_loss:.4f}")
    print(f"Test Reconstruction MAE: {avg_test_recon_loss:.4f}")

    # Write test results to file
    with open(f'{args.save_path}/contrastive_test_results.txt', 'w') as f:
        f.write(f"Test Contrastive Loss: {avg_test_loss:.4f}\n")
        f.write(f"Test Reconstruction MAE: {avg_test_recon_loss:.4f}\n")

    # Evaluate autoencoder reconstruction on test set
    test_recon_mae = evaluate_autoencoder_reconstruction(model_contrast, test_loader, device)
    print(f"Test Reconstruction MAE (comprehensive): {test_recon_mae:.4f}")
    with open(f'{args.save_path}/contrastive_test_results.txt', 'a') as f:
        f.write(f"Test Reconstruction MAE (comprehensive): {test_recon_mae:.4f}\n")

    # Save the trained model
    model_contrast.save_model(f"{args.save_path}/contrastive_model.pth")

    # Freeze the contrastively-trained encoder
    for param in model_contrast.parameters():
        param.requires_grad = False

    # Train classifier on frozen encoder
    classifier = LatentClassifier(latent_dim=args.latent_dim, num_classes=10).to(device)
    optimizer_clf, scheduler_clf = create_optimizer_scheduler(classifier.parameters(), args.lr, args.optimizer, T_max=args.epochs_clf)
    print("Training Classifier on Frozen Contrastive Encoder...")

    # Initialize lists to store metrics
    train_clf_losses = []
    val_clf_losses = []
    train_clf_accs = []
    val_clf_accs = []

    # Train classifier using standard data loaders
    for epoch in range(args.epochs_clf):
        train_clf_loss = train_classifier(model_contrast, classifier, standard_train_loader, optimizer_clf, device, scheduler_clf)
        val_clf_loss, val_clf_acc = evaluate_classifier(model_contrast, classifier, standard_val_loader, device)
        _, train_clf_acc = evaluate_classifier(model_contrast, classifier, standard_train_loader, device)

        # Store metrics
        train_clf_losses.append(train_clf_loss)
        val_clf_losses.append(val_clf_loss)
        train_clf_accs.append(train_clf_acc)
        val_clf_accs.append(val_clf_acc)

        print(f"[Contrastive CLF Epoch {epoch+1}/{args.epochs_clf}] Train Loss={train_clf_loss:.4f} | Val Loss={val_clf_loss:.4f}, Acc={val_clf_acc*100:.2f}%")

    # Plot training curves
    plt.figure(figsize=(12,5))
    print(f"Plotting {len(train_clf_losses)} and {len(val_clf_losses)} validation losses points")

    # Plot losses
    plt.subplot(1,2,1)
    plt.plot(range(1, len(train_clf_losses)+1), train_clf_losses, label='Train Loss')
    plt.plot(range(1, len(val_clf_losses)+1), val_clf_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Classifier Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(1,2,2)
    plt.plot(range(1, len(train_clf_accs)+1), [acc * 100 for acc in train_clf_accs], label='Train Accuracy')
    plt.plot(range(1, len(val_clf_accs)+1), [acc * 100 for acc in val_clf_accs], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Classifier Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{args.save_path}/contrastive_learning_classifier_training_curves.png')
    print(f"saved Contrastive Learning Classifier Training Curves in {os.path.abspath(f'{args.save_path}/contrastive_learning_classifier_training_curves.png')}")
    plt.close()

    # Evaluate on test data
    test_clf_loss, test_clf_acc = evaluate_classifier(model_contrast, classifier, standard_test_loader, device)
    print(f"Contrastive Test Classifier Loss={test_clf_loss:.4f}, Test Accuracy={test_clf_acc*100:.2f}%")

    # Write test results to file
    with open(f'{args.save_path}/contrastive_test_classifier_results.txt', 'w') as f:
        f.write(f"Test Classifier Loss: {test_clf_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_clf_acc*100:.2f}%\n")

    # Save the trained classifier
    classifier.save_model(f"{args.save_path}/contrastive_classifier.pth")

    # Generate t-SNE visualization
    print("Generating t-SNE plots for Contrastively-Trained Encoder...")
    plot_tsne(model_contrast.encode, standard_test_loader, device,
              image_tsne_path=f"{args.save_path}/tsne_img_contrastive.png",
              latent_tsne_path=f"{args.save_path}/tsne_latent_contrastive.png")

def main():
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    device = args.device
    freeze_seeds(args.seed)

    # Setup for 'all' or 'all-test' modes
    if args.mode in ['all', 'all-test']:
        if args.mode == 'all-test':
            print("Running in 'all-test' debugging mode with reduced epochs")
            # Use reduced epochs for testing
            args.epochs_ae = 2
            args.epochs_clf = 2
            args.epochs_cg = 2
            args.epochs_contrastive = 2

        datasets = ['mnist', 'cifar']
        modes = ['self_supervised', 'classification_guided', 'contrastive']

        for dataset in datasets:
            print(f"\n{'='*50}")
            print(f"Running experiments on {dataset.upper()} dataset")
            print(f"{'='*50}\n")

            # Set dataset flag
            args_copy = args
            args_copy.mnist = (dataset == 'mnist')

            # Set epoch counts based on dataset
            if dataset == 'mnist':
                args_copy.epochs_ae = 30 if args.mode != 'all-test' else 2
                args_copy.epochs_clf = 30 if args.mode != 'all-test' else 2
                args_copy.epochs_cg = 30 if args.mode != 'all-test' else 2
            else:  # cifar
                args_copy.epochs_ae = 50 if args.mode != 'all-test' else 2
                args_copy.epochs_clf = 180 if args.mode != 'all-test' else 2
                args_copy.epochs_cg = 60 if args.mode != 'all-test' else 2

            for mode in modes:
                print(f"\n{'-'*30}")
                print(f"Running {mode} mode on {dataset.upper()}")
                print(f"{'-'*30}\n")

                # Set mode and create appropriate save path
                args_copy.mode = mode
                dataset_dir = "MNIST" if dataset == "mnist" else "CIFAR"

                if mode == 'self_supervised':
                    mode_dir = "AE_CL"
                elif mode == 'classification_guided':
                    mode_dir = "CG"
                else:  # contrastive
                    mode_dir = "CONTRASTIVE"
                    # by default batch size is 24 for contrastive mode
                    original_batch_size = args_copy.batch_size
                    args_copy.batch_size = 24

                save_path = f"./results/{dataset_dir}/{mode_dir}"
                os.makedirs(save_path, exist_ok=True)
                args_copy.save_path = save_path

                # Run the experiment
                run_experiment(args_copy)

        print("\nAll experiments completed successfully!")
        return

    # Regular single experiment mode
    run_experiment(args)

def run_experiment(args):
    device = args.device

    # Data preprocessing and loading based on dataset
    if args.mnist:
        print("Using MNIST dataset")
        base_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        print("Using CIFAR10 dataset")
        base_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # Create standard datasets
    if args.mnist:
        full_train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=base_transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=test_transform)
    else:
        full_train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=base_transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

    # Split into train and validation sets
    val_size = 5000
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # For validation, we don't need augmentation
    val_dataset.dataset.transform = test_transform

    # Create standard loaders for most modes
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # For contrastive mode, we need special datasets and loaders
    contrastive_loaders = None
    if args.mode == 'contrastive':
        # Create datasets with TwoCropsTransform for contrastive learning
        train_transform_contrastive = TwoCropsTransform(base_transform)

        if args.mnist:
            full_train_dataset_contrastive = datasets.MNIST(root=args.data_path, train=True, download=True, transform=train_transform_contrastive)
        else:
            full_train_dataset_contrastive = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform_contrastive)

        train_dataset_contrastive, val_dataset_contrastive = random_split(full_train_dataset_contrastive, [train_size, val_size])
        val_dataset_contrastive.dataset.transform = train_transform_contrastive  # Keep two views for validation

        # Create contrastive data loaders
        train_loader_contrastive = DataLoader(train_dataset_contrastive, batch_size=args.batch_size, shuffle=True, collate_fn=identity_collate)
        val_loader_contrastive = DataLoader(val_dataset_contrastive, batch_size=args.batch_size, shuffle=False, collate_fn=identity_collate)

        # Save contrastive loaders for later use
        contrastive_loaders = (train_loader_contrastive, val_loader_contrastive, train_loader, val_loader, test_loader)

    # Run the appropriate mode
    if args.mode == 'self_supervised':
        run_self_supervised(args, train_loader, val_loader, test_loader, device)
    elif args.mode == 'classification_guided':
        run_classification_guided(args, train_loader, val_loader, test_loader, device)
    elif args.mode == 'contrastive':
        # Unpack contrastive loaders
        train_loader_contrastive, val_loader_contrastive, standard_train_loader, standard_val_loader, standard_test_loader = contrastive_loaders
        run_contrastive(args, train_loader_contrastive, val_loader_contrastive, test_loader,
                       standard_train_loader, standard_val_loader, standard_test_loader, device)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
