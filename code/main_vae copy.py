from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
## Import Model
from models import autoencoder
NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    return parser.parse_args()


def train_model(model, dataloader, criterion, optimizer, device, self_supervised=True):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images = images.view(images.size(0), -1).to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)

        loss = criterion(outputs, images if self_supervised else labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate_model(model, dataloader, criterion, device, self_supervised=True):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs, _ = model(images)

            loss = criterion(outputs, images if self_supervised else labels)
            total_loss += loss.item()

            if not self_supervised:
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

    accuracy = correct / total if not self_supervised else None
    return total_loss / len(dataloader), accuracy




if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  #one possible convenient normalization. You don't have to use it.
    ])

    args = get_args()
    freeze_seeds(args.seed)


    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
        input_dim = 28 * 28
        input_shape = (28, 28)
        in_channels = 1
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=False, transform=transform)
        input_dim = 32 * 32 * 3
        input_shape = (32, 32)
        in_channels = 3

    # When you create your dataloader you should split train_dataset or test_dataset to leave some aside for validation

    train_size , val_size = int(len(train_dataset) * 0.8), len(train_dataset) - int(len(train_dataset) * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    #####
    hp = {'batch_size': 96, 'h_dim': 512, 'z_dim': 64, 'x_sigma2': 0.0007, 'learn_rate': 0.0002, 'betas': (0.5, 0.999)}
    ## load hyperparameters from dict
    latent_dim =args.latent_dim
    ## Initialize model
    encoder = autoencoder.EncoderCNN(input_shape[0], latent_dim)
    decoder = autoencoder.DecoderCNN(latent_dim, input_shape[0])
    ae = autoencoder.VAE(encoder, decoder,input_shape , hp['z_dim'])
    model = DataParallel(vae).to(args.device)
    ## define optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=hp['learn_rate'], betas=hp['betas'])
    ## define criterion

    if args.self_supervised:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    #####


    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, args.device, args.self_supervised)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, args.device, args.self_supervised)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}",
                f"Val Accuracy={val_acc:.4f}" if val_acc is not None else "")

    torch.save(model.state_dict(), "checkpoints/autoencoder.pth")


    # Evaluate model
    # if self supervised use MAE
