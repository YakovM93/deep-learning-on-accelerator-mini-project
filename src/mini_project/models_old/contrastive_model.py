import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveEncoderMNIST(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        # Same encoder as ConvAutoencoderMNIST
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.encode(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class ContrastiveEncoderCIFAR(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # This should expect 3 input channels for CIFAR
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Input: 3 channels
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.encode(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.RReLU(0.07, 0.2, inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.RReLU(0.07, 0.2, inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
            )

    def forward(self, z):
        return self.fc(z)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
