import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Self-Supervised Autoencoders with Extra Conv Layer & BN/Dropout ---
class ConvAutoencoderMNIST(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
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
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),

            # Modified final layer to ensure 28×28 output
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Changed kernel_size to 4
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 64, 4, 4)
        x = self.dropout(x)
        x = self.decoder(x)
        # Ensure exactly 28×28 output by cropping if necessary
        if x.size(-1) != 28:
            x = x[:, :, :28, :28]
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class ConvAutoencoderCIFAR(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Added Extra Conv Layer
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Flatten(),
            # shape after 3 conv layers for CIFAR (32x32):
            # 1st conv -> 16x16, 2nd -> 8x8, 3rd -> 4x4 with 128 channels
            # so 128 * 4 * 4 = 2048
            nn.Linear(128 * 4 * 4, latent_dim)
        )
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # For symmetry with the new encoder:
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.2)  # optional dropout

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 4, 4)
        x = self.dropout(x)  # dropout
        x = self.decoder(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


# --- Classifier with extra Conv layer & BN/Dropout ---
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


# Classification-Guided Models with an extra Conv layer:
class ConvEncoderClassifierMNIST(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        # We'll add an initial conv-based feature extractor in the classifier:
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # -> shape ~ (16, 14, 14)
        )
        # Then our encoder is simpler, or we can keep it the same style:
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(),
            # Correct dimension: 64*4*4 = 1024
            nn.Linear(1024, latent_dim)
        )

        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Sequential(
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


    def forward(self, x):
        #debug
        x = self.initial_conv(x)
        z = self.encoder(x)
        z = self.dropout(z)
        logits = self.classifier(z)
        return logits

    def encode(self, x):
        x = self.initial_conv(x)
        z = self.encoder(x)
        return z

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class ConvEncoderClassifierCIFAR(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        # Extra conv at the start
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # -> shape ~ (32, 16, 16)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Flatten(),
            # shape after that -> (128, 4, 4) => 128*4*4 = 2048
            nn.Linear(128*4*4, latent_dim)
        )
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Sequential(
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

    def forward(self, x):
        x = self.initial_conv(x)
        z = self.encoder(x)
        z = self.dropout(z)
        logits = self.classifier(z)
        return logits

    def encode(self, x):
        x = self.initial_conv(x)
        z = self.encoder(x)
        return z

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

## Contrastive Models
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

        self.decoder_input = nn.Linear(latent_dim, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),

            # Modified final layer to ensure 28×28 output
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Changed kernel_size to 4
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 64, 4, 4)
        x = self.dropout(x)
        x = self.decoder(x)
        # Ensure exactly 28×28 output by cropping if necessary
        if x.size(-1) != 28:
            x = x[:, :, :28, :28]
        return x

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

        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # For symmetry with the new encoder:
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.2)  # optional dropout

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 4, 4)
        x = self.dropout(x)  # dropout
        x = self.decoder(x)
        return x

    def forward(self, x):
        return self.encode(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
