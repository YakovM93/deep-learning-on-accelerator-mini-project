import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, latent_dim, input_size):
        super().__init__()
        self.input_size = input_size  # Store input size
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Compute feature map size dynamically
        dummy_input = torch.zeros(1, in_channels, input_size, input_size)
        with torch.no_grad():
            feature_map_size = self.cnn(dummy_input).shape[1:]
        self.flatten_size = feature_map_size[0] * feature_map_size[1] * feature_map_size[2]

        # Fully connected layers for mean and log variance
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class DecoderCNN(nn.Module):
    def __init__(self, latent_dim, out_channels, input_size):
        super().__init__()
        self.input_size = input_size  # Store input size
        dummy_input = torch.zeros(1, out_channels, input_size, input_size)
        with torch.no_grad():
            feature_map_size = EncoderCNN(out_channels, latent_dim, input_size).cnn(dummy_input).shape[1:]
        self.flatten_size = feature_map_size[0] * feature_map_size[1] * feature_map_size[2]

        self.fc = nn.Linear(latent_dim, self.flatten_size)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Scale output to [-1, 1]
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 256, 4, 4)
        return self.deconv(z)


class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, input_size):
        super().__init__()
        self.encoder = EncoderCNN(in_channels, latent_dim, input_size)
        self.decoder = DecoderCNN(latent_dim, in_channels, input_size)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# Loss function for VAE
def vae_loss(x, x_reconstructed, mu, logvar):
    recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_div) / x.shape[0]  # Average over batch
