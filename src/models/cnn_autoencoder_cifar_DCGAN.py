import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """
    CNN Encoder that progressively downsamples the input image
    to extract latent representation.
    """
    def __init__(self, in_channels=3, latent_dim=128, initial_features=64):
            super(ConvEncoder, self).__init__()
            self.initial_features = initial_features

            # Starting with 32x32 image input
            self.main = nn.Sequential(
                # input: (N, in_channels, 32, 32)
                nn.Conv2d(in_channels, initial_features, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state: (N, initial_features, 16, 16)

                nn.Conv2d(initial_features, initial_features * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(initial_features * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state: (N, initial_features * 2, 8, 8)

                nn.Conv2d(initial_features * 2, initial_features * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(initial_features * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state: (N, initial_features * 4, 4, 4)

                nn.Conv2d(initial_features * 4, initial_features * 8, 4, 1, 0, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                # state: (N, initial_features * 8, 1, 1)
            )

            # Project to latent vector
            self.fc = nn.Linear(initial_features * 8, latent_dim)

    def forward(self, x):
        features = self.main(x)
        features = features.view(-1, self.initial_features * 8)
        latent = self.fc(features)
        return latent

class ConvDecoder(nn.Module):
    """
    CNN Decoder that progressively upsamples from latent space
    to reconstruct the original image.
    """
    def __init__(self, latent_dim=128, out_channels=3, initial_features=64):
        super(ConvDecoder, self).__init__()
        self.initial_features = initial_features

        # Project latent vector to initial feature map
        self.fc = nn.Linear(latent_dim, initial_features * 8)

        self.main = nn.Sequential(
            # input: (N, initial_features * 8, 1, 1)
            nn.ConvTranspose2d(initial_features * 8, initial_features * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(initial_features * 4),
            nn.ReLU(True),
            # state: (N, initial_features * 4, 4, 4)

            nn.ConvTranspose2d(initial_features * 4, initial_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(initial_features * 2),
            nn.ReLU(True),
            # state: (N, initial_features * 2, 8, 8)

            nn.ConvTranspose2d(initial_features * 2, initial_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(initial_features),
            nn.ReLU(True),
            # state: (N, initial_features, 16, 16)

            nn.ConvTranspose2d(initial_features, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # output: (N, out_channels, 32, 32)
        )

    def forward(self, z):
        # Reshape latent vector to initial feature map
        x = self.fc(z)
        x = x.view(-1, self.initial_features * 8, 1, 1)
        # Generate image through decoder
        x = self.main(x)
        return x

class SimpleClassifier(nn.Module):
    """
    Simple classifier that takes a latent vector as input and outputs logits for num_classes.
    """
    def __init__(self, latent_dim=128, num_classes=10, hidden_dims=None):
        super(SimpleClassifier, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        dims = [latent_dim] + hidden_dims

        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))

        # Final layer: no activation (logits)
        layers.append(nn.Linear(dims[-1], num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)

class ConvAutoencoder(nn.Module):
    """
    A CNN-based autoencoder model that includes an encoder, decoder, and classifier.

    It supports:
      • Self-supervised reconstruction (using the decoder)
      • Supervised classification (using the classifier)
      • Joint training via a single forward pass that returns reconstruction,
        classification logits, and the latent representation.

    Forward modes:
      - 'all': returns (reconstructed_image, classification_logits, latent)
      - 'autoencoder': returns (reconstructed_image, latent)
      - 'classifier': returns classification_logits
      - 'latent': returns latent representation.
    """
    def __init__(self, in_channels=3, latent_dim=128, num_classes=10, initial_features=64):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder(in_channels=in_channels,
                                   latent_dim=latent_dim,
                                   initial_features=initial_features)

        self.decoder = ConvDecoder(latent_dim=latent_dim,
                                   out_channels=in_channels,
                                   initial_features=initial_features)

        self.classifier = SimpleClassifier(latent_dim=latent_dim, num_classes=num_classes)

        # For compatibility, store the flattened image dimension (here, 3 x 32 x 32)
        self.input_dim = in_channels * 32 * 32
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def forward(self, x, mode='all'):
        """
        Args:
          x: an input batch of images of shape (N, C, H, W). Expected H = W = 32.
          mode: a string indicating which outputs to return:
                'all' returns (reconstruction, classification, latent)
                'autoencoder' returns (reconstruction, latent)
                'classifier' returns classification logits
                'latent' returns only the latent representation.
        """
        latent = self.encoder(x)  # shape: (N, latent_dim)

        if mode == 'all':
            reconstruction = self.decoder(latent)  # shape: (N, C, 32, 32)
            classification = self.classifier(latent)  # shape: (N, num_classes)
            return reconstruction, classification, latent
        elif mode == 'autoencoder':
            reconstruction = self.decoder(latent)
            return reconstruction, latent
        elif mode == 'classifier':
            classification = self.classifier(latent)
            return classification
        elif mode == 'latent':
            return latent
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def freeze_encoder(self, freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def freeze_decoder(self, freeze=True):
        for param in self.decoder.parameters():
            param.requires_grad = not freeze

    def freeze_classifier(self, freeze=True):
        for param in self.classifier.parameters():
            param.requires_grad = not freeze


class ConvEncoderClassifier(nn.Module):
    """
    A CNN-based autoencoder model that includes an encoder and classifier.

    It supports:
      • Joint training via a single forward pass that returns reconstruction,
        classification logits, and the latent representation.
    """
    def __init__(self, in_channels=3, latent_dim=128, num_classes=10, initial_features=64):
        super(ConvEncoderClassifier, self).__init__()
        self.encoder = ConvEncoder(in_channels=in_channels,
                                   latent_dim=latent_dim,
                                   initial_features=initial_features)
        self.classifier = SimpleClassifier(latent_dim=latent_dim, num_classes=num_classes)

        # For compatibility, store the flattened image dimension (here, 3 x 32 x 32)
        self.input_dim = in_channels * 32 * 32
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def forward(self, x, mode='classifier'):
        """
        Args:
          x: an input batch of images of shape (N, C, H, W). Expected H = W = 32.
          mode: a string indicating which outputs to return:
                'classifier' returns classification logits
                'latent' returns only the latent representation.
        """
        latent = self.encoder(x)  # shape: (N, latent_dim)

        if mode == 'classifier':
            classification = self.classifier(latent)
            return classification
        elif mode == 'latent':
            return latent
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def freeze_encoder(self, freeze=False):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def freeze_classifier(self, freeze=False):
        for param in self.classifier.parameters():
            param.requires_grad = not freeze
