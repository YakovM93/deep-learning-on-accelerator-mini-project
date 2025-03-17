import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """
    DCGAN-style Encoder that progressively downsamples the input image
    to extract latent representation optimized for classification.
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

class ClassifierHead(nn.Module):
    """
    Classification head that takes a latent vector and predicts class probabilities.
    """
    def __init__(self, latent_dim=128, num_classes=10, hidden_dims=None):
        super(ClassifierHead, self).__init__()
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

class ClassifierAutoencoder(nn.Module):
    """
    A classifier autoencoder that optimizes the latent space for classification.

    Unlike a traditional autoencoder, this model focuses on creating a latent
    representation that's good for classification, rather than for reconstruction.

    This model can be used for:
      • Extracting classification-optimized latent representations
      • Supervised classification directly

    Forward modes:
      - 'all': returns (classification_logits, latent)
      - 'classifier': returns classification_logits
      - 'latent': returns latent representation
    """
    def __init__(self, in_channels=3, latent_dim=128, num_classes=10, initial_features=64):
        super(ClassifierAutoencoder, self).__init__()
        self.encoder = ConvEncoder(in_channels=in_channels,
                                   latent_dim=latent_dim,
                                   initial_features=initial_features)

        self.classifier = ClassifierHead(latent_dim=latent_dim, num_classes=num_classes)

        # For compatibility, store the dimensions
        self.input_dim = in_channels * 32 * 32
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def forward(self, x, mode='all'):
        """
        Args:
          x: an input batch of images of shape (N, C, H, W). Expected H = W = 32.
          mode: a string indicating which outputs to return:
                'all' returns (classification_logits, latent)
                'classifier' returns classification logits
                'latent' returns only the latent representation.
        """
        latent = self.encoder(x)  # shape: (N, latent_dim)

        if mode == 'all':
            classification = self.classifier(latent)  # shape: (N, num_classes)
            return classification, latent
        elif mode == 'classifier':
            classification = self.classifier(latent)
            return classification
        elif mode == 'latent':
            return latent
        else:
            raise ValueError(f"Unknown mode: {mode}")
