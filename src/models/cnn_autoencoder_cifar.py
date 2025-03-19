import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """
    Convolutional Encoder that follows the structure from the original Keras model:
    Conv2D(64) -> MaxPool -> Conv2D(32) -> MaxPool -> Conv2D(16) -> Conv2D(8) -> MaxPool
    """
    def __init__(self, in_channels=3, latent_dim=128, initial_features=64):
        super(ConvEncoder, self).__init__()
        self.initial_features = initial_features

        # Following the structure from the Keras model
        # Starting with 32x32 image input (CIFAR-10)
        self.main = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8x8 -> 8x8 (no pooling)
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the flattened feature size
        # After 3 max pooling layers: 32/2/2/2 = 4, so feature maps are 4x4
        self.feature_size = 8 * 4 * 4

        # Project to latent vector
        self.fc = nn.Linear(self.feature_size, latent_dim)

    def forward(self, x):
        features = self.main(x)
        features = features.view(-1, self.feature_size)
        latent = self.fc(features)
        return latent

class ConvDecoder(nn.Module):
    """
    Convolutional Decoder that follows the structure from the original Keras model:
    Conv2D(8) -> Upsample -> Conv2D(16) -> Upsample -> Conv2D(32) -> Upsample -> Conv2D(64) -> Conv2D(3)
    """
    def __init__(self, latent_dim=128, out_channels=3, initial_features=64):
        super(ConvDecoder, self).__init__()
        self.initial_features = initial_features

        # Calculate the starting feature map size
        # After 3 max pooling layers: 32/2/2/2 = 4, so feature maps start at 4x4
        self.feature_size = 8 * 4 * 4

        # Project latent vector to initial feature map
        self.fc = nn.Linear(latent_dim, self.feature_size)

        # Following the structure from the Keras model
        self.main = nn.Sequential(
            # Block 1: Maintain 4x4 size
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),

            # Block 2: 4x4 -> 8x8
            nn.ConvTranspose2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),

            # Block 3: 8x8 -> 16x16
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),

            # Block 4: 16x16 -> 32x32
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),

            # Final layer: maintain 32x32 size
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, z):
        # Reshape latent vector to initial feature map
        x = self.fc(z)
        x = x.view(-1, 8, 4, 4)  # Reshape to (batch_size, 8, 4, 4)
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
    A convolutional autoencoder model that includes an encoder, decoder, and classifier.
    This implementation follows the structure of the original Keras model while integrating
    with the provided class structure.

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
