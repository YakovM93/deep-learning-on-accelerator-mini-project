import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A basic residual block with two 3x3 convolutional layers.
    If the input and output channels differ or if downsampling is needed, a
    skip convolution is applied.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolution changes resolution if stride > 1
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.downsample = None
        # If input and output dimensions differ, adapt skip connection.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class CustomResNetEncoder(nn.Module):
    """
    An encoder built from scratch following a ResNet-style design.
    It processes an input image (N, C, H, W) and outputs a latent vector.
    Architecture:
      - Stem: 7x7 conv with stride 2 and max pooling.
      - Residual stage 1: one residual block, channels=64.
      - Residual stage 2: one residual block with downsampling, channels=128.
      - Residual stage 3: one residual block with downsampling, channels=256.
      - Global average pooling, then a fully-connected layer to latent_dim.
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(CustomResNetEncoder, self).__init__()
        self.stem_conv = nn.Conv2d(in_channels, 64, kernel_size=4,
                                   stride=2, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)

        # Global Average Pooling and final FC to latent vector.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x shape: (N, C, H, W)
        out = self.stem_conv(x)
        out = self.stem_bn(out)
        out = self.stem_relu(out)
        out = self.stem_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)  # shape: (N, 256, 1, 1)
        out = torch.flatten(out, 1)  # shape: (N, 256)
        latent = self.fc(out)         # shape: (N, latent_dim)
        return latent

class MirrorResNetDecoder(nn.Module):
    """
    A decoder built as the mirror image of the encoder.
    Rather than using only fully-connected layers on flattened vectors, this decoder
    first projects the latent vector to a small spatial feature map (here 256 x 4 x 4),
    and then uses a series of transposed convolutions to upsample to the original resolution.

    Note: This implementation assumes that input images are 32x32. For different input sizes,
    adjust the projection size and upsampling layers accordingly.
    """
    def __init__(self, latent_dim=128, out_channels=3):
        super(MirrorResNetDecoder, self).__init__()
        # Project latent vector to 256 feature maps of spatial size 4x4.
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4,
                                          stride=2, padding=1, bias=False)  # 4x4 -> 8x8
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4,
                                          stride=2, padding=1, bias=False)   # 8x8 -> 16x16
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, out_channels, kernel_size=4,
                                          stride=2, padding=1, bias=False)   # 16x16 -> 32x32

        # Use Tanh so that outputs are in range [-1, 1]. (You may change to Sigmoid if needed.)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z shape: (N, latent_dim)
        out = self.fc(z)                     # (N, 256*4*4)
        out = out.view(-1, 256, 4, 4)          # (N, 256, 4, 4)
        out = F.relu(self.bn1(self.deconv1(out)))  # (N, 128, 8, 8)
        out = F.relu(self.bn2(self.deconv2(out)))  # (N, 64, 16, 16)
        out = self.deconv3(out)              # (N, out_channels, 32, 32)
        out = self.tanh(out)
        return out

class CustomClassifier(nn.Module):
    """
    Simple classifier that takes a latent vector as input and outputs logits for num_classes.
    """
    def __init__(self, latent_dim=128, num_classes=10, hidden_dims=None):
        super(CustomClassifier, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64]
        dims = [latent_dim] + hidden_dims

        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        # Final layer: no activation (logits)
        layers.append(nn.Linear(dims[-1], num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)

class ResNetBasedLatentModel(nn.Module):
    """
    A latent model built completely from scratch that uses a ResNet-style encoder,
    a mirror decoder, and a classifier.

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
    def __init__(self, in_channels=3, latent_dim=128, num_classes=10):
        super(ResNetBasedLatentModel, self).__init__()
        self.encoder = CustomResNetEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = MirrorResNetDecoder(latent_dim=latent_dim, out_channels=in_channels)
        self.classifier = CustomClassifier(latent_dim=latent_dim, num_classes=num_classes)
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
