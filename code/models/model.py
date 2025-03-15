import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Encoder(nn.Module):
    """
    Encoder network that compresses input images to a latent representation.
    """
    def __init__(self, input_dim, latent_dim=128, hidden_dims=None):
        super(Encoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Build encoder layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        
        # Final layer to latent space (no activation)
        layers.append(nn.Linear(dims[-1], latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        
    def forward(self, x):
        # Flatten the input if it's not already flattened
        if len(x.shape) > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        # Forward pass through network
        latent = self.encoder(x)
        return latent

class Decoder(nn.Module):
    """
    Decoder network that reconstructs images from latent representations.
    """
    def __init__(self, latent_dim, output_dim, hidden_dims=None):
        super(Decoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512]
        
        # Build decoder layers
        layers = []
        dims = [latent_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        
        # Final layer to output space with sigmoid activation for image pixels
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        # Forward pass through network
        reconstructed = self.decoder(x)
        return reconstructed

class Classifier(nn.Module):
    """
    Classifier network that takes latent representations and classifies them.
    """
    def __init__(self, input_dim, num_classes=10, hidden_dims=None):
        super(Classifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64]
        
        # Build classifier layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        
        # Final layer to class probabilities (no activation, will use cross-entropy loss)
        layers.append(nn.Linear(dims[-1], num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)

class LatentModel(nn.Module):
    """
    Flexible model that can be configured for different approaches:
    1. Self-supervised autoencoding
    2. Classification
    3. Both simultaneously
    """
    def __init__(self, input_dim, latent_dim=128, num_classes=10, 
                encoder_hidden_dims=None, decoder_hidden_dims=None, 
                classifier_hidden_dims=None):
        super(LatentModel, self).__init__()
        
        # Create model components
        self.encoder = Encoder(input_dim, latent_dim, encoder_hidden_dims)
        self.decoder = Decoder(latent_dim, input_dim, decoder_hidden_dims)
        self.classifier = Classifier(latent_dim, num_classes, classifier_hidden_dims)
        
        # Store dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
    def forward(self, x, mode='all'):
        """
        Forward pass with different modes:
        - 'all': Return reconstruction, classification, and latent
        - 'autoencoder': Return only reconstruction and latent
        - 'classifier': Return only classification
        - 'latent': Return only latent representation
        """
        # Save original shape for potential reshaping
        original_shape = x.shape
        
        # Flatten input if needed
        if len(x.shape) > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        # Get latent representation
        latent = self.encoder(x)
        
        # Return based on mode
        if mode == 'all':
            reconstruction = self.decoder(latent)
            classification = self.classifier(latent)
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
        """Freeze or unfreeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
    
    def freeze_decoder(self, freeze=True):
        """Freeze or unfreeze decoder parameters"""
        for param in self.decoder.parameters():
            param.requires_grad = not freeze
    
    def freeze_classifier(self, freeze=True):
        """Freeze or unfreeze classifier parameters"""
        for param in self.classifier.parameters():
            param.requires_grad = not freeze

def train_self_supervised(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    """
    Train the model using self-supervised reconstruction loss.
    
    Args:
        model: LatentModel
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    
    # Set model to only train encoder and decoder
    model.freeze_classifier(True)
    model.freeze_encoder(False)
    model.freeze_decoder(False)
    
    optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), 
        lr=lr
    )
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            
            # Forward pass (autoencoder mode)
            reconstructed, _ = model(data, mode='autoencoder')
            
            # Flatten the input if needed
            if len(data.shape) > 2:
                batch_size = data.size(0)
                data = data.view(batch_size, -1)
            
            # Calculate loss and backpropagate
            loss = criterion(reconstructed, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                
                # Forward pass (autoencoder mode)
                reconstructed, _ = model(data, mode='autoencoder')
                
                # Flatten the input if needed
                if len(data.shape) > 2:
                    batch_size = data.size(0)
                    data = data.view(batch_size, -1)
                
                # Calculate loss
                loss = criterion(reconstructed, data)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Autoencoder Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('autoencoder_training.png')
    plt.close()
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def train_classifier_with_frozen_encoder(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    """
    Train the classifier using the frozen encoder.
    
    Args:
        model: LatentModel with pre-trained encoder
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    
    # Freeze encoder, train only the classifier
    model.freeze_encoder(True)
    model.freeze_decoder(True)
    model.freeze_classifier(False)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass (classifier mode)
            outputs = model(data, mode='classifier')
            
            # Calculate loss and backpropagate
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                # Forward pass (classifier mode)
                outputs = model(data, mode='classifier')
                
                # Calculate loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.2f}%")
    
    # Plot training history - Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Classifier Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('classifier_training_loss.png')
    plt.close()
    
    # Plot training history - Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Classifier Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('classifier_training_accuracy.png')
    plt.close()
    
    return model, {
        'train_losses': train_losses, 
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def evaluate(model, test_loader, device):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        Test accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass (classifier mode)
            outputs = model(data, mode='classifier')
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy

def visualize_reconstructions(model, test_loader, device, num_examples=10):
    """
    Visualize original images and their reconstructions.
    
    Args:
        model: LatentModel
        test_loader: DataLoader for test data
        device: Device to use ('cuda' or 'cpu')
        num_examples: Number of examples to visualize
    """
    model.eval()
    
    # Get some test examples
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images[:num_examples].to(device)
    
    # Get reconstructions
    with torch.no_grad():
        reconstructed, _ = model(images, mode='autoencoder')
    
    # Convert to numpy for plotting
    images = images.cpu().numpy()
    
    # Reshape reconstructed images
    if len(images.shape) > 2:
        # For image datasets like MNIST or CIFAR
        reconstructed = reconstructed.view(images.shape).cpu().numpy()
    else:
        # For flat data
        reconstructed = reconstructed.cpu().numpy().reshape(images.shape)
    
    # Plot original and reconstructed images
    plt.figure(figsize=(20, 4))
    for i in range(num_examples):
        # Original
        ax = plt.subplot(2, num_examples, i + 1)
        if images.shape[1] == 1:  # Grayscale (e.g., MNIST)
            plt.imshow(images[i][0], cmap='gray')
        else:  # RGB (e.g., CIFAR)
            plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.title(f"Original: {labels[i]}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Reconstructed
        ax = plt.subplot(2, num_examples, num_examples + i + 1)
        if images.shape[1] == 1:  # Grayscale
            plt.imshow(reconstructed[i][0], cmap='gray')
        else:  # RGB
            plt.imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        plt.title(f"Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.suptitle('Autoencoder Reconstructions', fontsize=16)
    plt.tight_layout()
    plt.savefig('reconstructions.png')
    plt.close()