import torch
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(model, dataloader, device):
    '''
    model - torch.nn.Module subclass. This is your encoder model
    dataloader - test dataloader to over over data for which you wish to compute projections
    device - cuda or cpu (as a string)
    '''
    model.eval()

    images_list = []
    labels_list = []
    latent_list = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            #approximate the latent space from data
            latent_vector = model(images)

            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)

    # Plot TSNE for latent space
    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)  # Smaller points
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space')
    plt.savefig('latent_tsne.png')
    plt.close()

    #plot image domain tsne
    tsne_image = TSNE(n_components=2, random_state=42)
    images_flattened = images.reshape(images.shape[0], -1)
    image_tsne = tsne_image.fit_transform(images_flattened)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Image Space')
    plt.savefig('image_tsne.png')
    plt.close()


# Handle Checkpoint Saving and Loading
def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, accuracies=None, filename='checkpoint.pth'):
    """
    Save model checkpoint to file

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        train_losses: List of training losses
        val_losses: List of validation losses
        accuracies: List of validation accuracies (if available)
        filename: File path to save checkpoint
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }

    if accuracies is not None:
        state['accuracies'] = accuracies

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    # Save the checkpoint
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """
    Load model from checkpoint file

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        filename: File path to load checkpoint from

    Returns:
        epoch: Last epoch number
        train_losses: List of training losses
        val_losses: List of validation losses
        accuracies: List of validation accuracies (if available)
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return 0, [], [], []

    # Load checkpoint
    checkpoint = torch.load(filename)

    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    accuracies = checkpoint.get('accuracies', [])

    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, train_losses, val_losses, accuracies


def plot_training_curves(train_values, val_values, title, filename, xlabel='Epoch', ylabel='Loss'):
    """
    Plot training curves and save the figure.

    Args:
        train_values: List of training values (loss or accuracy)
        val_values: List of validation values (loss or accuracy)
        title: Plot title
        filename: Filename to save the plot
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_values, label='Training')
    plt.plot(val_values, label='Validation')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

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
