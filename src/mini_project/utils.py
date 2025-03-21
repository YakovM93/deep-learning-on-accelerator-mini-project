import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Plotting Functions:



def plot_tsne(encode_fn, dataloader, device,
              image_tsne_path='image_tsne.png',
              latent_tsne_path='latent_tsne.png'):

    """
    encode_fn - A function that takes in a batch of images and returns
                their latent embeddings (for example: autoencoder.encode).
    dataloader - Dataloader over which to compute and plot t-SNE.
    device - 'cpu' or 'cuda'
    image_tsne_path - File path for saving the t-SNE in image-space.
    latent_tsne_path - File path for saving the t-SNE in latent-space.
    """
    images_list = []
    labels_list = []
    latent_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            z = encode_fn(images)
            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(z.cpu().numpy())

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)

    # t-SNE on latent vectors
    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space')
    plt.grid(True)
    plt.savefig(latent_tsne_path)
    print(f"Saved t-SNE visualization to {os.path.abspath(latent_tsne_path)}")
    plt.close()

    # t-SNE on images (flattened)
    images_flattened = images.reshape(images.shape[0], -1)
    tsne_image = TSNE(n_components=2, random_state=42)
    image_tsne = tsne_image.fit_transform(images_flattened)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Image Space')
    plt.grid(True)
    plt.savefig(image_tsne_path)
    print(f"Saved t-SNE visualization to {os.path.abspath(image_tsne_path)}")
    plt.close()

def visualize_reconstructions(model, test_loader, device, num_examples=10,save_path='reconstructions.png'):
    """
    Visualize original images and their reconstructions.

    Args:
        model: LatentModel
        test_loader: DataLoader for test data
        device: Device to use ('cuda' or 'cpu')
        num_examples: Number of examples to visualize
        save_path: Path to save the visualization
    """
    model.eval()

    # Get some test examples
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images[:num_examples].to(device)

    # Get reconstructions
    with torch.no_grad():
        reconstructed = model(images)

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
    plt.savefig(save_path)
    print(f"Saved reconstruction visualization to {os.path.abspath(save_path)}")
    plt.close()

def linear_interpolation(model, dataloader, device, steps, n_image_pairs, save_path):
    """
    Perform linear interpolation between pairs of images in the latent space.

    Args:
        model: The autoencoder model
        dataloader: DataLoader containing images
        device: Device to use (cuda or cpu)
        steps: Number of interpolation steps between images
        n_image_pairs: Number of image pairs to interpolate
        save_path: Path to save the visualization (if None, will show the plot)
    """
    print("starting linear interpolation")
    model.eval()
    images, labels = next(iter(dataloader))

    if len(images) < 2*n_image_pairs:
        print(f"Not enough images for {n_image_pairs} pairs. Need at least {2*n_image_pairs} images.")
        return

    fig, all_axes = plt.subplots(n_image_pairs, steps+2, figsize=(3*(steps+2), 3*n_image_pairs))

    for pair_idx in range(n_image_pairs):
        # Get two images
        idx1, idx2 = pair_idx*2, pair_idx*2+1
        img1 = images[idx1].unsqueeze(0).to(device)
        img2 = images[idx2].unsqueeze(0).to(device)
        label1, label2 = labels[idx1].item(), labels[idx2].item()

        with torch.no_grad():
            z1 = model.encode(img1)
            z2 = model.encode(img2)

        # Create interpolation steps
        alphas = np.linspace(0, 1, steps)
        interpolated_images = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            with torch.no_grad():
                x_interp = model.decode(z_interp)
            interpolated_images.append(x_interp.squeeze(0).cpu())

        # Get axes for this pair
        if n_image_pairs == 1:
            axes = all_axes
        else:
            axes = all_axes[pair_idx]

        # Plot the original images at the beginning and end
        img1_np = img1.squeeze(0).cpu()
        img2_np = img2.squeeze(0).cpu()

        if img1_np.shape[0] == 1:  # Grayscale
            axes[0].imshow(img1_np.squeeze(), cmap="gray")
            axes[-1].imshow(img2_np.squeeze(), cmap="gray")
        else:  # RGB
            axes[0].imshow(np.transpose(img1_np, (1, 2, 0)))
            axes[-1].imshow(np.transpose(img2_np, (1, 2, 0)))

        axes[0].set_title(f"Original\nClass: {label1}")
        axes[-1].set_title(f"Original\nClass: {label2}")
        axes[0].axis("off")
        axes[-1].axis("off")

        # Plot the interpolated images
        for i, img in enumerate(interpolated_images):
            if img.shape[0] == 1:  # Grayscale
                axes[i+1].imshow(img.squeeze(), cmap="gray")
            else:  # RGB
                axes[i+1].imshow(np.transpose(img, (1, 2, 0)))
            axes[i+1].set_title(f"α={alphas[i]:.2f}")
            axes[i+1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved reconstruction visualization to {os.path.abspath(save_path)}")
