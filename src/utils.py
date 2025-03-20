import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    plt.close()
