import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label,binary_dilation
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
path = os.path.dirname(__file__)

def periodic_binary_blobs(shape=(128, 128), blob_density=0.5, sigma=2.0, seed=3):
    """Generate a periodic binary image using filtered noise."""
    rng = np.random.default_rng(seed)
    noise = rng.random(shape)
    smooth_noise = gaussian_filter(noise, sigma=sigma, mode='wrap')
    threshold = np.percentile(smooth_noise, (1.0 - blob_density) * 100)
    return smooth_noise > threshold  # True = solid, False = fluid

def label_fluid(img):
    """Label fluid regions (where img == False) with 8-connectivity."""
    structure = np.ones((3, 3), dtype=int)
    return label(np.logical_not(img), structure=structure)

def find_percolating_clusters(img):
    """Identify clusters that percolate in x and y directions (with periodic boundary)."""
    dilated = binary_dilation(img)
    labeled, num_features = label_fluid(dilated)
    h, w = labeled.shape

    percolates_x = False
    percolates_y = False

    # Check top-bottom (x-direction)
    for j in range(w):
        if labeled[0, j] != 0 and labeled[0, j] == labeled[-1, j]:
            percolates_x = True
            break

    # Check left-right (y-direction)
    for i in range(h):
        if labeled[i, 0] != 0 and labeled[i, 0] == labeled[i, -1]:
            percolates_y = True
            break

    return percolates_x, percolates_y, labeled

def fill_non_percolating(img):
    """Fill in non-percolating fluid regions to simulate disconnection."""
    labels = label_fluid(img)[0]
    top, bottom = set(labels[0, :]) - {0}, set(labels[-1, :]) - {0}
    left, right = set(labels[:, 0]) - {0}, set(labels[:, -1]) - {0}
    percolating = (top & bottom) | (left & right)

    non_perc_mask = (labels > 0) & ~np.isin(labels, list(percolating))
    img_filled = img.copy()
    img_filled[non_perc_mask] = True
    return img_filled

def make_cross_channels(shape=(64, 64), channel_width=3):
    """Creates two non-overlapping diagonal fluid channels."""
    img = np.full(shape, True, dtype=bool)  # True = solid
    h, w = shape
    cw = channel_width

    # left to top channel
    for i in range(h // 2):
        x, y = i, h // 2 - i - 1
        if 0 <= y < h and 0 <= x < w:
            img[max(y - cw // 2, 0):y + cw // 2 + 1,
                max(x - cw // 2, 0):x + cw // 2 + 1] = False

    # bottom to right channel
    for i in range(h // 2):
        x, y = w - h // 2 + i, h - i - 1
        if 0 <= x < w and 0 <= y < h:
            img[max(y - cw // 2, 0):y + cw // 2 + 1,
                max(x - cw // 2, 0):x + cw // 2 + 1] = False
    return img

def plot_percolation(img, labeled, filled, px, py):
    """Plot original, labeled, and filled images with percolation info."""
    num_labels = labeled.max()
    jet_colors = plt.cm.jet(np.linspace(0, 1, num_labels))
    colors = np.vstack(([1, 1, 1, 1], jet_colors))  # white for label 0 (solid region)
    custom_cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 1.5), len(colors))

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(labeled, cmap=custom_cmap, norm=norm)
    plt.title("Clusters")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(filled, cmap='gray')
    plt.title(f"Percolates: x: {px}, y: {py}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(path,"percolation_check_demo.pdf"))
    

# Example usage
if __name__ == "__main__":
    # img = periodic_binary_blobs(blob_density=0.5, sigma=2.0, shape=(128, 128), seed=9)
    img = make_cross_channels(shape=(128, 128), channel_width=30)

    px, py, labeled = find_percolating_clusters(img)
    filled = fill_non_percolating(img)
    print("Percolates:", px and py)

    plot_percolation(img, labeled, filled, px, py)
