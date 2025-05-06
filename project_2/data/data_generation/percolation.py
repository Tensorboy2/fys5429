import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label

def periodic_binary_blobs(shape=(128, 128), blob_density=0.5, sigma=2.0, seed=3):
    rng = np.random.default_rng(seed)
    noise = rng.random(shape)
    smooth_noise = gaussian_filter(noise, sigma=sigma, mode='wrap')
    threshold = np.percentile(smooth_noise, (1.0 - blob_density) * 100)
    blobs = smooth_noise > threshold
    return blobs

def label_fluid(img):
    structure = np.ones((3, 3), dtype=int)
    fluid = np.logical_not(img)
    labels, _ = label(fluid, structure=structure)
    return labels

def check_percolation(img, show=False):
    labels = label_fluid(img)

    if show:
        plt.imshow(labels, cmap='nipy_spectral')
        plt.title("Labeled fluid regions")
        plt.colorbar()
        plt.show()

    # Get labels on edges
    top_labels = set(labels[0, :]) - {0}
    bottom_labels = set(labels[-1, :]) - {0}
    left_labels = set(labels[:, 0]) - {0}
    right_labels = set(labels[:, -1]) - {0}

    # Look for any label that exists both on top and bottom or left and right
    vertical_percolates = bool(top_labels & bottom_labels)
    horizontal_percolates = bool(left_labels & right_labels)

    return vertical_percolates or horizontal_percolates

def fill_non_percolating(img):
    labels = label_fluid(img)
    top_labels = set(labels[0, :]) - {0}
    bottom_labels = set(labels[-1, :]) - {0}
    left_labels = set(labels[:, 0]) - {0}
    right_labels = set(labels[:, -1]) - {0}

    percolating = (top_labels & bottom_labels) | (left_labels & right_labels)

    mask = (labels > 0) & ~np.isin(labels, list(percolating))
    img_filled = img.copy()
    img_filled[mask] = True  # Fill in non-percolating fluid

    return img_filled

def preprocess_image(img, show=False):
    percolates = check_percolation(img, show=show)
    filled_img = fill_non_percolating(img)
    return filled_img, percolates

def find_percolating_clusters(img):
    """Check for percolating clusters in x and y with periodic boundaries."""
    N = img.shape[0]
    structure = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]])  # 8-connectivity
    grid = np.logical_not(img)
    labeled, num_features = label(grid, structure=structure)
    
    # Find percolating clusters
    percolates_x = False
    percolates_y = False
    for cluster_id in range(1, num_features+1):
        coords = np.argwhere(labeled == cluster_id)
        xs = coords[:, 0]
        ys = coords[:, 1]

        # Check for wrap-around
        if (0 in xs and N-1 in xs):
            percolates_x = True
        if (0 in ys and N-1 in ys):
            percolates_y = True
    
    return percolates_x, percolates_y, labeled
def make_cross_channels(shape=(64, 64), channel_width=3):
    """
    Creates a binary image with:
    - One channel from left to top (↖)
    - One channel from bottom to right (↗)
    Channels do not overlap.
    
    Returns:
    - binary image: True = solid, False = void
    """
    img = np.ones(shape, dtype=bool)  # All solid

    h, w = shape
    cw = channel_width

    # Diagonal from left to top
    for i in range(h//2):
        x = 0 + i
        y = h//2 - i - 1
        if y < 0 or x >= w:
            break
        img[max(y - cw//2, 0):min(y + cw//2 + 1, h),
            max(x - cw//2, 0):min(x + cw//2 + 1, w)] = False

    # Diagonal from bottom to right
    for i in range(h//2):
        x = w - h//2 + i
        y = h - i - 1
        if x >= w or y < 0:
            break
        img[max(y - cw//2, 0):min(y + cw//2 + 1, h),
            max(x - cw//2, 0):min(x + cw//2 + 1, w)] = False

    return img
from matplotlib.colors import ListedColormap, BoundaryNorm
# Example
if __name__ == "__main__":
    # img = periodic_binary_blobs(blob_density=0.9, shape=(128, 128),seed=3)
    img = make_cross_channels(shape=(128,128),channel_width=30)
    px,py,labeled = find_percolating_clusters(img)
    filled = fill_non_percolating(img)
    # filled, percolates = preprocess_image(img, show=True)
    print("Percolates:", px and py)

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title("original image")
    plt.imshow(img,cmap='gray')


    plt.subplot(1,3,2)
    num_labels = labeled.max()
    jet_colors = plt.cm.jet(np.linspace(0, 1, num_labels))
    colors = np.vstack(([1, 1, 1, 1], jet_colors))  # Add white for 0
    custom_cmap = ListedColormap(colors)

    # Normalize to match the labels (0 to num_labels)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 1.5), len(colors))

    plt.imshow(labeled,cmap=custom_cmap, norm=norm)
    plt.title("clusters")


    plt.subplot(1,3,3)
    plt.imshow(filled)
    plt.title(f"Percolates: x: {px}, y: {py}")
    plt.tight_layout()
    plt.show()
