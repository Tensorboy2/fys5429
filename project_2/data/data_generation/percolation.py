import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label,binary_dilation
from matplotlib.colors import ListedColormap, BoundaryNorm
from periodic_labeling import label_fluid_periodic
import os
path = os.path.dirname(__file__)

def periodic_binary_blobs(shape=(128, 128), blob_density=0.5, sigma=2.0, seed=3):
    """Generate a periodic binary image using filtered noise."""
    rng = np.random.default_rng(seed)
    noise = rng.random(shape)
    smooth_noise = gaussian_filter(noise, sigma=sigma, mode='wrap')
    threshold = np.percentile(smooth_noise, (1.0 - blob_density) * 100)
    return smooth_noise > threshold  # True = solid, False = fluid

def periodic_span(coords, size):
    """
    Calculates the effective span of a set of coordinates along one dimension,
    considering periodic boundary conditions.
    `coords`: 1D array of coordinates (e.g., all x-coordinates of a cluster).
    `size`: The size of the domain in that dimension (e.g., width or height).
    Returns the length of the domain covered by the cluster in that dimension,
    accounting for periodicity. A value >= size-1 typically indicates spanning.
    """
    if coords.size == 0:
        return 0
    
    # Sort and get unique coordinates to correctly calculate differences
    unique_sorted_coords = np.unique(coords) # np.sort is implicit in np.unique
    
    if unique_sorted_coords.size == 0: # Should be caught by coords.size == 0
        return 0

    diffs = np.diff(np.concatenate([unique_sorted_coords, [unique_sorted_coords[0] + size]]))
    max_gap = np.max(diffs) if diffs.size > 0 else size
    if unique_sorted_coords.size == 1: 
        max_gap = size

    return size - max_gap

def check_periodic_percolation(labeled_img):
    """
    Checks if any cluster in the labeled image percolates across periodic boundaries.
    `labeled_img`: Image where each fluid cluster has a unique positive integer label,
                   and solid regions are 0. This labeling MUST correctly handle
                   periodic boundary conditions.
    Returns:
        percolates_x (bool): True if any cluster percolates in the x-direction.
        percolates_y (bool): True if any cluster percolates in the y-direction.
    """
    if labeled_img is None or labeled_img.size == 0:
        return False, False
        
    h, w = labeled_img.shape
    percolates_x = False
    percolates_y = False

    if not np.issubdtype(labeled_img.dtype, np.integer):
        raise TypeError("labeled_img must be an integer type array.")

    unique_labels = np.unique(labeled_img)

    for label_id in unique_labels:
        if label_id == 1:  # Skip background/solid
            continue # CRITICAL FIX: Was 'break', which stopped checking after solid.

        ys, xs = np.where(labeled_img == label_id)

        if xs.size == 0: # Should not happen if label_id > 0 from unique() on a valid labeled_img
            continue

        # Check percolation in x-direction if not already found
        if not percolates_x:
            span_x = periodic_span(xs, w)
            # Condition: span_x >= w-1 means max_gap <= 1. Cluster is continuous.
            if span_x >= w - 1: 
                percolates_x = True

        # Check percolation in y-direction if not already found
        if not percolates_y:
            span_y = periodic_span(ys, h)
            if span_y >= h - 1:
                percolates_y = True
        
        if percolates_x and percolates_y:
            break  # Found a cluster that percolates in both X and Y

    return percolates_x, percolates_y

def fill_non_percolating_periodic(img, labeled_img, overall_px, overall_py):
    h, w = img.shape
    img_filled = img.copy() # Start with original, True=solid

    unique_labels_in_labeled_img = np.unique(labeled_img)
    
    # Identify which specific labels are responsible for the detected overall percolation
    contributing_labels = set()

    if overall_px or overall_py: # Only find contributing labels if there's percolation
        for lbl_id in unique_labels_in_labeled_img:
            if lbl_id == 0:
                continue
            
            ys_cluster, xs_cluster = np.where(labeled_img == lbl_id)
            if xs_cluster.size == 0:
                continue

            # Does this specific cluster percolate in X?
            if overall_px:
                cluster_span_x = periodic_span(xs_cluster, w)
                if cluster_span_x >= w - 1:
                    contributing_labels.add(lbl_id)
            
            # Does this specific cluster percolate in Y?
            # (A label can contribute to both X and Y percolation)
            if overall_py and lbl_id not in contributing_labels: # Avoid re-check if already added by X
                cluster_span_y = periodic_span(ys_cluster, h)
                if cluster_span_y >= h - 1:
                    contributing_labels.add(lbl_id)
            elif overall_py and lbl_id in contributing_labels: 
                pass 

    mask_to_fill = (img != 1) & (~np.isin(labeled_img, list(contributing_labels)))
    
    img_filled[mask_to_fill] = True 
            
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

def make_periodic_edge_cross(shape=(64, 64), channel_width=3):
    """
    Create diagonal channels that touch opposite edges but do not percolate.
    These are connected via periodicity but do not form a spanning path.
    """
    img = np.full(shape, True, dtype=bool)  # True = solid
    h, w = shape
    cw = channel_width

    # left to top channel
    for i in range(h // 4):
        x, y = i, h // 4 - i - 1
        if 0 <= y < h and 0 <= x < w:
            img[max(y - cw // 2, 0):y + cw // 2 + 1,
                max(x - cw // 2, 0):x + cw // 2 + 1] = False
    # left to bottom channel
    for i in range(h // 4):
        x, y = i, 3*h//4+ i - 1
        if 0 <= y < h and 0 <= x < w:
            img[max(y - cw // 2, 0):y + cw // 2 + 1,
                max(x - cw // 2, 0):x + cw // 2 + 1] = False

    # bottom to right channel
    for i in range(h // 4):
        x, y = w - h // 4 + i, h - i - 1
        if 0 <= x < w and 0 <= y < h:
            img[max(y - cw // 2, 0):y + cw // 2 + 1,
                max(x - cw // 2, 0):x + cw // 2 + 1] = False
            
    # top to right channel
    for i in range(h // 4):
        x, y = w - h//4 + i,  i - 1
        if 0 <= y < h and 0 <= x < w:
            img[max(y - cw // 2, 0):y + cw // 2 + 1,
                max(x - cw // 2, 0):x + cw // 2 + 1] = False
    return img


def plot_percolation(img, labeled, filled, px, py,name=""):
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
    plt.imshow(labeled, cmap="viridis")
    plt.title("Clusters")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(filled, cmap='gray')
    plt.title(f"Percolates: x: {px}, y: {py}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(path,f"percolation_check_demo{name}.pdf"))
    

from test import big_LBM

# Example usage
if __name__ == "__main__":
    # img = np.array([
    #     [1, 0, 1, 0, 1],
    #     [0, 1, 1, 1, 0],
    #     [1, 1, 1, 1, 1],
    #     [0, 1, 1, 1, 0],
    #     [1, 0, 1, 0, 1]
    # ])
    # labeled = label_fluid_periodic(img)
    # print(labeled)
    # px, py = check_periodic_percolation(labeled)
    # print("Percolates X:", px, "Percolates Y:", py)

    # img = np.array([
    #     [1, 1, 0, 1, 1],
    #     [1, 0, 1, 1, 1],
    #     [0, 1, 1, 1, 0],
    #     [1, 1, 1, 0, 1],
    #     [1, 1, 0, 1, 1]
    # ])
    # labeled = label_fluid_periodic(img)
    # print(labeled)
    # px, py = check_periodic_percolation(labeled)
    # print("Percolates X:", px, "Percolates Y:", py)

    # img = np.array([
    #     [1, 0, 1, 1, 1],
    #     [0, 0, 1, 0, 0],
    #     [0, 1, 1, 1, 0],
    #     [1, 1, 1, 0, 1],
    #     [1, 0, 0, 1, 1]
    # ])
    # labeled = label_fluid_periodic(img)
    # print(labeled)
    # px, py = check_periodic_percolation(labeled)
    # print("Percolates X:", px, "Percolates Y:", py)

    
    # img = make_periodic_edge_cross(shape=(128, 128), channel_width=10)
    # labeled = label_fluid_periodic(img)
    # px, py = check_periodic_percolation(labeled) 
    # filled = fill_non_percolating_periodic(img,labeled,px,py)
    # print("Percolates:", px and py)
    # plot_percolation(img, labeled, filled, px, py,name="_edge_cross")
    
    # img = make_cross_channels(shape=(128, 128), channel_width=10)
    # labeled = label_fluid_periodic(img)
    # px, py = check_periodic_percolation(labeled)
    # print("Percolates X:", px, "Percolates Y:", py)
    # filled = fill_non_percolating_periodic(img,labeled,px,py)
    # print("Percolates:", px and py)
    # plot_percolation(img, labeled, filled, px, py,name="_cross")

    img = periodic_binary_blobs(blob_density=0.5, sigma=4.0, shape=(128, 128), seed=9)

    # img = binary_dilation(img)

    ux, k_xx, k_xy, t = big_LBM(img,T=10_000, force_dir=0)
    print(f"ran for {t} iterations")
    plt.subplot(1,2,1)
    plt.imshow(np.sqrt(ux[:,:,0]**2 + ux[:,:,1]**2))
    plt.colorbar()
    ux, k_xx, k_xy, t = big_LBM(img,T=10_000, force_dir=1)
    print(f"ran for {t} iterations")
    plt.subplot(1,2,2)
    plt.imshow(np.sqrt(ux[:,:,0]**2 + ux[:,:,1]**2))
    plt.colorbar()

    plt.show()


    # labeled = label_fluid_periodic(img)
    # px, py = check_periodic_percolation(labeled)
    # print("Percolates X:", px, "Percolates Y:", py)
    # filled = fill_non_percolating_periodic(img,labeled,px,py)
    # print("Percolates:", px and py)
    # plot_percolation(img, labeled, filled, px, py,name="_full_media")