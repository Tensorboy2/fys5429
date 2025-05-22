import numpy as np
import os
path = os.path.dirname(__file__)
from numba import njit

@njit(fastmath=True)
def big_LBM(solid, T, force_dir):
    """
    A Lattice Boltzmann simulation.
    This single function inlines equilibrium, collision, streaming,
    bounce-back, and macroscopic updates to optimize njit performance.
    
    # Parameters:
    - solid : 2D numpy array (Nx, Ny) of type bool
            True indicates a solid node.
    - T     : int
            Number of simulation time steps.
              
    # Returns:
    - u            : 3D numpy array (Nx, Ny, 2) velocity field.
    - k            : Permeability calculated from velocity field and density.
    """
    Nx = solid.shape[0]
    Ny = solid.shape[1]
    
    # Create fluid mask: fluid = not solid:
    fluid = np.empty((Nx, Ny), dtype=np.bool_)
    for x in range(Nx):
        for y in range(Ny):
            if solid[x, y]:
                fluid[x, y] = False
            else:
                fluid[x, y] = True
    
    # Lattice vectors (9 directions):
    c = np.empty((9, 2), dtype=np.int64)
    c[0, 0] =  0; c[0, 1] =  0
    c[1, 0] =  1; c[1, 1] =  0
    c[2, 0] =  0; c[2, 1] =  1
    c[3, 0] = -1; c[3, 1] =  0
    c[4, 0] =  0; c[4, 1] = -1
    c[5, 0] =  1; c[5, 1] =  1
    c[6, 0] = -1; c[6, 1] =  1
    c[7, 0] = -1; c[7, 1] = -1
    c[8, 0] =  1; c[8, 1] = -1

    # Lattice weights:
    w = np.empty(9, dtype=np.float64)
    w[0] = 4.0/9.0
    w[1] = 1.0/9.0
    w[2] = 1.0/9.0
    w[3] = 1.0/9.0
    w[4] = 1.0/9.0
    w[5] = 1.0/36.0
    w[6] = 1.0/36.0
    w[7] = 1.0/36.0
    w[8] = 1.0/36.0

    # Bounce-back mapping:
    bounce_back_pairs = np.empty(9, dtype=np.int64)
    bounce_back_pairs[0] = 0
    bounce_back_pairs[1] = 3
    bounce_back_pairs[2] = 4
    bounce_back_pairs[3] = 1
    bounce_back_pairs[4] = 2
    bounce_back_pairs[5] = 7
    bounce_back_pairs[6] = 8
    bounce_back_pairs[7] = 5
    bounce_back_pairs[8] = 6

    # Initialize macroscopic variables:
    rho = np.empty((Nx, Ny), dtype=np.float64)
    u   = np.zeros((Nx, Ny, 2), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x,y]:
                rho[x, y] = 1.0  # initial density

    # Gravity and forcing term:
    grav = 0.00001
    F = np.zeros((Nx, Ny, 2), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            F[x, y, force_dir] = -grav  # gravity in x-direction (adjust as needed)
            # F[x, y, 1] = 0.0

    # Relaxation parameter:
    omega = 0.6
    relax_corr = 1.0 - 1.0/(2.0 * omega)

    # Initialize lattice distributions f using equilibrium with forcing:
    f = np.empty((Nx, Ny, 9), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x,y]:
                # Square of velocity:
                u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
                for i in range(9):
                    # Compute dot product u·c[i]:
                    eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                    # Forcing term contribution:
                    Fi = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
                    f[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi
    
    Fi = np.empty((Nx, Ny, 9), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                for i in range(9):
                    Fi[x, y, i] = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
    
    # Main simulation loop

    for step in range(T):
        # Collision: compute equilibrium distribution and relax toward it
        feq = np.empty((Nx, Ny, 9), dtype=np.float64)
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x,y]:
                    u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
                    for i in range(9):
                        eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                        feq[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi[x, y, i]
                        f[x, y, i] = f[x, y, i] + omega * (feq[x, y, i] - f[x, y, i])
        
        
        # Streaming step: propagate distributions
        f_stream = np.copy(f)
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    for i in range(9):
                        new_x = (x + c[i, 0]) % Nx
                        new_y = (y + c[i, 1]) % Ny
                        if fluid[new_x, new_y]:
                            f_stream[new_x, new_y, i] = f[x, y, i]
                        else:
                            f_stream[x, y, bounce_back_pairs[i]] = f[x, y, i]
        f = f_stream  # update f
        
        # Update macroscopic variables: density and velocity
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x,y]:
                    sum_f = 0.0
                    u0 = 0.0
                    u1 = 0.0
                    for i in range(9):
                        sum_f += f[x, y, i]
                        u0 += f[x, y, i] * c[i, 0]
                        u1 += f[x, y, i] * c[i, 1]
                    rho[x, y] = sum_f
                    if sum_f != 0.0:
                        u[x, y, 0] = u0 / sum_f
                        u[x, y, 1] = u1 / sum_f
                    else:
                        u[x, y, 0] = 0.0
                        u[x, y, 1] = 0.0
    t=T
    u_max_last = np.inf
    u_max = 0
    while (t<100_000):
        # Collision: compute equilibrium distribution and relax toward it
        feq = np.empty((Nx, Ny, 9), dtype=np.float64)
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x,y]:
                    u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
                    for i in range(9):
                        eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                        feq[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi[x, y, i]
                        f[x, y, i] = f[x, y, i] + omega * (feq[x, y, i] - f[x, y, i])
        
        
        # Streaming step: propagate distributions
        f_stream = np.copy(f)
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    for i in range(9):
                        new_x = (x + c[i, 0]) % Nx
                        new_y = (y + c[i, 1]) % Ny
                        if fluid[new_x, new_y]:
                            f_stream[new_x, new_y, i] = f[x, y, i]
                        else:
                            f_stream[x, y, bounce_back_pairs[i]] = f[x, y, i]
        f = f_stream  # update f
        
        # Update macroscopic variables: density and velocity
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x,y]:
                    sum_f = 0.0
                    u0 = 0.0
                    u1 = 0.0
                    for i in range(9):
                        sum_f += f[x, y, i]
                        u0 += f[x, y, i] * c[i, 0]
                        u1 += f[x, y, i] * c[i, 1]
                    rho[x, y] = sum_f
                    if sum_f != 0.0:
                        u[x, y, 0] = u0 / sum_f
                        u[x, y, 1] = u1 / sum_f
                    else:
                        u[x, y, 0] = 0.0
                        u[x, y, 1] = 0.0
        t+=1
        u_max = np.amax(u)
        if abs(u_max-u_max_last) <1e-10:
            break
        u_max_last=u_max


    u_x = 0.0
    u_y = 0.0
    tot_rho = 0.0
    num = 0.0
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x,y]:
                u_x += u[x, y,0] 
                u_y += u[x, y,1] 
                tot_rho += rho[x, y]
                num += 1.0
    avg_u_x = u_x/num
    avg_u_y = u_y/num
    avg_rho = tot_rho/num
    mu = (relax_corr-1/2)/3
    k_x = avg_u_x*mu/(avg_rho*grav)
    k_y = avg_u_y*mu/(avg_rho*grav)
    return u, k_x, k_y, t

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
def run_lbm(img):
    ux, k_xx, k_xy, t = big_LBM(img, T=10_000, force_dir=0)
    uy, k_yx, k_yy, t = big_LBM(img, T=10_000, force_dir=1)
    return np.array([[k_xx, k_xy], [k_yx, k_yy]])

def print_diff(name, k_ref, k_transformed, tol=1e-3):
    diff = np.abs(k_ref - k_transformed)
    print(f"\n{name} transformation:")
    print("Expected:\n", k_ref)
    print("Got:\n", k_transformed)
    print("Difference:\n", diff)
    print("OK ✅" if np.all(diff < tol) else "Mismatch ❌")
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

    # img = periodic_binary_blobs(blob_density=0.5, sigma=4.0, shape=(128, 128), seed=9)
    # ux, k_xx, k_xy, t = big_LBM(img,T=10_000, force_dir=0)
    # print(f"ran for {t} iterations")
    # uy, k_yx, k_yy, t = big_LBM(img,T=10_000, force_dir=1)
    # print(f"ran for {t} iterations")
    # k = np.array([[k_xx, k_xy],[k_yx,k_yy]])
    # print(k)




    # plt.subplot(1,2,1)
    # plt.imshow(np.sqrt(ux[:,:,0]**2 + ux[:,:,1]**2))
    # plt.colorbar()
    # plt.subplot(1,2,2)
    # plt.imshow(np.sqrt(ux[:,:,0]**2 + ux[:,:,1]**2))
    # plt.colorbar()

    # plt.show()


    # labeled = label_fluid_periodic(img)
    # px, py = check_periodic_percolation(labeled)
    # print("Percolates X:", px, "Percolates Y:", py)
    # filled = fill_non_percolating_periodic(img,labeled,px,py)
    # print("Percolates:", px and py)
    # plot_percolation(img, labeled, filled, px, py,name="_full_media")

    img = periodic_binary_blobs(blob_density=0.5, sigma=4.0, shape=(128, 128), seed=9)
    k = run_lbm(img)
    print("Original permeability tensor:\n", k)

    # --- Rotation matrices ---
    R90  = np.array([[0, -1], [1, 0]])
    R180 = np.array([[-1, 0], [0, -1]])
    R270 = np.array([[0, 1], [-1, 0]])
    I    = np.eye(2)

    # --- Test transforms ---
    tests = [
        ("Rotate 90°",     np.rot90(img, k=1), lambda K: R90.T @ K @ R90),
        ("Rotate 180°",    np.rot90(img, k=2), lambda K: R180.T @ K @ R180),
        ("Rotate 270°",    np.rot90(img, k=3), lambda K: R270.T @ K @ R270),
        ("Flip vertical",  np.flipud(img),     lambda K: np.array([[ K[0,0], -K[0,1]], [-K[1,0], K[1,1]]])),
        ("Flip horizontal",np.fliplr(img),     lambda K: np.array([[ K[0,0], -K[0,1]], [-K[1,0], K[1,1]]]))
    ]

    # --- Run tests ---
    for name, img_t, transform_fn in tests:
        k_expected = transform_fn(k)
        k_actual = run_lbm(img_t)
        print_diff(name, k_expected, k_actual)