import numpy as np
import os
from tqdm import tqdm
path = os.path.dirname(__file__)

folder = "output_checkpoints"
# all_images = []

N = 20000  # number of triplets

u_x_list = []
# u_y_list = []
# k_list = []

for i in tqdm(range(N)):
    u_x = np.load(os.path.join(folder, f"u_x_{i:05d}.npy"))
    # u_y = np.load(os.path.join(folder, f"u_y_{i:05d}.npy"))
    # k = np.load(os.path.join(folder, f"k_{i:05d}.npy"))
    
    u_x_list.append(u_x)
    # u_y_list.append(u_y)
    # k_list.append(k)

# Stack each into one big array: shape (N, H, W) or (N,) for scalars
u_x_all = np.stack(u_x_list)
# u_y_all = np.stack(u_y_list)
# k_all = np.stack(k_list)

np.savez_compressed("u_x.npz", u_x=u_x_all)