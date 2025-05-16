import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.dirname(__file__)
from mpl_toolkits.axes_grid1 import make_axes_locatable

img = np.load(os.path.join(path,"images_filled/00000.npy"))

ux = np.load(os.path.join(path,"output_checkpoints/u_x_00000.npy"))
uy = np.load(os.path.join(path,"output_checkpoints/u_y_00000.npy"))
k = np.load(os.path.join(path,"output_checkpoints/k_00000.npy"))

# Plot ux[:,:,0]
fig, ax = plt.subplots(figsize=(6,6))
divider = make_axes_locatable(ax)
im = ax.imshow(ux[:,:,0], cmap="viridis")
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path,"image_uxx_demo.pdf"))
plt.close(fig)

# Plot ux[:,:,1]
fig, ax = plt.subplots(figsize=(6,6))
divider = make_axes_locatable(ax)
im = ax.imshow(ux[:,:,1], cmap="viridis")
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path,"image_uxy_demo.pdf"))
plt.close(fig)

# Plot uy[:,:,0]
fig, ax = plt.subplots(figsize=(6,6))
divider = make_axes_locatable(ax)
im = ax.imshow(uy[:,:,0], cmap="viridis")
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path,"image_uyx_demo.pdf"))
plt.close(fig)

# Plot uy[:,:,1]
fig, ax = plt.subplots(figsize=(6,6))
divider = make_axes_locatable(ax)
im = ax.imshow(uy[:,:,1], cmap="viridis")
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path,"image_uyy_demo.pdf"))
plt.close(fig)

print(k)