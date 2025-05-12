import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.dirname(__file__)


img = np.load(os.path.join(path,"images_filled/00000.npy"))

ux = np.load(os.path.join(path,"output_checkpoints/u_x_00000.npy"))
uy = np.load(os.path.join(path,"output_checkpoints/u_y_00000.npy"))
k = np.load(os.path.join(path,"output_checkpoints/k_00000.npy"))

plt.figure(figsize=(6,6))
u_x = np.sqrt(ux[:,:,0]**2 + ux[:,:,1]**2)
plt.imshow(u_x,cmap="viridis")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path,"image_ux_demo.pdf"))
plt.show()

plt.figure(figsize=(6,6))
u_y = np.sqrt(uy[:,:,0]**2 + uy[:,:,1]**2)
plt.imshow(u_y,cmap="viridis")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path,"image_uy_demo.pdf"))
plt.show()

print(k)