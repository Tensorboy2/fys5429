import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.dirname(__file__)


img = np.load(os.path.join(path,"images/00000.npy"))
plt.figure(figsize=(6,6))
plt.imshow(img,cmap="gray", interpolation="none")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path,"image_demo.pdf"))
# plt.show()

h = img.shape[0] // 4
w = img.shape[1] // 4

fix,ax = plt.subplots(4,4, figsize=(6,6))
for i in range(4):
    for j in range(4):
        start_row = i * h
        end_row = start_row + h
        start_col = j * w
        end_col = start_col + w

        ax[i,j].imshow(img[start_row:end_row, start_col:end_col],cmap="gray", interpolation="none")
        ax[i,j].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path,"image_patched_demo.pdf"))
# plt.show()

fix,ax = plt.subplots(1,16, figsize=(6,6))
patch_index = 0
for i in range(4):
    for j in range(4):
        start_row = i * h
        end_row = start_row + h
        start_col = j * w
        end_col = start_col + w

        patch = img[start_row:end_row, start_col:end_col]

        flattened_patch = patch.reshape(-1,1)

        ax[patch_index].imshow(flattened_patch, cmap="gray", aspect="auto", interpolation="none") # Display the original patch content
        ax[patch_index].axis("off")

        patch_index += 1
plt.tight_layout()
plt.savefig(os.path.join(path,"image_patched_flattened_demo.pdf"))
# plt.show()

weights = np.random.rand(h**2,1024)
plt.figure(figsize=(6,6))
plt.imshow(weights,cmap="viridis", interpolation="none")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path,"image_projection_demo.pdf"))
# plt.show()

fix,ax = plt.subplots(16,1, figsize=(6,6))
patch_index = 0
for i in range(4):
    for j in range(4):
        start_row = i * h
        end_row = start_row + h
        start_col = j * w
        end_col = start_col + w

        patch = img[start_row:end_row, start_col:end_col]

        flattened_patch = patch.flatten()
        proj = weights @ flattened_patch
        ax[patch_index].imshow(proj.reshape(1,-1), cmap="viridis", aspect="auto", interpolation="none") # Display the original patch content
        ax[patch_index].axis("off")

        patch_index += 1
plt.tight_layout()
plt.savefig(os.path.join(path,"image_projected_demo.pdf"))
# plt.show()