import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
path = os.path.dirname(__file__)
from data_loader import Dataset, get_data
from models.resnet import ResNet50
from models.vit import ViT_B16
import matplotlib.cm as cm
import matplotlib.colors as mcolors

save_path = os.path.join(path, "data/predictions.npz")

if os.path.exists(save_path):
    data = np.load(save_path,mmap_mode="r")
    preds_v = data["preds_v"]
    preds_r = data["preds_r"]
    targets = data["targets"]
    print(f"Loaded cached predictions from {save_path}")
else:
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ViT_B16 = ViT_B16(pre_trained=True).eval().to(device)
    ResNet50 = ResNet50(pre_trained=True).eval().to(device)

    # Load data
    images = np.load(os.path.join(path, "data/images_filled.npz"), mmap_mode="r")["images_filled"]
    k = np.load(os.path.join(path, "data/k.npz"), mmap_mode="r")["k"]

    # Storage
    all_preds_v = []
    all_preds_r = []
    all_targets = []

    # Inference loop
    with torch.no_grad():
        for i, (images_filled_n, k_n) in enumerate(zip(images, k)):
            images_filled_t = torch.from_numpy(images_filled_n).float().unsqueeze(0).unsqueeze(0).to(device)

            preds_v = ViT_B16(images_filled_t).cpu().numpy().flatten()
            preds_r = ResNet50(images_filled_t).cpu().numpy().flatten()

            all_preds_v.append(preds_v)
            all_preds_r.append(preds_r)
            all_targets.append(k_n.flatten())
            if i % 100 == 0:
                print(f"Inferred {i}/{len(images)} samples")

    # Convert to arrays
    all_preds_v = np.array(all_preds_v)
    all_preds_r = np.array(all_preds_r)
    all_targets = np.array(all_targets)

    # Save to .npz file
    np.savez_compressed(
        save_path,
        preds_v=all_preds_v,
        preds_r=all_preds_r,
        targets=all_targets
    )

    print(f"Predictions saved to {save_path}")




# Plot per k value (index 0-3)
plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    sns.scatterplot(x=targets[:, i], y=preds_r[:, i],c=(np.abs(preds_r[:, i] - targets[:, i])),cmap="viridis_r", alpha=0.4, label="ResNet50")
    # sns.scatterplot(x=targets[:, i], y=preds_v[:, i],c=(np.abs(preds_v[:, i] - targets[:, i])),cmap="plasma", alpha=0.4, label="ViT B/16")

    # Optional: y = x reference line
    min_val = min(targets[:, i].min(), preds_r[:, i].min(), preds_v[:, i].min())
    max_val = max(targets[:, i].max(), preds_r[:, i].max(), preds_v[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Element k[{i // 2}, {i % 2}]")
    plt.legend()
    plt.grid(True)

# plt.suptitle("Predicted vs Actual per Matrix Entry (2×2 Output)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    # sns.scatterplot(x=targets[:, i], y=preds_r[:, i],c=(np.abs(preds_r[:, i] - targets[:, i])),cmap="viridis", alpha=0.4, label="ResNet50")
    sns.scatterplot(x=targets[:, i], y=preds_v[:, i],c=(np.abs(preds_v[:, i] - targets[:, i])),cmap="plasma_r", alpha=0.4, label="ViT B/16")

    # Optional: y = x reference line
    min_val = min(targets[:, i].min(), preds_r[:, i].min(), preds_v[:, i].min())
    max_val = max(targets[:, i].max(), preds_r[:, i].max(), preds_v[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Element k[{i // 2}, {i % 2}]")
    plt.legend()
    plt.grid(True)

# plt.suptitle("Predicted vs Actual per Matrix Entry (2×2 Output)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

epsilon = 1e-8
threshold = 50.0
relative_errors_v = np.clip(1 - preds_v / (targets + epsilon),-threshold,threshold)
relative_errors_r = np.clip(1 - preds_r / (targets + epsilon),-threshold,threshold)

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    sns.histplot(relative_errors_r[:, i], kde=True, stat="density", bins=50, alpha=0.7)

    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("1 - Predicted / Actual")
    plt.ylabel("Density")
    plt.title(f"ResNet50: Relative Error for k[{i // 2}, {i % 2}]")
    plt.grid(True)

plt.suptitle("ResNet50 - Relative Error Histograms per Matrix Element", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    sns.histplot(relative_errors_v[:, i], kde=True, stat="density", bins=50, alpha=0.7)

    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("1 - Predicted / Actual")
    plt.ylabel("Density")
    plt.title(f"ViT B/16: Relative Error for k[{i // 2}, {i % 2}]")
    plt.grid(True)

plt.suptitle("ViT B/16 - Relative Error Histograms per Matrix Element", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()