import numpy as np
import torch
import os
import matplotlib.pyplot as plt
# import seaborn as sns

from data_loader import get_data
from models.resnet import ResNet50
from models.vit import ViT_B16
from models.convnext import ConvNeXtTiny, ConvNeXtSmall

path = os.path.dirname(__file__)
save_path = os.path.join(path, "data/predictions.npz")

def run_inference(model, images, device):
    preds = []
    with torch.no_grad():
        for i, img in enumerate(images):
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
            output = model(img_tensor).cpu().numpy().flatten()
            preds.append(output)
            if i % 100 == 0:
                print(f"[{model.__class__.__name__}] Processed {i}/{len(images)} samples")
    return np.array(preds)

if os.path.exists(save_path):
    data = np.load(save_path, mmap_mode="r")
    preds_vit = data["preds_vit"]
    preds_resnet = data["preds_resnet"]
    preds_tiny = data["preds_tiny"]
    preds_small = data["preds_small"]
    targets = data["targets"]
    print(f"Loaded cached predictions from {save_path}")
else:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    model_r = ResNet50(pre_trained=True).eval().to(device)
    model_v = ViT_B16(pre_trained=True).eval().to(device)
    # Uncomment if you want ConvNeXtTiny or Small:
    model_tiny = ConvNeXtTiny(pre_trained=True).eval().to(device)
    model_small = ConvNeXtSmall(pre_trained=True).eval().to(device)

    # Load data
    images = np.load(os.path.join(path, "data/images_filled.npz"), mmap_mode="r")["images_filled"]
    k = np.load(os.path.join(path, "data/k.npz"), mmap_mode="r")["k"]

    # Inference
    preds_resnet = run_inference(model_r, images, device)
    preds_vit = run_inference(model_v, images, device)
    preds_tiny = run_inference(model_tiny, images, device)
    preds_small = run_inference(model_small, images, device)
    targets = k.reshape(len(k), -1)

    # Save predictions
    np.savez_compressed(save_path, preds_vit=preds_vit, preds_resnet=preds_resnet, preds_tiny=preds_tiny, preds_small=preds_small, targets=targets)
    print(f"Predictions saved to {save_path}")

# --- Plotting Utilities ---
# def plot_scatter(preds, label, cmap):
#     plt.figure(figsize=(10, 8))
#     for i in range(4):
#         plt.subplot(2, 2, i + 1)
#         error = np.abs(preds[:, i] - targets[:, i])
#         sns.scatterplot(x=targets[:, i], y=preds[:, i], hue=error, palette=cmap, alpha=0.4, legend=False)
#         min_val = min(targets[:, i].min(), preds[:, i].min())
#         max_val = max(targets[:, i].max(), preds[:, i].max())
#         plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
#         plt.xlabel("Actual")
#         plt.ylabel("Predicted")
#         plt.title(f"{label}: k[{i // 2}, {i % 2}]")
#         plt.grid(True)
#     plt.suptitle(f"{label} - Predicted vs Actual", fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig(os.path.join(path,f"similarity_plot_{label}.pdf"))


# def plot_histogram(relative_errors, label):
#     plt.figure(figsize=(10, 8))
#     for i in range(4):
#         plt.subplot(2, 2, i + 1)
#         sns.histplot(relative_errors[:, i], kde=True, stat="density", bins=50, alpha=0.7)
#         plt.axvline(0, color='black', linestyle='--', linewidth=1)
#         plt.xlabel("1 - Predicted / Actual")
#         plt.ylabel("Density")
#         plt.title(f"{label}: Relative Error for k[{i // 2}, {i % 2}]")
#         plt.grid(True)
#     plt.suptitle(f"{label} - Relative Error Histograms", fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig(os.path.join(path,f"histogram_{label}.pdf"))

# # --- Relative Errors ---
# epsilon = 1e-8
# threshold = 50.0
# relative_errors_vit = np.clip(1 - preds_vit / (targets + epsilon), -threshold, threshold)
# relative_errors_resnet = np.clip(1 - preds_resnet / (targets + epsilon), -threshold, threshold)
# relative_errors_tiny = np.clip(1 - preds_tiny / (targets + epsilon), -threshold, threshold)
# relative_errors_small = np.clip(1 - preds_small / (targets + epsilon), -threshold, threshold)

# # --- Plot ---
# plot_scatter(preds_resnet, label="ResNet50", cmap="viridis_r")
# plot_scatter(preds_vit, label="ViT_B16", cmap="plasma_r")
# plot_scatter(preds_tiny, label="ConvNeXtTiny", cmap="viridis_r")
# plot_scatter(preds_small, label="ConvNeXtSmall", cmap="plasma_r")

# plot_histogram(relative_errors_resnet, label="ResNet50")
# plot_histogram(relative_errors_vit, label="ViT_B16")
# plot_histogram(relative_errors_tiny, label="ConvNeXtTiny")
# plot_histogram(relative_errors_small, label="ConvNeXtSmall")

# plt.show()
