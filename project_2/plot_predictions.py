import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams.update({
    # "text.usetex": False,  # Set to True if you want LaTeX-style math formatting (requires LaTeX install)
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,       # High-res for crisp images
    "savefig.dpi": 300,
    "figure.autolayout": True,
    "pdf.fonttype": 42,      # Avoids type-3 fonts in PDFs (LaTeX likes this)
})

from data_loader import get_data
from models.resnet import ResNet50
from models.vit import ViT_B16
from models.convnext import ConvNeXtTiny, ConvNeXtSmall

path = os.path.dirname(__file__)
save_path = os.path.join(path, "plots/prediction_plots/predictions.npz")

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
def plot_scatter(preds, label, cmap):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5), dpi=300)
    for i in range(4):
        ax = axes[i // 2, i % 2]
        error = np.abs(preds[:, i] - targets[:, i])
        sns.scatterplot(
            x=targets[:, i], y=preds[:, i], hue=error,
            palette=cmap, s=10, alpha=0.6, legend=False, ax=ax
        )
        min_val = min(targets[:, i].min(), preds[:, i].min())
        max_val = max(targets[:, i].max(), preds[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"$k_{{{i // 2}{i % 2}}}$", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.suptitle(f"{label} - Predicted vs Actual", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(path, "plots/prediction_plots", f"similarity_plot_{label}.pdf"))
    plt.close()


def plot_histogram(relative_errors, label):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5), dpi=300)
    for i in range(4):
        ax = axes[i // 2, i % 2]
        sns.histplot(
            relative_errors[:, i], kde=True, stat="density", bins=40,
            color="C0", alpha=0.7, ax=ax
        )
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel(r"$1 - \hat{k} / k$")
        ax.set_ylabel("Density")
        ax.set_title(f"$k_{{{i // 2}{i % 2}}}$", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.suptitle(f"{label} - Relative Error Distribution", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(path, "plots/prediction_plots", f"histogram_{label}.pdf"))
    plt.close()

# --- Relative Errors ---
epsilon = 1e-8
threshold = 50.0
relative_errors_vit = np.clip(1 - preds_vit / (targets + epsilon), -threshold, threshold)
relative_errors_resnet = np.clip(1 - preds_resnet / (targets + epsilon), -threshold, threshold)
relative_errors_tiny = np.clip(1 - preds_tiny / (targets + epsilon), -threshold, threshold)
relative_errors_small = np.clip(1 - preds_small / (targets + epsilon), -threshold, threshold)

# --- Plot ---
plot_scatter(preds_resnet, label="ResNet50", cmap="viridis_r")
plot_scatter(preds_vit, label="ViT_B16", cmap="plasma_r")
plot_scatter(preds_tiny, label="ConvNeXtTiny", cmap="viridis_r")
plot_scatter(preds_small, label="ConvNeXtSmall", cmap="plasma_r")

plot_histogram(relative_errors_resnet, label="ResNet50")
plot_histogram(relative_errors_vit, label="ViT_B16")
plot_histogram(relative_errors_tiny, label="ConvNeXtTiny")
plot_histogram(relative_errors_small, label="ConvNeXtSmall")

# plt.show()
