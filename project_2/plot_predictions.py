import os
import numpy as np
import torch
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

from models.resnet import ResNet50, ResNet101
from models.vit import ViT_B16
from models.convnext import ConvNeXtTiny, ConvNeXtSmall

ROOT = os.path.dirname(__file__)
SAVE_PATH = os.path.join(ROOT, "plots/prediction_plots/predictions.npz")
IMAGES_PATH = os.path.join(ROOT, "data/images_filled.npz")
K_PATH = os.path.join(ROOT, "data/k.npz")

MODELS_INFO = {
    "ViT-B16": {
        "class": ViT_B16,
        "key": "preds_vit"
    },
    "ResNet50": {
        "class": ResNet50,
        "key": "preds_resnet50"
    },
    "ResNet101": {
        "class": ResNet101,
        "key": "preds_resnet101"
    },
    "ConvNeXt-Tiny": {
        "class": ConvNeXtTiny,
        "key": "preds_tiny"
    },
    "ConvNeXt-Small": {
        "class": ConvNeXtSmall,
        "key": "preds_small"
    }
}

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

# def plot_scatter(preds, targets, label, cmap):
#     fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4))
#     for i in range(4):
#         ax = axes[i // 2, i % 2]
#         error = np.abs(preds[:, i] - targets[:, i])
#         sns.scatterplot(
#             x=targets[:, i], y=preds[:, i], hue=error,
#             palette=cmap, s=10, alpha=0.6, legend=False, ax=ax
#         )
#         min_val = min(targets[:, i].min(), preds[:, i].min())
#         max_val = max(targets[:, i].max(), preds[:, i].max())
#         ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
#         ax.set_xlabel("Actual")
#         ax.set_ylabel("Predicted")
#         ax.set_title(f"$k_{{{i // 2}{i % 2}}}$")
#         ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

#     plt.tight_layout()
#     plt.savefig(os.path.join(ROOT, "plots/prediction_plots", f"similarity_plot_{label}.pdf"), bbox_inches='tight')
#     plt.close()


# def plot_histogram(relative_errors, label):
#     fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4))
#     for i in range(4):
#         ax = axes[i // 2, i % 2]
#         sns.histplot(
#             relative_errors[:, i], kde=True, stat="density", bins=40,
#             color="C0", alpha=0.7, ax=ax
#         )
#         ax.axvline(0, color='black', linestyle='--', linewidth=1)
#         ax.set_xlabel(r"$R$")
#         ax.set_ylabel("Density")
#         ax.set_title(f"$k_{{{i // 2}{i % 2}}}$")
#         ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

#     plt.tight_layout()
#     plt.savefig(os.path.join(ROOT, "plots/prediction_plots", f"histogram_{label}.pdf"), bbox_inches='tight')
#     plt.close()


def compute_relative_errors(preds, targets, threshold=50.0, epsilon=1e-8):
    return np.clip(1 - preds / (targets + epsilon), -threshold, threshold)


# def generate_all_plots(predictions, targets):
#     colormaps = {
#         "ResNet50": "viridis_r",
#         "ResNet101": "viridis_r",
#         "ViT-B16": "plasma_r",
#         "ConvNeXt-Tiny": "viridis_r",
#         "ConvNeXt-Small": "plasma_r"
#     }

#     for model_name, config in MODELS_INFO.items():
#         label = model_name
#         key = config["key"]
#         if key not in predictions:
#             print(f"Skipping {label} – no predictions found.")
#             continue

#         preds = predictions[key]
#         rel_errors = compute_relative_errors(preds, targets)

#         plot_scatter(preds, targets, label=label, cmap=colormaps.get(label, "viridis"))
#         plot_histogram(rel_errors, label=label)
#         print(f"Plots generated for {label}")


def main():
    if os.path.exists(SAVE_PATH):
        data = np.load(SAVE_PATH, mmap_mode="r")
        predictions = {
            "preds_vit": data["preds_vit"],
            "preds_resnet50": data["preds_resnet50"],
            "preds_resnet101": data["preds_resnet101"],
            "preds_tiny": data["preds_tiny"],
            "preds_small": data["preds_small"]
        }
        targets = data["targets"]
        print(f"Loaded cached predictions from {SAVE_PATH}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data
        images = np.load(IMAGES_PATH, mmap_mode="r")["images_filled"]
        k = np.load(K_PATH, mmap_mode="r")["k"]
        targets = k.reshape(len(k), -1)

        # Load models and run inference
        predictions = {}
        for name, config in MODELS_INFO.items():
            print(f"Running inference for {name}")
            model = config["class"](pre_trained=True).eval().to(device)
            preds = run_inference(model, images, device)
            predictions[config["key"]] = preds

        np.savez_compressed(SAVE_PATH, **predictions, targets=targets)
        print(f"Predictions saved to {SAVE_PATH}")

    # generate_all_plots(predictions, targets)


if __name__ == "__main__":
    main()
