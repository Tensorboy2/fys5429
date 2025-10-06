import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

sns.set_theme(style="whitegrid")

from models.resnet import ResNet50, ResNet101
from models.vit import ViT_B16
from models.convnext import ConvNeXtTiny, ConvNeXtSmall

ROOT = os.path.dirname(__file__)
SAVE_PATH = os.path.join(ROOT, "results/prediction_plots/predictions.npz")
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
    '''
    Runs inference on all samples in the datasets and returns its predicted values.
    '''
    preds = []
    with torch.no_grad():
        for i, img in enumerate(images):
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
            output = model(img_tensor).cpu().numpy().flatten()
            preds.append(output)
            if i % 100 == 0:
                print(f"[{model.__class__.__name__}] Processed {i}/{len(images)} samples")
    return np.array(preds)

def plot_scatter(preds, targets, label, cmap):
    '''
    Similarity plot for model inference.
    '''
    plt.figure(figsize=(6.4, 6.4))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        
        error = np.abs(preds[:, i] - targets[:, i])
        sns.scatterplot(
            x=targets[:, i], y=preds[:, i], hue=error,
            palette=cmap, s=10, alpha=0.6, legend=False
        )
        min_val = min(targets[:, i].min(), preds[:, i].min())
        max_val = max(targets[:, i].max(), preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"$k_{{{i // 2}{i % 2}}}$")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results/prediction_plots", f"similarity_plot_{label}.png"))
    plt.close()

def compute_relative_errors(preds, targets, threshold=50.0, epsilon=1e-8):
    return np.clip(1 - preds / (targets + epsilon), -threshold, threshold)

def plot_histogram(relative_errors, label):
    '''
    Histogram of relative error in the prediction vs. target values.
    '''
    plt.figure(figsize=(6.4, 6.4))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        
        sns.histplot(
            relative_errors[:, i], kde=True, stat="density", bins=40,
            color="C0", alpha=0.7
        )
        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.xlabel(r"$R$")
        plt.ylabel("Density")
        plt.title(f"$k_{{{i // 2}{i % 2}}}$")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results/prediction_plots", f"histogram_{label}.png"))
    plt.close()




def generate_all_plots(predictions, targets):
    colormaps = {
        "ResNet50": "viridis_r",
        "ResNet101": "viridis_r",
        "ViT-B16": "plasma_r",
        "ConvNeXt-Tiny": "viridis_r",
        "ConvNeXt-Small": "plasma_r"
    }

    for model_name, config in MODELS_INFO.items():
        label = model_name
        key = config["key"]
        if key not in predictions:
            print(f"Skipping {label} – no predictions found.")
            continue

        preds = predictions[key]
        rel_errors = compute_relative_errors(preds, targets)

        plot_scatter(preds, targets, label=label, cmap=colormaps.get(label, "viridis"))
        plot_histogram(rel_errors, label=label)
        print(f"Plots generated for {label}")


def main():
    '''
    If the prediction.npz does not exists, then inference is run and then plotted. 
    If it exists then inference is skipped. 
    '''
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

    generate_all_plots(predictions, targets)


if __name__ == "__main__":
    main()
