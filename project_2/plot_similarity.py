import os
import numpy as np
import torch
from data_loader import get_data
from tqdm import tqdm
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
from models.vit import ViT_B16, ViT_S16, ViT_T16, ViT_T8, ViT_S8
from models.convnext import ConvNeXtTiny, ConvNeXtSmall

ROOT = os.path.join(os.path.dirname(__file__), "results_2")

models = {
    "ConvNeXtSmall": [ConvNeXtSmall, "ConvNeXtSmall_400_epochs.pth"],
    # "ViT-T16": [ViT_T16,"ViT_T16_all.pth"],
    "ViT-S8": [ViT_S8,"ViT_S8.pth"],
}

# Path for the combined cache file containing predictions for all models
CACHE_PATH = os.path.join(ROOT, "all_models_similarity_filled.npz")

# Try loading an existing combined cache. If it exists, we'll only run inference
# for models that are missing from the cache. If it doesn't exist we create a
# fresh dict and populate it by running inference for every model.
preds = {}
if os.path.exists(CACHE_PATH):
    print("Loading existing combined predictions cache...")
    loaded = np.load(CACHE_PATH, allow_pickle=True)
    # Each entry was saved as an object; convert back to dict
    preds = {model: loaded[model].item() for model in loaded.files}
    print(f"Loaded predictions for models: {list(preds.keys())}")
else:
    print("No combined cache found. Will generate predictions for all models.")

for model_name, (model_func, model_path) in models.items():
    # If we have cached results for this model, skip inference
    if model_name in preds:
        print(f"Skipping inference for {model_name} (cache found).")
        continue

    print(f"No existing predictions found for {model_name}. Generating new predictions.")
    _, test_loader = get_data(batch_size=1, num_workers=0, num_samples=None)

    print(f"Processing model: {model_name}, with weights from {model_path}")
    preds[model_name] = {}

    model = model_func(pre_trained_path=model_path)  # paths may be relative
    model.eval()

    all_preds_filled = []
    all_targets = []
    with torch.no_grad():
        for image, image_filled, k in tqdm(test_loader):
            outputs_image_filled = model(image_filled)
            all_preds_filled.append(outputs_image_filled.cpu().numpy().reshape(-1, 2, 2))
            all_targets.append(k.cpu().numpy().reshape(-1, 2, 2))

    preds[model_name]["image_filled"] = np.vstack(all_preds_filled).squeeze()
    preds[model_name]["target"] = np.vstack(all_targets).squeeze()
    print(f"Completed model: {model_name}")

    # Save combined cache after each model so partial results are preserved
    np.savez_compressed(CACHE_PATH, **{k: v for k, v in preds.items()})
    print(f"Saved combined cache with models: {list(preds.keys())}")


def plot_model_scatter(preds, model_name, cmap="viridis"):
    """
    Makes a 2x2 scatter plot for predictions vs targets using seaborn.
    Each subplot corresponds to one component of k.
    """
    fig, ax = plt.subplots(2, 2, figsize=(5.4, 5.4))
    
    for i in range(2):
        for j in range(2):
            # Extract predictions and targets
            y_true = preds[model_name]["target"][:, i, j]
            y_pred = preds[model_name]["image_filled"][:, i, j]
            
            # Error for coloring
            error = np.abs(y_pred - y_true)
            
            # Seaborn scatterplot
            sns.scatterplot(
                x=y_true, y=y_pred,
                hue=error,
                palette=cmap,
                s=10, alpha=0.7,
                legend=False,
                ax=ax[i, j]
            )
            
            # Diagonal reference line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax[i, j].plot([min_val, max_val], [min_val, max_val], '-', lw=1, alpha=0.4,color='red',label="Identity")
            # ax[i,j].legend()
            # Labels
            ax[i, j].set_xlabel("Target", fontsize=12)
            ax[i, j].set_ylabel("Prediction", fontsize=12)
            ax[i, j].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
            sub = f"{'x' if i == 0 else 'y'}{'x' if j == 0 else 'y'}"
            ax[i, j].set_title(f"$k_{{{sub}}}$", fontsize=14)
    
    # plt.suptitle(f"Scatter for {model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, f"{model_name}_similarity_filled.pdf"), dpi=300)
    plt.close()
cmaps = ["viridis","viridis"]
for model_name,cmap in zip(models.keys(), cmaps):
    plot_model_scatter(preds, model_name, cmap=cmap)

