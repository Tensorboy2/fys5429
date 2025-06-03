import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.lines import Line2D
import matplotlib as mpl
path = os.path.dirname(__file__)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

sns.set_theme(style="whitegrid")


models = {
    "ResNet101": "resnet101_metrics_all_models_2.csv",
    "ResNet50": "resnet50_metrics_all_models_2.csv",
    "ViT-B16": "vit_b16_metrics_all_models_2.csv",
    "ConvNeXtTiny": "convnexttiny_metrics_all_models_2.csv",
    "ConvNeXtSmall": "convnextsmall_metrics_all_models_2.csv",
}

# Assign distinct colors:
colors = sns.color_palette("tab10", n_colors=len(models))

# Load all data:
models_info = {}
for (label, file), color in zip(models.items(), colors):
    df = pd.read_csv(os.path.join(path, file))
    models_info[label] = {"df": df, "color": color}

# Plot R^2 scores:
plt.figure(figsize=(6.4, 6.4))
for label, info in models_info.items():
    df = info["df"]
    color = info["color"]

    # Plot test and train R^2:
    plt.plot(df["epoch"], df["test_r2"], c=color, linestyle="-")
    plt.plot(df["epoch"], df["train_r2"], c=color, linestyle="--", alpha=0.5)

    # Print max test R^2 info for leaderboard:
    idx_max = np.argmax(df['test_r2'])
    print(f"model: {label}, test R²: {df['test_r2'][idx_max]:.5f}, train R²: {df['train_r2'][idx_max]:.5f}, "
          f"test MSE: {df['test_mse'][idx_max]:.6f}, train MSE: {df['train_mse'][idx_max]:.6f}")
    
# Custom legend:
legend_elements = [
    Line2D([0], [0], color=info["color"], lw=2, label=label)
    for label, info in models_info.items()
] + [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.5, label='Train')
]
plt.legend(handles=legend_elements, fontsize=8, title="Models", frameon=False)
plt.xlabel("Epochs")
plt.ylabel(r"$R^2$ Score")
plt.ylim(0.95, 1)
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, "full_training_r2.pdf"), bbox_inches='tight')

# Plot MSE:
plt.figure(figsize=(6.4, 6.4))
for label, info in models_info.items():
    df = info["df"]
    color = info["color"]
    plt.plot(df["epoch"], df["test_mse"], c=color, linestyle="-")
    plt.plot(df["epoch"], df["train_mse"], c=color, linestyle="--", alpha=0.5)
plt.legend(handles=legend_elements, fontsize=8, title="Models", frameon=False)
plt.xlabel("Epochs")
plt.ylabel("MSE (Lattice Units)")
plt.yscale("log")
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, "full_training_mse.pdf"), bbox_inches='tight')
