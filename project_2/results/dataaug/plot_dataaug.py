import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.lines import Line2D
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

# Get current script directory
path = os.path.dirname(__file__)

# Define model variants and files
augmentation_variants = {
    "None": "convnextsmall_metrics_dataaugmentation_test.csv",
    "Horizontal Flip": "convnextsmall_hflip_metrics_dataaugmentation_test.csv",
    "Vertical Flip": "convnextsmall_vflip_metrics_dataaugmentation_test.csv",
    "Rotation": "convnextsmall_rotate_metrics_dataaugmentation_test_2.csv",
    "H+V+Rotate": "convnextsmall_hflip_vflip_rotate_metrics_dataaugmentation_test_2.csv",
    "H+V Flip": "convnextsmall_hflip_vflip_metrics_dataaugmentation_test.csv",
    "Group": "convnextsmall_group_dataaugmentation_test.csv",
}

# Assign distinct colors
colors = sns.color_palette("tab10", n_colors=len(augmentation_variants))

# Load all data
models_info = {}
for (label, file), color in zip(augmentation_variants.items(), colors):
    df = pd.read_csv(os.path.join(path, file))
    models_info[label] = {"df": df, "color": color}

# Plot R^2 scores
plt.figure(figsize=(6.4, 6.4))

for label, info in models_info.items():
    df = info["df"]
    color = info["color"]

    # Plot test and train R2
    plt.plot(df["epoch"], df["test_r2"], c=color, linestyle="-")
    plt.plot(df["epoch"], df["train_r2"], c=color, linestyle="--", alpha=0.5)

    # Print max test R2 info
    idx_max = np.argmax(df['test_r2'])
    print(f"Aug: {label}, test R²: {df['test_r2'][idx_max]:.5f}, train R²: {df['train_r2'][idx_max]:.5f}, "
          f"test MSE: {df['test_mse'][idx_max]:.6f}, train MSE: {df['train_mse'][idx_max]:.6f}")

# Custom legend
legend_elements = [
    Line2D([0], [0], color=info["color"], lw=2, label=label)
    for label, info in models_info.items()
] + [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.5, label='Train')
]

plt.legend(handles=legend_elements, fontsize=8, title="Augmentations", frameon=False)
plt.xlabel("Epochs")
plt.ylabel(r"$R^2$ Score")
plt.ylim(0.967, 1.001)
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, "dataaug_r2.pdf"), bbox_inches='tight')

# Plot MSE
plt.figure(figsize=(6.4, 6.4))

for label, info in models_info.items():
    df = info["df"]
    color = info["color"]
    plt.plot(df["epoch"], df["test_mse"], c=color, linestyle="-")
    plt.plot(df["epoch"], df["train_mse"], c=color, linestyle="--", alpha=0.5)

# Legend again
plt.legend(handles=legend_elements, fontsize=8, title="Augmentations", frameon=False)
plt.xlabel("Epochs")
plt.ylabel("MSE (Lattice Units)")
plt.yscale("log")
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, "dataaug_mse.pdf"), bbox_inches='tight')
