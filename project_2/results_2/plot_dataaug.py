'''
plot_dataaug.py

Module for plotting the R^2 accuracy and the mean square error form training with different data augmentation techniques.
'''

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

# Choose which model to plot: "convnextsmall", "vit_s16", or "both"
plot_model = "both"  # options: "convnextsmall", "vit_s16", "both"

augmentation_variants_convnext = {
    "None": "convnextsmall_metrics_dataaugmentation_test.csv",
    "Horizontal Flip": "convnextsmall_hflip_metrics_dataaugmentation_test.csv",
    "Vertical Flip": "convnextsmall_vflip_metrics_dataaugmentation_test.csv",
    "Rotation": "convnextsmall_rotate_metrics_dataaugmentation_test.csv",
    "H+V+Rotate": "convnextsmall__hflip_vflip_rotate_metrics_dataaugmentation_test.csv",
    "H+V Flip": "convnextsmall_hflip_vflip_metrics_dataaugmentation_test.csv",
    "Group": "convnextsmall__group_dataaugmentation_test.csv",
}

augmentation_variants_vit_s16 = {
    "None": "vit_s16_metrics_dataaugmentation_test.csv",
    "Horizontal Flip": "vit_s16_hflip_metrics_dataaugmentation_test.csv",
    "Vertical Flip": "vit_s16_vflip_metrics_dataaugmentation_test.csv",
    "Rotation": "vit_s16_rotate_metrics_dataaugmentation_test.csv",
    "H+V+Rotate": "vit_s16_hflip_vflip_rotate_metrics_dataaugmentation_test.csv",
    "H+V Flip": "vit_s16_hflip_vflip_metrics_dataaugmentation_test.csv",
    "Group": "vit_s16_group_dataaugmentation_test.csv",
}

def load_models_info(augmentation_variants):
    colors = sns.color_palette("tab10", n_colors=len(augmentation_variants))
    models_info = {}
    for (label, file), color in zip(augmentation_variants.items(), colors):
        try:
            df = pd.read_csv(os.path.join(path, file))
            models_info[label] = {"df": df, "color": color}
        except FileNotFoundError:
            print(f"Warning: File not found: {file}")
    return models_info


if plot_model in ["convnextsmall", "both"]:
    models_info_convnext = load_models_info(augmentation_variants_convnext)
if plot_model in ["vit_s16", "both"]:
    models_info_vit_s16 = load_models_info(augmentation_variants_vit_s16)

def plot_metrics(models_info, title_prefix):
    # Plot R^2 scores:
    plt.figure(figsize=(6.4, 6.4))
    for label, info in models_info.items():
        df = info["df"]
        color = info["color"]
        plt.plot(df["epoch"], 1-df["test_r2"], c=color, linestyle="-")
        plt.plot(df["epoch"], 1-df["train_r2"], c=color, linestyle="--", alpha=0.5)
        idx_max = np.argmax(df['test_r2'])
        print(f"{title_prefix} Aug: {label}, test R²: {df['test_r2'][idx_max]:.5f}, train R²: {df['train_r2'][idx_max]:.5f}, "
              f"test MSE: {df['test_mse'][idx_max]:.6f}, train MSE: {df['train_mse'][idx_max]:.6f}")
    legend_elements = [
        Line2D([0], [0], color=info["color"], lw=2, label=label)
        for label, info in models_info.items()
    ] + [
        Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.5, label='Train')
    ]
    plt.legend(handles=legend_elements, fontsize=8, title="Augmentations", frameon=False)
    plt.xlabel("Epochs")
    plt.ylabel(r"$1-R^2$ Score")
    # plt.ylim(0.9, 1.001)
    plt.yscale('log')
    plt.xscale("log")
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.title(f"{title_prefix} Data Augmentation: 1- R² Score")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{title_prefix.lower()}_dataaug_r2_v2.pdf"), bbox_inches='tight')

    # Plot MSE:
    plt.figure(figsize=(6.4, 6.4))
    for label, info in models_info.items():
        df = info["df"]
        color = info["color"]
        plt.plot(df["epoch"], df["test_mse"], c=color, linestyle="-")
        plt.plot(df["epoch"], df["train_mse"], c=color, linestyle="--", alpha=0.5)
    plt.legend(handles=legend_elements, fontsize=8, title="Augmentations", frameon=False)
    plt.xlabel("Epochs")
    plt.ylabel("MSE (Lattice Units)")
    plt.yscale("log")
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.title(f"{title_prefix} Data Augmentation: MSE")
    plt.tight_layout()
    # plt.savefig(os.path.join(path, f"{title_prefix.lower()}_dataaug_mse.pdf"), bbox_inches='tight')

    # Print header once:
    print(f"{'Augmentation':<15} {'Test R²':>10} {'Train R²':>10} {'Test MSE':>12} {'Train MSE':>12}")
    print("-" * 62)
    for label, info in models_info.items():
        df = info["df"]
        idx_max = np.argmax(df["test_r2"])
        row = df.iloc[idx_max]
        print(f"{label:<15} {row['test_r2']:.5f} {row['train_r2']:.5f} "
              f"{row['test_mse']:.6f} {row['train_mse']:.6f}")

if plot_model in ["convnextsmall", "both"]:
    plot_metrics(models_info_convnext, "ConvNeXtSmall")
if plot_model in ["vit_s16", "both"]:
    plot_metrics(models_info_vit_s16, "ViT-S16")
