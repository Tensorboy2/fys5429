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


runs = {
    # "V1": "vit_s16_extralong.csv",
    # "ViT-T16_long": "vit_t16_extralong.csv",
    # "ViT-B16_long": "vit_b16_extralong.csv",
    # "ViT-T16": "vit_t16_metrics_more_vits.csv",
    # "ViT-S16": "vit_s16_metrics_more_vits.csv",
    # "ViT-B16": "vit_b16_metrics_more_vits.csv",
    # "ResNet50_GC": "resnet50_gradient_clip_test.csv",
    # "ConvNeXtTiny": "convnexttiny_metrics_all_models_2.csv",
    # "ConvNeXtTiny_GC": "convnexttiny_gradient_clip_test.csv",
    # "ConvNext-Small": "convnextsmall_metrics_all_models_2.csv",
    # "ConvNeXtSmall-GC-100-epochs": "convnextsmall_gradient_clip_test_2.csv",
    # "ConvNext-Small-500-epochs": "convnextsmall_extralong.csv",
    # "ConvNeXtTiny_GC_long": "convnexttiny_gradient_clip_long_test_3.csv",

    # "ViT-S16-gc": "vit_s16_gradient_clip_test_2.csv",

    "ViT-S16-1k-epochs":"vit_s16_gradient_clip_long_test_3.csv",
    "ViT-S16-600-epochs": "ViT_S16_600_epochs.csv",
    "ViT-S16-500-epochs": "ViT_S16_500_epochs.csv",
    "ViT-S16-400-epochs": "ViT_S16_400_epochs.csv",
    "ViT-S16-300-epochs": "ViT_S16_300_epochs.csv",
    "ViT-S16-200-epochs": "ViT_S16_200_epochs.csv",
    "ViT-S16-100-epochs": "ViT_S16_100_epochs.csv",
}


def load_models_info(augmentation_variants,cmap="Reds"):
    colors = sns.color_palette(cmap, n_colors=len(augmentation_variants))#[::-1]
    models_info = {}
    for (label, file), color in zip(augmentation_variants.items(), colors):
        try:
            df = pd.read_csv(os.path.join(path, file))
            models_info[label] = {"df": df, "color": color}
        except FileNotFoundError:
            print(f"Warning: File not found: {file}")
    return models_info


def plot_metrics(models_info, title_prefix,):
    # Plot R^2 scores:
    plt.figure(figsize=(6.4, 6.4))
    for label, info in models_info.items():
        df = info["df"]
        color = info["color"]
        plt.plot(df["epoch"],1-df["test_r2"], c=color, linestyle="-")
        plt.plot(df["epoch"],1-df["train_r2"], c=color, linestyle="--", alpha=0.5)
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
    plt.legend(handles=legend_elements, fontsize=8, title="Run", frameon=False)
    plt.xlabel("Epochs")
    plt.ylabel(r"$1-R^2$ Score")
    # plt.ylim(0.99, 1.001)
    # plt.xlim(40, 1100)
    plt.xscale('log')
    plt.yscale('log')
    # plt.xticks([100, 200, 300, 400, 500, 1000], [100, 200, 300, 400, 500, 1000])
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.title(f"{title_prefix} Extra long run: 1-R²")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{title_prefix.lower()}_extralong_r2.pdf"), bbox_inches='tight')

    # Plot MSE:
    plt.figure(figsize=(6.4, 6.4))
    for label, info in models_info.items():
        df = info["df"]
        color = info["color"]
        plt.plot(df["epoch"], df["test_mse"], c=color, linestyle="-")
        plt.plot(df["epoch"], df["train_mse"], c=color, linestyle="--", alpha=0.5)
    plt.legend(handles=legend_elements, fontsize=8, title="Run", frameon=False)
    plt.xlabel("Epochs")
    plt.ylabel("MSE (Lattice Units)")
    plt.yscale("log")
    plt.xscale("log")
    # plt.xlim(40, 1100)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.title(f"{title_prefix} Long runs: MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{title_prefix.lower()}_extralong_mse.pdf"), bbox_inches='tight')

    # Print header once:
    print(f"{'Augmentation':<15} {'Test R²':>10} {'Train R²':>10} {'Test MSE':>12} {'Train MSE':>12}")
    print("-" * 62)
    for label, info in models_info.items():
        df = info["df"]
        idx_max = np.argmax(df["test_r2"])
        row = df.iloc[idx_max]
        print(f"{label:<15} {row['test_r2']:.5f} {row['train_r2']:.5f} "
              f"{row['test_mse']:.6f} {row['train_mse']:.6f}")

plot_metrics(load_models_info(runs,"Reds"), "ViT-S16")

runs_convnexttiny = {
    "ConvNext-T-600-epochs": "ConvNeXtTiny_600_epochs.csv",
    "ConvNext-T-500-epochs": "ConvNeXtTiny_500_epochs.csv",
    "ConvNext-T-400-epochs": "ConvNeXtTiny_400_epochs.csv",
    "ConvNext-T-300-epochs": "ConvNeXtTiny_300_epochs.csv",
    "ConvNext-T-200-epochs": "ConvNeXtTiny_200_epochs.csv",
    "ConvNext-T-100-epochs": "ConvNeXtTiny_100_epochs.csv",
}
plot_metrics(load_models_info(runs_convnexttiny,"Blues"), "ConvNeXtTiny")

runs_convnextsmall = {
    "ConvNext-S-500-epochs": "convnextsmall_extralong.csv",
    "ConvNeXt-S-300-epochs": "ConvNeXtSmall_300_epochs.csv",
    "ConvNeXt-S-200-epochs": "ConvNeXtSmall_200_epochs.csv",
    "ConvNeXt-S-100-epochs": "ConvNeXtSmall_100_epochs.csv",
}
plot_metrics(load_models_info(runs_convnextsmall,"Greens"), "ConvNeXtSmall")

runs_vit_t16 = {
    "ViT-B16-500-epochs": "vit_b16_extralong.csv",
    "ViT-t16-600-epochs": "ViT_T16_600_epochs.csv",
    "ViT-t16-500-epochs": "ViT_T16_500_epochs.csv",
    "ViT-t16-400-epochs": "ViT_T16_400_epochs.csv",
    "ViT-t16-300-epochs": "ViT_T16_300_epochs.csv",
    "ViT-t16-200-epochs": "ViT_T16_200_epochs.csv",
    "ViT-t16-100-epochs": "ViT_T16_100_epochs.csv",
}
plot_metrics(load_models_info(runs_vit_t16,"Purples"), "ViT-T16")