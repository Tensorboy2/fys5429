import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Set a nice seaborn theme
sns.set_theme(style="whitegrid")

# Read data
path = os.path.dirname(__file__)
# vit_test = pd.read_csv(os.path.join(path, "vit_b16_metrics_vit_test.csv"))
# vit_test_2 = pd.read_csv(os.path.join(path, "vit_b16_metrics_vit_test_2.csv"))
vit_test_3 = pd.read_csv(os.path.join(path, "vit_b16_metrics_vit_test_3.csv"))

resnet50 = pd.read_csv(os.path.join(path, "resnet50_metrics_vit_test.csv"))
resnet101 = pd.read_csv(os.path.join(path, "resnet101_metrics_vit_test.csv"))
# resnet152 = pd.read_csv(os.path.join(path, "resnet152_metrics_vit_test.csv"))

convnexttiny = pd.read_csv(os.path.join(path, "convnexttiny_metrics_all_convnext.csv"))
convnextsmall = pd.read_csv(os.path.join(path, "convnextsmall_metrics_all_convnext.csv"))

# Separate colormaps
vit_colors = cm.plasma(np.linspace(0.7, 0.8, 1))       # 3 ViT runs
resnet_colors = cm.viridis(np.linspace(0.3, 0.8, 2))    # 3 ResNets
convnext_colors = cm.cividis(np.linspace(0.3, 0.8, 2))  # 2 ConvNeXts

### ---------- Plot R² Scores ---------- ###
plt.figure(figsize=(10, 6))

# ViT runs
for i, df in enumerate([vit_test_3]):
    plt.plot(df["epoch"], df["test_r2"], c=vit_colors[i], label=f"Run {i+1} - Test R²", linestyle="-")
    plt.plot(df["epoch"], df["train_r2"], c=vit_colors[i], label=f"Run {i+1} - Train R²", linestyle="--", alpha=0.5)

# ResNets
for i, (df, name) in enumerate(zip([resnet50, resnet101,2], ["ResNet50", "ResNet101"])):
    plt.plot(df["epoch"], df["test_r2"], c=resnet_colors[i], label=f"{name} - Test R²", linestyle="-")
    plt.plot(df["epoch"], df["train_r2"], c=resnet_colors[i], label=f"{name} - Train R²", linestyle="--", alpha=0.5)

# ConvNeXts
for i, (df, name) in enumerate(zip([convnexttiny, convnextsmall], ["ConvNeXt-Tiny", "ConvNeXt-Small"])):
    plt.plot(df["epoch"], df["test_r2"], c=convnext_colors[i], label=f"{name} - Test R²", linestyle="-")
    plt.plot(df["epoch"], df["train_r2"], c=convnext_colors[i], label=f"{name} - Train R²", linestyle="--", alpha=0.5)

# Axes & layout
plt.ylim(0.9, 1)
plt.xlim(0, 31)
plt.xlabel("Epochs")
plt.ylabel("R² Score")
legend_elements = [
    # Line2D([0], [0], color=vit_colors[0], lw=2, label='Run 1'),
    # Line2D([0], [0], color=vit_colors[1], lw=2, label='Run 2'),
    Line2D([0], [0], color=vit_colors[0], lw=2, label='ViT-B16'),
    Line2D([0], [0], color=resnet_colors[0], lw=2, label='ResNet50'),
    Line2D([0], [0], color=resnet_colors[1], lw=2, label='ResNet101'),
    # Line2D([0], [0], color=resnet_colors[2], lw=2, label='ResNet152'),
    Line2D([0], [0], color=convnext_colors[0], lw=2, label='ConvNeXtTiny'),
    Line2D([0], [0], color=convnext_colors[1], lw=2, label='ConvNeXtSmall'),
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='Train')
]
plt.legend(handles=legend_elements,title="Models", fontsize=8)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()

### ---------- Plot MSE Scores ---------- ###
plt.figure(figsize=(10, 6))

# ViT runs
for i, df in enumerate([vit_test_3]):
    plt.plot(df["epoch"], df["test_mse"], c=vit_colors[i], label=f"Run {i+1} - Test MSE", linestyle="-")
    plt.plot(df["epoch"], df["train_mse"], c=vit_colors[i], label=f"Run {i+1} - Train MSE", linestyle="--", alpha=0.5)

# ResNets
for i, (df, name) in enumerate(zip([resnet50, resnet101], ["ResNet50", "ResNet101"])):
    plt.plot(df["epoch"], df["test_mse"], c=resnet_colors[i], label=f"{name} - Test MSE", linestyle="-")
    plt.plot(df["epoch"], df["train_mse"], c=resnet_colors[i], label=f"{name} - Train MSE", linestyle="--", alpha=0.5)

# ConvNeXts
for i, (df, name) in enumerate(zip([convnexttiny, convnextsmall], ["ConvNeXt-Tiny", "ConvNeXt-Small"])):
    plt.plot(df["epoch"], df["test_mse"], c=convnext_colors[i], label=f"{name} - Test MSE", linestyle="-")
    plt.plot(df["epoch"], df["train_mse"], c=convnext_colors[i], label=f"{name} - Train MSE", linestyle="--", alpha=0.5)

# Axes & layout
plt.xlim(-1, 31)
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(handles=legend_elements, title="Models", fontsize=8)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()

plt.show()
