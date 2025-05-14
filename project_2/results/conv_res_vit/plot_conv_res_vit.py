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

models_info = {
    "ViT-B16": {
        "file": "vit_b16_metrics_conv_res_vit.csv",
        "color": cm.plasma(0.75)
    },
    "ResNet50": {
        "file": "resnet50_metrics_conv_res_vit.csv",
        "color": cm.viridis(0.4)
    },
    "ResNet101": {
        "file": "resnet101_metrics_conv_res_vit.csv",
        "color": cm.viridis(0.7)
    },
    "ConvNeXt-Tiny": {
        "file": "convnexttiny_metrics_conv_res_vit.csv",
        "color": cm.cividis(0.4)
    },
    "ConvNeXt-Small": {
        "file": "convnextsmall_metrics_conv_res_vit.csv",
        "color": cm.cividis(0.7)
    }
}


for model in models_info:
    models_info[model]["df"] = pd.read_csv(os.path.join(path, models_info[model]["file"]))

plt.figure(figsize=(10, 6))

for name, info in models_info.items():
    df = info["df"]
    color = info["color"]
    plt.plot(df["epoch"], df["test_r2"], c=color, label=f"{name} - Test R²", linestyle="-")
    plt.plot(df["epoch"], df["train_r2"], c=color, label=f"{name} - Train R²", linestyle="--", alpha=0.5)

legend_elements = [
    Line2D([0], [0], color=info["color"], lw=2, label=name)
    for name, info in models_info.items()
] + [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='Train')
]

plt.legend(handles=legend_elements, title="Models", fontsize=8)
plt.xlabel("Epochs")
plt.ylabel("R² Score")
plt.yscale("log")
plt.xscale("log")

# plt.xlim(0, 31)
plt.ylim(0.9, 1)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))

for name, info in models_info.items():
    df = info["df"]
    color = info["color"]
    plt.plot(df["epoch"], df["test_mse"], c=color, label=f"{name} - Test MSE", linestyle="-")
    plt.plot(df["epoch"], df["train_mse"], c=color, label=f"{name} - Train MSE", linestyle="--", alpha=0.5)

legend_elements = [
    Line2D([0], [0], color=info["color"], lw=2, label=name)
    for name, info in models_info.items()
] + [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='Train')
]

plt.legend(handles=legend_elements, title="Models", fontsize=8)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.yscale("log")
plt.xscale("log")

# plt.xlim(0, 31)
# plt.ylim(0.9, 1)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

