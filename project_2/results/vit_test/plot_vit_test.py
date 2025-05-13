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
vit_test_3 = pd.read_csv(os.path.join(path, "vit_b16_metrics_vit_test_3.csv"))
vit_test_2 = pd.read_csv(os.path.join(path, "vit_b16_metrics_vit_test_2.csv"))
vit_test = pd.read_csv(os.path.join(path, "vit_b16_metrics_vit_test.csv"))
resnet50 = pd.read_csv(os.path.join(path, "resnet50_metrics_vit_test.csv"))
resnet101 = pd.read_csv(os.path.join(path, "resnet101_metrics_vit_test.csv"))
resnet152 = pd.read_csv(os.path.join(path, "resnet152_metrics_vit_test.csv"))

# Separate colormaps for ViTs and ResNets
vit_colors = cm.viridis(np.linspace(0.2, 0.8, 3))    # 3 ViT runs
resnet_colors = cm.plasma(np.linspace(0.4, 0.8, 3))  # 2 ResNets


# Create a figure
plt.figure(figsize=(10, 6))

# Plot all curves with clear labeling
# ViT runs
plt.plot(vit_test["epoch"], vit_test["test_r2"], c=vit_colors[0], label="Run 1 - Test R²", linestyle="-")
plt.plot(vit_test["epoch"], vit_test["train_r2"], c=vit_colors[0], label="Run 1 - Train R²", linestyle="--", alpha=0.5)

plt.plot(vit_test_2["epoch"], vit_test_2["test_r2"], c=vit_colors[1], label="Run 2 - Test R²", linestyle="-")
plt.plot(vit_test_2["epoch"], vit_test_2["train_r2"], c=vit_colors[1], label="Run 2 - Train R²", linestyle="--", alpha=0.5)

plt.plot(vit_test_3["epoch"], vit_test_3["test_r2"], c=vit_colors[2], label="Run 3 - Test R²", linestyle="-")
plt.plot(vit_test_3["epoch"], vit_test_3["train_r2"], c=vit_colors[2], label="Run 3 - Train R²", linestyle="--", alpha=0.5)

# ResNets
plt.plot(resnet50["epoch"], resnet50["test_r2"], c=resnet_colors[0], label="ResNet50 - Test R²", linestyle="-")
plt.plot(resnet50["epoch"], resnet50["train_r2"], c=resnet_colors[0], label="ResNet50 - Train R²", linestyle="--", alpha=0.5)

plt.plot(resnet101["epoch"], resnet101["test_r2"], c=resnet_colors[1], label="ResNet101 - Test R²", linestyle="-")
plt.plot(resnet101["epoch"], resnet101["train_r2"], c=resnet_colors[1], label="ResNet101 - Train R²", linestyle="--", alpha=0.5)

plt.plot(resnet152["epoch"], resnet152["test_r2"], c=resnet_colors[2], label="ResNet152 - Test R²", linestyle="-")
plt.plot(resnet152["epoch"], resnet152["train_r2"], c=resnet_colors[2], label="ResNet152 - Train R²", linestyle="--", alpha=0.5)

# Customize axes
# plt.xscale("log")
# plt.yscale("log")
plt.ylim(0.8,1)
plt.xlim(0,31)
plt.xlabel("Epochs")
plt.ylabel("R² Score")
# plt.title("ViT R² Scores Across Epochs")
legend_elements = [
    Line2D([0], [0], color=vit_colors[0], lw=2, label='Run 1'),
    Line2D([0], [0], color=vit_colors[1], lw=2, label='Run 2'),
    Line2D([0], [0], color=vit_colors[2], lw=2, label='Run 3'),
    Line2D([0], [0], color=resnet_colors[0], lw=2, label='ResNet50'),
    Line2D([0], [0], color=resnet_colors[1], lw=2, label='ResNet101'),
    Line2D([0], [0], color=resnet_colors[2], lw=2, label='ResNet152'),
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='Train')
]

plt.legend(handles=legend_elements, title="Models:")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

# Improve layout
plt.tight_layout()
# plt.show()

# Create a figure
plt.figure(figsize=(10, 6))

# Plot all curves with clear labeling
plt.plot(vit_test["epoch"], vit_test["test_mse"], label="Run 1 - Test MSE", linestyle="-",c=vit_colors[0])
plt.plot(vit_test["epoch"], vit_test["train_mse"], label="Run 1 - Train MSE", linestyle="--",c=vit_colors[0],alpha=0.5)

plt.plot(vit_test_2["epoch"], vit_test_2["test_mse"], label="Run 2 - Test MSE", linestyle="-",c=vit_colors[1])
plt.plot(vit_test_2["epoch"], vit_test_2["train_mse"], label="Run 2 - Train MSE", linestyle="--",c=vit_colors[1],alpha=0.5)

plt.plot(vit_test_3["epoch"], vit_test_3["test_mse"], label="Run 3 - Test MSE", linestyle="-",c=vit_colors[2])
plt.plot(vit_test_3["epoch"], vit_test_3["train_mse"], label="Run 3 - Train MSE", linestyle="--",c=vit_colors[2],alpha=0.5)

plt.plot(resnet50["epoch"], resnet50["test_mse"], label="Run 3 - Test MSE", linestyle="-",c=resnet_colors[0])
plt.plot(resnet50["epoch"], resnet50["train_mse"], label="Run 3 - Train MSE", linestyle="--",c=resnet_colors[0],alpha=0.5)

plt.plot(resnet101["epoch"], resnet101["test_mse"], label="Run 3 - Test MSE", linestyle="-",c=resnet_colors[1])
plt.plot(resnet101["epoch"], resnet101["train_mse"], label="Run 3 - Train MSE", linestyle="--",c=resnet_colors[1],alpha=0.5)

plt.plot(resnet152["epoch"], resnet152["test_mse"], label="Run 3 - Test MSE", linestyle="-",c=resnet_colors[1])
plt.plot(resnet152["epoch"], resnet152["train_mse"], label="Run 3 - Train MSE", linestyle="--",c=resnet_colors[1],alpha=0.5)

# Customize axes
# plt.xscale("log")
plt.xlim(-1,31)
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("MSE")
# plt.title("ViT MSE Epochs")
plt.legend(handles=legend_elements, title="Models:")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

# Improve layout
plt.tight_layout()
plt.show()
