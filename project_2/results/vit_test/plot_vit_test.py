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

colors = cm.viridis(np.linspace(0.1, 0.9, 3))

# Create a figure
plt.figure(figsize=(10, 6))

# Plot all curves with clear labeling
plt.plot(vit_test["epoch"], vit_test["test_r2"],c=colors[0], label="Run 1 - Test R²", linestyle="-")
plt.plot(vit_test["epoch"], vit_test["train_r2"],c=colors[0], label="Run 1 - Train R²", linestyle="--",alpha=0.5)
plt.plot(vit_test_2["epoch"], vit_test_2["test_r2"],c=colors[1], label="Run 2 - Test R²", linestyle="-")
plt.plot(vit_test_2["epoch"], vit_test_2["train_r2"],c=colors[1], label="Run 2 - Train R²", linestyle="--",alpha=0.5)
plt.plot(vit_test_3["epoch"], vit_test_3["test_r2"],c=colors[2], label="Run 3 - Test R²", linestyle="-")
plt.plot(vit_test_3["epoch"], vit_test_3["train_r2"],c=colors[2], label="Run 3 - Train R²", linestyle="--",alpha=0.5)

# Customize axes
# plt.xscale("log")
# plt.yscale("log")
plt.ylim(0.8,1)
plt.xlim(0,31)
plt.xlabel("Epochs")
plt.ylabel("R² Score")
# plt.title("ViT R² Scores Across Epochs")
legend_elements = [
    Line2D([0], [0], color=colors[0], lw=2, label='Run 1'),
    Line2D([0], [0], color=colors[1], lw=2, label='Run 2'),
    Line2D([0], [0], color=colors[2], lw=2, label='Run 3'),
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='Train')
]
plt.legend(handles=legend_elements, title="Legend")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

# Improve layout
plt.tight_layout()
# plt.show()

# Create a figure
plt.figure(figsize=(10, 6))

# Plot all curves with clear labeling
plt.plot(vit_test["epoch"], vit_test["test_mse"], label="Run 1 - Test MSE", linestyle="-",c=colors[0])
plt.plot(vit_test["epoch"], vit_test["train_mse"], label="Run 1 - Train MSE", linestyle="--",c=colors[0],alpha=0.5)
plt.plot(vit_test_2["epoch"], vit_test_2["test_mse"], label="Run 2 - Test MSE", linestyle="-",c=colors[1])
plt.plot(vit_test_2["epoch"], vit_test_2["train_mse"], label="Run 2 - Train MSE", linestyle="--",c=colors[1],alpha=0.5)
plt.plot(vit_test_3["epoch"], vit_test_3["test_mse"], label="Run 3 - Test MSE", linestyle="-",c=colors[2])
plt.plot(vit_test_3["epoch"], vit_test_3["train_mse"], label="Run 3 - Train MSE", linestyle="--",c=colors[2],alpha=0.5)

# Customize axes
# plt.xscale("log")
plt.xlim(-1,31)
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("MSE")
# plt.title("ViT MSE Epochs")
plt.legend(handles=legend_elements, title="Legend")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

# Improve layout
plt.tight_layout()
plt.show()
