import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib as mpl

mpl.rcParams.update({
    # "text.usetex": True,  # Requires LaTeX installed
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# Set a nice seaborn theme
sns.set_theme(style="whitegrid")

# Read data
path = os.path.dirname(__file__)

models_info = {
    "ViT-B16": {
        "file": "ViT_B16_metrics_vit.csv",
        "color": cm.plasma(0.75)
    },
    # "ResNet50": {
    #     "file": "resnet50_metrics_conv_res_vit.csv",
    #     "color": cm.viridis(0.4)
    # },
    "ResNet101": {
        "file": "resnet101_metrics_conv_res_vit.csv",
        "color": cm.viridis(0.7)
    },
    "ConvNeXt-Tiny": {
        "file": "convnexttiny_metrics_conv_res.csv",
        "color": cm.cividis(0.4)
    },
    "ConvNeXt-Small": {
        "file": "convnextsmall_metrics_conv_res.csv",
        "color": cm.cividis(0.7)
    },
    "ResNet50": {
        "file": "resnet50_metrics_conv_res.csv",
        "color": cm.viridis(0.0)
    }
}


for model in models_info:
    models_info[model]["df"] = pd.read_csv(os.path.join(path, models_info[model]["file"]))

fig_width, fig_height = 6.4, 6.4  # or 6.4 for full width
plt.figure(figsize=(fig_width, fig_height))

for name, info in models_info.items():
    df = info["df"]
    color = info["color"]
    plt.plot(df["epoch"], df["test_r2"], c=color, linestyle="-")
    idx_max = np.argmax(df['test_r2'])
    print(f"model: {name}, test r2: {df['test_r2'][idx_max]:.5f}, train r2: {df['train_r2'][idx_max]:.5f}, test mse: {df['test_mse'][idx_max]:.6f}, train mse: {df['train_mse'][idx_max]:.6f}, ")

    plt.plot(df["epoch"], df["train_r2"], c=color, linestyle="--", alpha=0.5)

legend_elements = [
    Line2D([0], [0], color=info["color"], lw=2, label=name)
    for name, info in models_info.items()
] + [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.5, label='Train')
]

plt.legend(handles=legend_elements, fontsize=8, title="Models", frameon=False)
plt.xlabel("Epochs")
plt.ylabel(r"$R^2$ Score")
# plt.yscale("log")
plt.ylim(0.92, 1)
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, "r2.pdf"), bbox_inches='tight')
# plt.show()

fig_width, fig_height = 6.4, 6.4  # or 6.4 for full width
plt.figure(figsize=(fig_width, fig_height))

for name, info in models_info.items():
    df = info["df"]
    color = info["color"]
    plt.plot(df["epoch"], df["test_mse"], c=color, linestyle="-")
    plt.plot(df["epoch"], df["train_mse"], c=color, linestyle="--", alpha=0.5)

legend_elements = [
    Line2D([0], [0], color=info["color"], lw=2, label=name)
    for name, info in models_info.items()
] + [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.5, label='Train')
]

plt.legend(handles=legend_elements, fontsize=8, title="Models", frameon=False)
plt.xlabel("Epochs")
plt.ylabel(r"MSE Score")
plt.yscale("log")
# plt.ylim(0.99, 1)
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, "mse.pdf"), bbox_inches='tight')

# plt.show()

