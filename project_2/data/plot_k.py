import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
path = os.path.dirname(__file__)
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


k_all = np.load(os.path.join(path, "k.npz"), mmap_mode="r")
k_all = k_all['k']

# Sanity check:
print("Shape of k_all:", k_all.shape)  

k_00 = k_all[:, 0, 0]
k_01 = k_all[:, 0, 1]
k_10 = k_all[:, 1, 0]
k_11 = k_all[:, 1, 1]

components = {
    "k_xx": k_00,
    "k_xy": k_01,
    "k_yx": k_10,
    "k_yy": k_11
}

# Optional: Apply smoothing (rolling mean)
window = 200  # Adjust as needed
components_smoothed = {
    name: pd.Series(data).rolling(window, center=True).mean()
    for name, data in components.items()
}

# Create a figure for line plots
# plt.figure(figsize=(12, 6))
# for name, smoothed in components_smoothed.items():
#     plt.plot(smoothed, label=f"{name} (smoothed)")
# plt.title("Smoothed Line Plot of k_ij Components")
# plt.xlabel("Sample Index")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Create histograms:
plt.figure(figsize=(6.4, 6.4))
for i, (name, data) in enumerate(components.items(), 1):
    plt.subplot(2, 2, i)
    sns.histplot(data, kde=True, bins=100)
    plt.title(f"{name}")
    plt.xlabel("Permeability")
    plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(path,"permeability_distribution.pdf"))
plt.show()

