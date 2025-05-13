import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Load data
path = os.path.dirname(__file__)
k_all = np.load(os.path.join(path, "k.npz"), mmap_mode="r")

# Load the correct key if the npz file has named arrays
k_all = k_all['k']  # Change 'k' if your key is different

# Sanity check
print("Shape of k_all:", k_all.shape)  # Should be (20000, 2, 2)

# Extract each component
k_00 = k_all[:, 0, 0]
k_01 = k_all[:, 0, 1]
k_10 = k_all[:, 1, 0]
k_11 = k_all[:, 1, 1]

components = {
    "k_00": k_00,
    "k_01": k_01,
    "k_10": k_10,
    "k_11": k_11
}

# Optional: Apply smoothing (rolling mean)
window = 200  # Adjust as needed
components_smoothed = {
    name: pd.Series(data).rolling(window, center=True).mean()
    for name, data in components.items()
}

# Create a figure for line plots
plt.figure(figsize=(14, 6))
for name, smoothed in components_smoothed.items():
    plt.plot(smoothed, label=f"{name} (smoothed)")
plt.title("Smoothed Line Plot of k_ij Components")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create histograms
plt.figure(figsize=(14, 6))
for i, (name, data) in enumerate(components.items(), 1):
    plt.subplot(2, 2, i)
    sns.histplot(data, kde=True, bins=50)
    plt.title(f"Histogram of {name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Print and display statistics
print("\n=== k_ij Component Statistics ===")
for name, data in components.items():
    stats = {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
    }
    print(f"\n{name}:")
    for stat_name, value in stats.items():
        print(f"  {stat_name}: {value:.4f}")
