import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Model names
model_names = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000", "full"]

# Storage for all models
all_data = []

# Read and parse each model's file
for model_name in model_names:
    try:
        with open(f"/home/sigvar/2_semester/fys5429/project_2/results/diff_data_amounts/train_data_resnet50_{model_name}.txt", "r") as f:
            lines = f.readlines()

        for i in range(0, 1000, 3):  # every 3 lines = 1 epoch block
            try:
                epoch_line = lines[i].strip()
                train_line = lines[i + 1].strip()
                test_line = lines[i + 2].strip()

                epoch = int(re.search(r'Epoch (\d+)', epoch_line).group(1))
                train_mse_val = float(re.search(r'MSE = ([\d.]+)', train_line).group(1))
                train_r2_val = float(re.search(r'R2 = ([\d.-]+)', train_line).group(1))
                test_mse_val = float(re.search(r'MSE = ([\d.]+)', test_line).group(1))
                test_r2_val = float(re.search(r'R2 = ([\d.-]+)', test_line).group(1))

                all_data.append({
                    "Data points:": model_name,
                    "epoch": epoch,
                    "train_mse": train_mse_val,
                    "train_r2": train_r2_val,
                    "test_mse": test_mse_val,
                    "test_r2": test_r2_val
                })

            except Exception as e:
                print(f"Skipping block at line {i} in {model_name}: {e}")

    except FileNotFoundError:
        print(f"File for {model_name} not found!")

# Convert to DataFrame
df = pd.DataFrame(all_data)
df.to_csv("/home/sigvar/2_semester/fys5429/project_2/results/diff_data_amounts/resnet50_diff_data_points.csv")
# Melt R2 values
df_r2 = df.melt(id_vars=["epoch", "Data points:"], value_vars=["train_r2", "test_r2"],
                var_name="type", value_name="R2")

# R2 over epochs plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_r2[df_r2['type'] == 'test_r2'], x="epoch", y="R2",
             hue="Data points:", palette="viridis", linewidth=2, markers=False)
plt.title("Test R2 over Epochs for different data set sizes")
plt.grid(True)
plt.ylim(bottom=0)
plt.xscale("log")
plt.tight_layout()
plt.savefig("/home/sigvar/2_semester/fys5429/project_2/plots/r2_diff_data_pointss.pdf")

# --- NEW: Heatmap plot of test R² ---
# Pivot the data to prepare for heatmap
heatmap_df = df.pivot(index="epoch", columns="Data points:", values="test_r2")

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_df, cmap="viridis", cbar_kws={"label": "Test R2"}, linewidths=1.0)
plt.title("Test R2 Heatmap: Epoch vs Data points")
plt.xlabel("Data Points")
plt.ylabel("Epoch")
plt.tight_layout()
plt.savefig("/home/sigvar/2_semester/fys5429/project_2/plots/heatmap_r2_epochs_vs_data.pdf")
