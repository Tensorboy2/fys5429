import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Model names
model_names = ["resnet50","resnet50_wd0.01","resnet50_wd0.05","resnet101"]

# Storage for all models
all_data = []

# Read and parse each model's file
for model_name in model_names:
    epochs = []
    train_mse = []
    train_r2 = []
    test_mse = []
    test_r2 = []
    try:
        with open(f"/home/sigvar/2_semester/fys5429/project_2/results/train_data_{model_name}.txt", "r") as f:
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

                # Append to storage
                all_data.append({
                    "model": model_name,
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

# Melt data for seaborn
df_mse = df.melt(id_vars=["epoch", "model"], value_vars=["train_mse", "test_mse"],
                 var_name="type", value_name="MSE")

df_r2 = df.melt(id_vars=["epoch", "model"], value_vars=["train_r2", "test_r2"],
                var_name="type", value_name="R2")

# Seaborn MSE plot
plt.figure(figsize=(10, 6))
# sns.lineplot(data=df_mse, x="epoch", y="MSE", hue="model", style="type", markers=False)
# Plot the train lines with reduced visibility
sns.lineplot(data=df_mse[df_mse['type'] == 'train_mse'], x="epoch", y="MSE", hue="model", linewidth=1, alpha=0.5, markers=False, legend=False)

# Plot the test lines with normal visibility
sns.lineplot(data=df_mse[df_mse['type'] == 'test_mse'], x="epoch", y="MSE", hue="model", linewidth=2, markers=False)
plt.title("Train/Test MSE over Epochs")
# plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/sigvar/2_semester/fys5429/project_2/plots/mse_all_models.pdf")

# Seaborn R2 plot
plt.figure(figsize=(10, 6))
# sns.lineplot(data=df_r2, x="epoch", y="R2", hue="model", style="type", markers=False)
# Plot the train lines with reduced visibility
sns.lineplot(data=df_r2[df_r2['type'] == 'train_r2'], x="epoch", y="R2", hue="model",palette="viridis", linewidth=1, alpha=0.5, markers=False, legend=False)

# Plot the test lines with normal visibility
sns.lineplot(data=df_r2[df_r2['type'] == 'test_r2'], x="epoch", y="R2", hue="model",palette="viridis", linewidth=2, markers=False)
plt.title("Train/Test R2 over Epochs")
plt.grid(True)
plt.ylim(bottom=0.5)
plt.xscale("log")
# plt.yscale("log")
plt.tight_layout()
plt.savefig("/home/sigvar/2_semester/fys5429/project_2/plots/r2_all_models.pdf")
