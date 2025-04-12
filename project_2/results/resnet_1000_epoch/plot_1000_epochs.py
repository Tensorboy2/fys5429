import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Optional: set a nice style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

# Read data
path = os.path.dirname(__file__)

resnet50_1000_epoch = pd.read_csv(os.path.join(path, "resnet50_1000_epochs_metrics.csv"))
resnet101_1000_epoch = pd.read_csv(os.path.join(path, "resnet101_1000_epochs_metrics.csv"))
# Add model identifier
resnet50_1000_epoch["model"] = "ResNet50"
resnet101_1000_epoch["model"] = "ResNet101"

# Combine both into one DataFrame
combined_df = pd.concat([resnet50_1000_epoch, resnet101_1000_epoch], ignore_index=True)

# Melt the DataFrame for R2 and MSE
r2_df = pd.melt(combined_df, id_vars=["epoch", "model"], value_vars=["train_r2", "test_r2"],
                var_name="dataset", value_name="r2")
r2_df["dataset"] = r2_df["dataset"].str.replace("_r2", "")

mse_df = pd.melt(combined_df, id_vars=["epoch", "model"], value_vars=["train_mse", "test_mse"],
                 var_name="dataset", value_name="mse")
mse_df["dataset"] = mse_df["dataset"].str.replace("_mse", "")



# Define save path once
save_dir = "/home/sigvar/2_semester/fys5429/project_2/results/resnet_1000_epoch"

# plt.figure(figsize=(10, 6))
# sns.lineplot(data=resnet50_1000_epoch, x="epoch", y="train_r2", label="Train", linewidth=2, alpha=0.6)
# sns.lineplot(data=resnet50_1000_epoch, x="epoch", y="test_r2", label="Test", color="crimson", linewidth=2)
# sns.lineplot(data=resnet101_1000_epoch, x="epoch", y="train_r2", label="Train", linewidth=2, alpha=0.6)
# sns.lineplot(data=resnet101_1000_epoch, x="epoch", y="test_r2", label="Test", color="crimson", linewidth=2)

# plt.title("Train vs Test R2 over Epochs", fontsize=16)
# plt.ylabel("R2 Score")
# plt.xlabel("Epochs")
# plt.xscale("log")
# plt.yscale("log")
# plt.legend(loc="lower right")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "1000_epochs_r2.pdf"))
# plt.close()
plt.figure(figsize=(10, 6))
sns.lineplot(data=r2_df, x="epoch", y="r2", hue="model", style="dataset", linewidth=2)
plt.title("Train vs Test R² over Epochs")
plt.xlabel("Epochs")
plt.ylabel("R² Score")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "1000_epochs_r2_reframed.pdf"))
plt.close()


# plt.figure(figsize=(10, 6))
# sns.lineplot(data=resnet50_1000_epoch, x="epoch", y="train_mse", label="Train", linewidth=2, alpha=0.6)
# sns.lineplot(data=resnet50_1000_epoch, x="epoch", y="test_mse", label="Test", color="crimson", linewidth=2)
# sns.lineplot(data=resnet101_1000_epoch, x="epoch", y="train_mse", label="Train", linewidth=2, alpha=0.6)
# sns.lineplot(data=resnet101_1000_epoch, x="epoch", y="test_mse", label="Test", color="crimson", linewidth=2)

# plt.title("Train vs Test MSE over Epochs", fontsize=16)
# plt.ylabel("Mean Squared Error")
# plt.xlabel("Epochs")
# plt.xscale("log")
# plt.yscale("log")
# plt.legend(loc="upper right")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "1000_epochs_mse.pdf"))
# plt.close()
plt.figure(figsize=(10, 6))
sns.lineplot(data=mse_df, x="epoch", y="mse", hue="model", style="dataset", linewidth=2)
plt.title("Train vs Test MSE over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "1000_epochs_mse_reframed.pdf"))
plt.close()



# fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# sns.lineplot(ax=axes[0], data=resnet50_1000_epoch, x="epoch", y="train_r2", label="Train ResNet50", linewidth=2, alpha=0.6)
# sns.lineplot(ax=axes[0], data=resnet50_1000_epoch, x="epoch", y="test_r2", label="Test ResNet50", linewidth=2)
# sns.lineplot(ax=axes[0], data=resnet101_1000_epoch, x="epoch", y="train_r2", label="Train ResNet101", linewidth=2, alpha=0.6)
# sns.lineplot(ax=axes[0], data=resnet101_1000_epoch, x="epoch", y="test_r2", label="Test ResNet101", linewidth=2)
# # axes[0].set_title("Train vs Test R2 over Epochs", fontsize=16)
# axes[0].set_ylabel("R2")
# axes[0].set_xscale("log")
# axes[0].set_yscale("log")
# axes[0].legend(loc="lower right")
# axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)

# sns.lineplot(ax=axes[1], data=resnet50_1000_epoch, x="epoch", y="train_mse", linewidth=2, alpha=0.6)
# sns.lineplot(ax=axes[1], data=resnet50_1000_epoch, x="epoch", y="test_mse", linewidth=2)
# sns.lineplot(ax=axes[1], data=resnet101_1000_epoch, x="epoch", y="train_mse", linewidth=2, alpha=0.6)
# sns.lineplot(ax=axes[1], data=resnet101_1000_epoch, x="epoch", y="test_mse", linewidth=2)
# # axes[1].set_title("Train vs Test MSE over Epochs", fontsize=16)
# axes[1].set_ylabel("MSE")
# axes[1].set_xlabel("Epochs")
# axes[1].set_xscale("log")
# axes[1].set_yscale("log")
# # axes[1].legend(loc="upper right")
# axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)

# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "1000_epochs_combined.pdf"))
# plt.show()
# Add a 'metric' column for stacking
r2_df["metric"] = "R2"
mse_df["metric"] = "MSE"
mse_df = mse_df.rename(columns={"mse": "value"})
r2_df = r2_df.rename(columns={"r2": "value"})

full_df = pd.concat([r2_df, mse_df], ignore_index=True)

g = sns.FacetGrid(full_df, row="metric", height=5, aspect=1.8, sharex=True, sharey=True)
g.map_dataframe(sns.lineplot, x="epoch", y="value", hue="model", style="dataset", linewidth=2)
g.set(xscale="log", yscale="log")
g.add_legend()
g.set_axis_labels("Epochs","")
g.set_titles("{row_name}")
plt.tight_layout()
g.savefig(os.path.join(save_dir, "1000_epochs_combined_reframed.pdf"))
