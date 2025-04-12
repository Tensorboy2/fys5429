import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Optional: set a nice style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

# Read data
path = os.path.dirname(__file__)

resnet50 = pd.read_csv(os.path.join(path, "resnet50_metrics.csv"))
resnet50v2 = pd.read_csv(os.path.join(path, "resnet50v2_metrics.csv"))
# Add model identifier
resnet50["model"] = "ResNet50"
resnet50v2["model"] = "ResNet50v2"

# Combine both into one DataFrame
combined_df = pd.concat([resnet50, resnet50v2], ignore_index=True)

# Melt the DataFrame for R2 and MSE
r2_df = pd.melt(combined_df, id_vars=["epoch", "model"], value_vars=["train_r2", "test_r2"],
                var_name="dataset", value_name="r2")
r2_df["dataset"] = r2_df["dataset"].str.replace("_r2", "")

mse_df = pd.melt(combined_df, id_vars=["epoch", "model"], value_vars=["train_mse", "test_mse"],
                 var_name="dataset", value_name="mse")
mse_df["dataset"] = mse_df["dataset"].str.replace("_mse", "")

save_dir = "/home/sigvar/2_semester/fys5429/project_2/results/resnet50_vs_resnet50v2"

r2_df["metric"] = "R2"
mse_df["metric"] = "MSE"
mse_df = mse_df.rename(columns={"mse": "value"})
r2_df = r2_df.rename(columns={"r2": "value"})

full_df = pd.concat([r2_df, mse_df], ignore_index=True)
print(r2_df)
plt.figure(figsize=(10, 6))
sns.lineplot(data=r2_df, x="epoch", y="value", hue="model", style="dataset", linewidth=2)
plt.title("Train vs Test R² over Epochs")
plt.xlabel("Epochs")
plt.ylabel("R² Score")
plt.xscale("log")
# plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "50_vs_50v2_r2.pdf"))
plt.close()

plt.figure(figsize=(10, 6))
sns.lineplot(data=mse_df, x="epoch", y="value", hue="model", style="dataset", linewidth=2)
plt.title("Train vs Test MSE over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "50_vs_50v2_mse.pdf"))
plt.close()

g = sns.FacetGrid(full_df, row="metric", height=5, aspect=1.8, sharex=True, sharey=False)
g.map_dataframe(sns.lineplot, x="epoch", y="value", hue="model", style="dataset", linewidth=2)
g.set(xscale="log", yscale="log")
g.add_legend()
g.set_axis_labels("Epochs","")
g.set_titles("{row_name}")
plt.tight_layout()
g.savefig(os.path.join(save_dir, "50_vs_50v2_metrics.pdf"))
