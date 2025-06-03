'''
plot_diff_num_datapoints.py

Module for plotting the R^2 accuracy, the mean square error and the max R^2 for different dataset sizes.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.lines import Line2D
import matplotlib as mpl
path = os.path.dirname(__file__)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

sns.set_theme(style="whitegrid")


models = {
    "ConvNeXtSmall": { 
        "2000": "convnextsmall_2000_metrics_diff_num_datapoints.csv",
        "4000": "convnextsmall_4000_metrics_diff_num_datapoints.csv",
        "6000": "convnextsmall_6000_metrics_diff_num_datapoints.csv",
        "8000": "convnextsmall_8000_metrics_diff_num_datapoints.csv",
        "10000": "convnextsmall_10000_metrics_diff_num_datapoints.csv",
        "12000": "convnextsmall_12000_metrics_diff_num_datapoints.csv",
        "14000": "convnextsmall_14000_metrics_diff_num_datapoints.csv",
        "16000": "convnextsmall_16000_metrics_diff_num_datapoints.csv",
        "18000": "convnextsmall_18000_metrics_diff_num_datapoints.csv",
        "20000": "convnextsmall_metrics_all_models_2.csv",
    },
    "ConvNeXtTiny": { 
        "2000": "convnexttiny_2000_metrics_diff_num_datapoints.csv",
        "4000": "convnexttiny_4000_metrics_diff_num_datapoints.csv",
        "6000": "convnexttiny_6000_metrics_diff_num_datapoints.csv",
        "8000": "convnexttiny_8000_metrics_diff_num_datapoints.csv",
        "10000": "convnexttiny_10000_metrics_diff_num_datapoints.csv",
        "12000": "convnexttiny_12000_metrics_diff_num_datapoints.csv",
        "14000": "convnexttiny_14000_metrics_diff_num_datapoints.csv",
        "16000": "convnexttiny_16000_metrics_diff_num_datapoints.csv",
        "18000": "convnexttiny_18000_metrics_diff_num_datapoints.csv",
        "20000": "convnexttiny_metrics_all_models_2.csv",
    },
}

# Color maps per model:
colormaps = {
    "ConvNeXtSmall": plt.cm.viridis,
    "ConvNeXtTiny": plt.cm.plasma,
}

# Normalize based on number of datapoints:
datapoint_keys = list(models["ConvNeXtSmall"].keys())
datapoint_values = np.array([int(k) for k in datapoint_keys])
norm = plt.Normalize(vmin=datapoint_values.min(), vmax=datapoint_values.max())

# Store all model info:
model_data = {}
for model_name, files in models.items():
    model_data[model_name] = []
    for datapoints, filename in files.items():
        df = pd.read_csv(os.path.join(path, filename))
        dp_int = int(datapoints)
        color = colormaps[model_name](norm(dp_int))
        model_data[model_name].append({
            "datapoints": dp_int,
            "df": df,
            "color": color
        })


# Plot R^2:
plt.figure(figsize=(6.4, 6.4))
for i, (model_name, runs) in enumerate(model_data.items()):
    ax = plt.subplot(1, 2, i + 1)
    legend_lines = []

    for run in runs:
        df = run["df"]
        color = run["color"]
        datapoints = run["datapoints"]

        # Plot lines:
        ax.plot(df["epoch"], df["test_r2"], c=color, linestyle="-")
        ax.plot(df["epoch"], df["train_r2"], c=color, linestyle="--", alpha=0.5)

        # Store legend handles for datapoints:
        legend_lines.append(Line2D([0], [0], color=color, lw=2, label=f"{datapoints}"))

    ax.legend(handles=legend_lines, title=model_name, frameon=True)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(r"$R^2$ Score")
    ax.set_ylim(0.92, 1)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, "diff_num_datapoints_r2.pdf"), bbox_inches='tight')


# Plot MSE:
plt.figure(figsize=(6.4, 6.4))
for i, (model_name, runs) in enumerate(model_data.items()):
    ax = plt.subplot(1, 2, i + 1)
    legend_lines = []

    for run in runs:
        df = run["df"]
        color = run["color"]
        datapoints = run["datapoints"]

        ax.plot(df["epoch"], df["test_mse"], c=color, linestyle="-")
        ax.plot(df["epoch"], df["train_mse"], c=color, linestyle="--", alpha=0.5)

        legend_lines.append(Line2D([0], [0], color=color, lw=2, label=f"{datapoints}"))

    ax.legend(handles=legend_lines, title=model_name, frameon=True)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE (Lattice Units)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, "diff_num_datapoints_mse.pdf"), bbox_inches='tight')


# Plot max R^2 for each dataset size:
data_records = []
# Collect all data points into a list of dicts:
for model_name, runs in model_data.items():
    for run in runs:
        df = run["df"]
        num_data = int(run["datapoints"])
        max_r2 = df["test_r2"].max()
        
        data_records.append({
            "model": model_name,
            "num_datapoints": num_data,
            "max_r2": max_r2,
            "color": run["color"],
        })

df_points = pd.DataFrame(data_records)
plt.figure(figsize=(6.4, 6.4))
sns.scatterplot(
    data=df_points,
    x="num_datapoints",
    y="max_r2",
    hue="model"
)
plt.ylabel(r"$R^2$ Score")
plt.xlabel("Num Datapoints")
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(path, "diff_num_datapoints_fit.pdf"), bbox_inches='tight')