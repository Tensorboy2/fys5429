import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import os
from matplotlib.lines import Line2D
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

sns.set_theme(style="whitegrid")

path = os.path.dirname(__file__)

# Define models and their file paths
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

# Color maps per model
colormaps = {
    "ConvNeXtSmall": plt.cm.viridis,
    "ConvNeXtTiny": plt.cm.plasma,
}

# Normalize based on number of datapoints
datapoint_keys = list(models["ConvNeXtSmall"].keys())
datapoint_values = np.array([int(k) for k in datapoint_keys])
norm = plt.Normalize(vmin=datapoint_values.min(), vmax=datapoint_values.max())

# Store all model info
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

# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test'),
#     Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.5, label='Train'),
#     Line2D([0], [0], color=colormaps["ConvNeXtSmall"](0.8), lw=2, label="ConvNeXtSmall"),
#     Line2D([0], [0], color=colormaps["ConvNeXtTiny"](0.8), lw=2, label="ConvNeXtTiny"),
# ]

# Plot R^2:
plt.figure(figsize=(6.4, 6.4))

for i, (model_name, runs) in enumerate(model_data.items()):
    ax = plt.subplot(1, 2, i + 1)
    legend_lines = []

    for run in runs:
        df = run["df"]
        color = run["color"]
        datapoints = run["datapoints"]

        # Plot lines
        ax.plot(df["epoch"], df["test_r2"], c=color, linestyle="-")
        ax.plot(df["epoch"], df["train_r2"], c=color, linestyle="--", alpha=0.5)

        # Store legend handles for datapoints
        legend_lines.append(Line2D([0], [0], color=color, lw=2, label=f"{datapoints}"))

        # Print stats
        idx_max = np.argmax(df['test_r2'])
        print(f"model: {model_name}, datapoints: {datapoints}, "
              f"test R²: {df['test_r2'][idx_max]:.5f}, train R²: {df['train_r2'][idx_max]:.5f}, "
              f"test MSE: {df['test_mse'][idx_max]:.6f}, train MSE: {df['train_mse'][idx_max]:.6f}")

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




# Plot regression:
# Initialize list for collecting all rows
data_records = []

# Collect all data points into a list of dicts
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

# Create a DataFrame
df_points = pd.DataFrame(data_records)

# Optional: Check structure
print(df_points.head())

# Plot using seaborn for hue support
import seaborn as sns
plt.figure(figsize=(6.4, 6.4))
sns.scatterplot(
    data=df_points,
    x="num_datapoints",
    y="max_r2",
    hue="model"
    # palette={row["model"]: row["color"] for row in data_records}
)

# Optional: add fits per model
# for model in df_points["model"].unique():
#     df_model = df_points[df_points["model"] == model]
#     x = np.array(df_model["num_datapoints"])
#     y = np.array(df_model["max_r2"])
    
#     # Fit log-log regression
#     log_x = np.log(x)
#     log_y = np.log(y)
#     slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

#     # Prediction
#     x_fit = np.linspace(x.min(), x.max(), 200)
#     y_fit = np.exp(intercept + slope * np.log(x_fit))

#     plt.plot(x_fit, y_fit, linestyle="--", label=f"{model} Fit (R²={r_value**2:.3f})", color=df_model["color"].iloc[0])

# Labels & save
plt.ylabel(r"$R^2$ Score")
plt.xlabel("Num Datapoints")
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(path, "diff_num_datapoints_fit.pdf"), bbox_inches='tight')