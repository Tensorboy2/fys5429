'''Module for plotting with CNN'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
path = os.path.dirname(__file__)


def plot_metrics(df_epoch_results):
    for model_name in ["CNN", "Autoencoder"]:
        df_model = df_epoch_results[df_epoch_results["Model"] == model_name].copy()
        
        # Convert Learning Rate to string to avoid hue issues
        df_model["Learning Rate"] = df_model["Learning Rate"].astype(str)

        ## ---- Train MSE Plot ---- ##
        g = sns.FacetGrid(df_model, col="L2 Weight Decay", col_wrap=2, height=3, sharey=False)
        g.map_dataframe(sns.lineplot, x="Epoch", y="Train MSE", hue="Learning Rate")#, marker="o", linestyle="-")
        g.set_axis_labels("Epoch", "Train MSE (log scale)")
        g.set_titles(f"{model_name} - L2 = {{col_name}}")
        g.set(yscale="log")  # Set y-axis to logarithmic scale
        g.add_legend()
        plt.savefig(os.path.join(path, f"plots/training_metrics_{model_name.lower()}_train_mse.pdf"))
        plt.clf()

        ## ---- Test MSE Plot ---- ##
        g = sns.FacetGrid(df_model, col="L2 Weight Decay", col_wrap=2, height=3, sharey=False)
        g.map_dataframe(sns.lineplot, x="Epoch", y="Test MSE", hue="Learning Rate")#, marker="s", linestyle="--")
        g.set_axis_labels("Epoch", "Test MSE (log scale)")
        g.set_titles(f"{model_name} - L2 = {{col_name}}")
        g.set(yscale="log")  # Set y-axis to logarithmic scale
        g.add_legend()
        plt.savefig(os.path.join(path, f"plots/training_metrics_{model_name.lower()}_test_mse.pdf"))
        plt.clf()

        ## ---- Train MAE Plot ---- ##
        g = sns.FacetGrid(df_model, col="L2 Weight Decay", col_wrap=2, height=3, sharey=False)
        g.map_dataframe(sns.lineplot, x="Epoch", y="Train MAE", hue="Learning Rate")#, marker="o", linestyle="-")
        g.set_axis_labels("Epoch", "Train MAE (log scale)")
        g.set_titles(f"{model_name} - L2 = {{col_name}}")
        g.set(yscale="log")  # Set y-axis to logarithmic scale
        g.add_legend()
        plt.savefig(os.path.join(path, f"plots/training_metrics_{model_name.lower()}_train_mae.pdf"))
        plt.clf()

        ## ---- Test MAE Plot ---- ##
        g = sns.FacetGrid(df_model, col="L2 Weight Decay", col_wrap=2, height=3, sharey=False)
        g.map_dataframe(sns.lineplot, x="Epoch", y="Test MAE", hue="Learning Rate")#, marker="s", linestyle="--")
        g.set_axis_labels("Epoch", "Test MAE (log scale)")
        g.set_titles(f"{model_name} - L2 = {{col_name}}")
        g.set(yscale="log")  # Set y-axis to logarithmic scale
        g.add_legend()
        plt.savefig(os.path.join(path, f"plots/training_metrics_{model_name.lower()}_test_mae.pdf"))
        plt.clf()

        ## ---- Train R² Plot ---- ##
        g = sns.FacetGrid(df_model, col="L2 Weight Decay", col_wrap=2, height=3, sharey=False)
        g.map_dataframe(sns.lineplot, x="Epoch", y="Train R2", hue="Learning Rate")#, marker="o", linestyle="-")
        g.set_axis_labels("Epoch", "Train R²")
        g.set_titles(f"{model_name} - L2 = {{col_name}}")
        g.add_legend()
        plt.savefig(os.path.join(path, f"plots/training_metrics_{model_name.lower()}_train_r2.pdf"))
        plt.clf()

        ## ---- Test R² Plot ---- ##
        g = sns.FacetGrid(df_model, col="L2 Weight Decay", col_wrap=2, height=3, sharey=False)
        g.map_dataframe(sns.lineplot, x="Epoch", y="Test R2", hue="Learning Rate")#, marker="s", linestyle="--")
        g.set_axis_labels("Epoch", "Test R²")
        g.set_titles(f"{model_name} - L2 = {{col_name}}")
        g.add_legend()
        plt.savefig(os.path.join(path, f"plots/training_metrics_{model_name.lower()}_test_r2.pdf"))
        plt.clf()


def plot_heat(df_results):
    model_name='CNN'
    df_model = df_results[df_results["Model"] == model_name]
    # df_pivot = df_model.pivot("Learning Rate", "L2 Weight Decay", "Final Test MSE")
    df_pivot = df_model.pivot(index="Learning Rate", columns="L2 Weight Decay", values="Final Test MSE")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pivot, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title(f"Final Test MSE for {model_name}")
    plt.xlabel("L2 Weight Decay")
    plt.ylabel("Learning Rate")
    plt.savefig(os.path.join(path,'plots/CNN_grid_search_mse.pdf'))

    df_pivot = df_model.pivot(index="Learning Rate", columns="L2 Weight Decay", values="Final Test R2")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pivot, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title(f"Final Test R2 for {model_name}")
    plt.xlabel("L2 Weight Decay")
    plt.ylabel("Learning Rate")
    plt.savefig(os.path.join(path,'plots/CNN_grid_search_r2.pdf'))



    model_name='Autoencoder'
    df_model = df_results[df_results["Model"] == model_name]
    # df_pivot = df_model.pivot("Learning Rate", "L2 Weight Decay", "Final Test MSE")
    df_pivot = df_model.pivot(index="Learning Rate", columns="L2 Weight Decay", values="Final Test MSE")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pivot, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title(f"Final Test MSE for {model_name}")
    plt.xlabel("L2 Weight Decay")
    plt.ylabel("Learning Rate")
    plt.savefig(os.path.join(path,'plots/Autoencoder_grid_search_mse.pdf'))

    df_pivot = df_model.pivot(index="Learning Rate", columns="L2 Weight Decay", values="Final Test R2")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pivot, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title(f"Final Test R2 for {model_name}")
    plt.xlabel("L2 Weight Decay")
    plt.ylabel("Learning Rate")
    plt.savefig(os.path.join(path,'plots/Autoencoder_grid_search_r2.pdf'))

if __name__ == '__main__':
    df_results = pd.read_csv(os.path.join(path,'training_data/grid_search_last.csv'))
    df_epoch_results = pd.read_csv(os.path.join(path,'training_data/grid_search_full.csv'))
    plot_metrics(df_epoch_results)
    plot_heat(df_results)
