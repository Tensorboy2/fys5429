'''Module for plotting with CNN'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import os
path = os.path.dirname(__file__)


def plot_params():
    params = torch.load('project_1/data/k.pt',weights_only=False).numpy()
    sns.lineplot(np.sort(params,kind='heapsort'))
    plt.xlabel("Index")
    plt.ylabel("Permeability")
    plt.savefig(os.path.join(path,'plots/perms.pdf'))
    # plt.show()

def plot_best_r2_grid_search_cnn(data):
    # Get the index of the best model
    best_model_idx = data["Test R2"].idxmax()

    # Extract hyperparameters of the best model
    best_model_params = data.loc[best_model_idx, ["Learning Rate", "L2 Weight Decay", "CNNs", "Hiddens", "Activation"]]

    # Filter all epochs of the best model
    best_model_df = data[
        (data["Learning Rate"] == best_model_params["Learning Rate"]) &
        (data["L2 Weight Decay"] == best_model_params["L2 Weight Decay"]) &
        (data["CNNs"] == best_model_params["CNNs"]) &
        (data["Hiddens"] == best_model_params["Hiddens"]) &
        (data["Activation"] == best_model_params["Activation"])
    ]
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    sns.lineplot(best_model_df, x="Epoch",y="Test R2",label="Test R2")
    sns.lineplot(best_model_df, x="Epoch",y="Train R2",linestyle="--",label="Train R2")
    plt.ylabel('R2')
    plt.subplot(1,2,2)
    sns.lineplot(best_model_df, x="Epoch",y="Test MSE",label="Test MSE")
    sns.lineplot(best_model_df, x="Epoch",y="Train MSE",linestyle="--",label="Train MSE")
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig(os.path.join(path,'plots/best_cnn_grid_search_r2.pdf'))

def plot_best_r2_grid_search_cnn_vary_conv_hid(data):
    best_model_idx = data["Test R2"].idxmax()
    best_model_params = data.loc[best_model_idx, ["Learning Rate", "L2 Weight Decay", "Activation"]]
    best_model_df = data[
        (data["Learning Rate"] == best_model_params["Learning Rate"]) &
        (data["L2 Weight Decay"] == best_model_params["L2 Weight Decay"]) &
        (data["Activation"] == best_model_params["Activation"])
    ]
    # Define color palette
    palette = sns.color_palette("mako_r", best_model_df["CNNs"].nunique())
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    # Subplot 1: Test R² 
    sns.lineplot(data=best_model_df, x="Epoch", y="Test R2", hue="CNNs",style="Hiddens", palette=palette, ax=axes[0])
    
    # Subplot 2: Test R2 
    sns.lineplot(data=best_model_df, x="Epoch", y="Train R2", hue="CNNs", style="Hiddens", palette=palette, ax=axes[1])
    plt.tight_layout()


    plt.savefig(os.path.join(path,'plots/diff_conv_hid_cnn_gird_search_r2.pdf'))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

    # Subplot 1: Test MSE
    sns.lineplot(data=best_model_df, x="Epoch", y="Test MSE", hue="CNNs", style="Hiddens", palette=palette, ax=axes[0])
    axes[1].set_yscale("log")  # Log scale for better readability

    # Subplot 2: Train MSE
    sns.lineplot(data=best_model_df, x="Epoch", y="Train MSE", hue="CNNs", style="Hiddens", palette=palette, ax=axes[1])
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(path,'plots/diff_conv_hid_cnn_gird_search_mse.pdf'))

def plot_best_r2_grid_search_ff(data):
    # Get the index of the best model
    best_model_idx = data["Test R2"].idxmax()
    best_model = data.loc[data["Test R2"].idxmax()]
    print(best_model)
    # Extract hyperparameters of the best model
    best_model_params = data.loc[best_model_idx, ["Learning Rate", "L2 Weight Decay", "Hiddens", "Activation"]]
    print
    # Filter all epochs of the best model
    best_model_df = data[
        (data["Learning Rate"] == best_model_params["Learning Rate"]) &
        (data["L2 Weight Decay"] == best_model_params["L2 Weight Decay"]) &
        (data["Hiddens"] == best_model_params["Hiddens"]) &
        (data["Activation"] == best_model_params["Activation"])
    ]
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    sns.lineplot(best_model_df, x="Epoch",y="Test R2",label="Test R2")
    sns.lineplot(best_model_df, x="Epoch",y="Train R2",linestyle="--",label="Train R2")
    plt.ylabel('R2')
    plt.subplot(1,2,2)
    sns.lineplot(best_model_df, x="Epoch",y="Test MSE",label="Test MSE")
    sns.lineplot(best_model_df, x="Epoch",y="Train MSE",linestyle="--",label="Train MSE")
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig(os.path.join(path,'plots/best_ff_grid_search_r2.pdf'))

def plot_best_r2_grid_search_ff_vary_hid(data):
    best_model_idx = data["Test R2"].idxmax()
    best_model_params = data.loc[best_model_idx, ["Learning Rate", "L2 Weight Decay", "Activation"]]
    best_model_df = data[
        (data["Learning Rate"] == best_model_params["Learning Rate"]) &
        (data["L2 Weight Decay"] == best_model_params["L2 Weight Decay"]) &
        (data["Activation"] == best_model_params["Activation"])
    ]
    # Define color palette
    palette = sns.color_palette("mako_r", best_model_df["Hiddens"].nunique())
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    # Subplot 1: Test R² 
    sns.lineplot(data=best_model_df, x="Epoch", y="Test R2", hue="Hiddens", palette=palette, ax=axes[0])
    
    # Subplot 2: Test R2 
    sns.lineplot(data=best_model_df, x="Epoch", y="Train R2", hue="Hiddens", palette=palette, ax=axes[1])
    plt.tight_layout()


    plt.savefig(os.path.join(path,'plots/diff_hid_ff_gird_search_r2.pdf'))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

    # Subplot 1: Test MSE
    sns.lineplot(data=best_model_df, x="Epoch", y="Test MSE", hue="Hiddens", palette=palette, ax=axes[0])
    axes[1].set_yscale("log")  # Log scale for better readability

    # Subplot 2: Train MSE
    sns.lineplot(data=best_model_df, x="Epoch", y="Train MSE", hue="Hiddens", palette=palette, ax=axes[1])
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(path,'plots/diff_hid_ff_gird_search_mse.pdf'))

if __name__ == '__main__':
    df_cnn_gird_search = pd.read_csv(os.path.join(path,'training_data/cnn_grid_search_full.csv'))
    plot_best_r2_grid_search_cnn(df_cnn_gird_search)
    # plot_best_r2_grid_search_cnn_vary_conv_hid(df_cnn_gird_search)

    df_ff_gird_search = pd.read_csv(os.path.join(path,'training_data/ff_grid_search_full.csv'))
    # plot_best_r2_grid_search_ff(df_ff_gird_search)
    plot_best_r2_grid_search_ff_vary_hid(df_ff_gird_search)

    


