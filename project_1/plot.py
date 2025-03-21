'''Module for plotting with CNN'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import os
path = os.path.dirname(__file__)

from cnn import CNN
from feedforward import FeedForward

def plot_params():
    '''
    Function for showing the distribution of permeability's calculated from the simulations.
    '''
    params = torch.load('project_1/data/k.pt',weights_only=False).numpy()
    sns.lineplot(np.sort(params,kind='heapsort'))
    plt.xlabel("Index")
    plt.ylabel("Permeability")
    plt.savefig(os.path.join(path,'plots/perms.pdf'))
    # plt.show()

def plot_best_r2_grid_search_cnn(data):
    '''
    Plotting training R2 and MSE for best CNN after grid search.
    '''
    # Get the index of the best model:
    best_model_idx = data["Test R2"].idxmax()

    # Extract hyperparameters of the best model:
    best_model_params = data.loc[best_model_idx, ["Learning Rate", "L2 Weight Decay", "CNNs", "Hiddens", "Activation"]]

    # Filter all epochs of the best model:
    best_model_df = data[
        (data["Learning Rate"] == best_model_params["Learning Rate"]) &
        (data["L2 Weight Decay"] == best_model_params["L2 Weight Decay"]) &
        (data["CNNs"] == best_model_params["CNNs"]) &
        (data["Hiddens"] == best_model_params["Hiddens"]) &
        (data["Activation"] == best_model_params["Activation"])
    ]

    # Plot metrics:
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
    '''
    Function for plotting the variations of convolutional layers and hidden layers 
    for the best CNN after grid search.
    '''
    # Get the index of the best model:
    best_model_idx = data["Test R2"].idxmax()
    best_model_params = data.loc[best_model_idx, ["Learning Rate", "L2 Weight Decay", "Activation"]]
    best_model_df = data[
        (data["Learning Rate"] == best_model_params["Learning Rate"]) &
        (data["L2 Weight Decay"] == best_model_params["L2 Weight Decay"]) &
        (data["Activation"] == best_model_params["Activation"])
    ]

    # Define color palette:
    palette = sns.color_palette("mako_r", best_model_df["CNNs"].nunique())
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    # Subplot 1: Test R2:
    sns.lineplot(data=best_model_df, x="Epoch", y="Test R2", hue="CNNs",style="Hiddens", palette=palette, ax=axes[0])
    
    # Subplot 2: Train R2:
    sns.lineplot(data=best_model_df, x="Epoch", y="Train R2", hue="CNNs", style="Hiddens", palette=palette, ax=axes[1])
    plt.tight_layout()

    plt.savefig(os.path.join(path,'plots/diff_conv_hid_cnn_gird_search_r2.pdf'))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

    # Subplot 1: Test MSE:
    sns.lineplot(data=best_model_df, x="Epoch", y="Test MSE", hue="CNNs", style="Hiddens", palette=palette, ax=axes[0])
    axes[1].set_yscale("log")  # Log scale for better readability

    # Subplot 2: Train MSE:
    sns.lineplot(data=best_model_df, x="Epoch", y="Train MSE", hue="CNNs", style="Hiddens", palette=palette, ax=axes[1])
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(path,'plots/diff_conv_hid_cnn_gird_search_mse.pdf'))


def plot_best_r2_grid_search_ff(data):
    '''
    Plotting training R2 and MSE for best CNN after grid search.
    '''
    # Get the index of the best model:
    best_model_idx = data["Test R2"].idxmax()
    
    # Extract hyperparameters of the best model:
    best_model_params = data.loc[best_model_idx, ["Learning Rate", "L2 Weight Decay", "Hiddens", "Activation"]]

    # Filter all epochs of the best model:
    best_model_df = data[
        (data["Learning Rate"] == best_model_params["Learning Rate"]) &
        (data["L2 Weight Decay"] == best_model_params["L2 Weight Decay"]) &
        (data["Hiddens"] == best_model_params["Hiddens"]) &
        (data["Activation"] == best_model_params["Activation"])
    ]

    # Plot metrics:
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
    '''
    Plotting training R2 and MSE for best CNN  with varying hidden size after grid search.
    '''
    # Get the index of the best model:
    best_model_idx = data["Test R2"].idxmax()
    best_model_params = data.loc[best_model_idx, ["Learning Rate", "L2 Weight Decay", "Activation"]]
    best_model_df = data[
        (data["Learning Rate"] == best_model_params["Learning Rate"]) &
        (data["L2 Weight Decay"] == best_model_params["L2 Weight Decay"]) &
        (data["Activation"] == best_model_params["Activation"])
    ]

    # Define color palette:
    palette = sns.color_palette("mako_r", best_model_df["Hiddens"].nunique())
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    # Subplot 1: Test R2:
    sns.lineplot(data=best_model_df, x="Epoch", y="Test R2", hue="Hiddens", palette=palette, ax=axes[0])
    
    # Subplot 2: Train R2: 
    sns.lineplot(data=best_model_df, x="Epoch", y="Train R2", hue="Hiddens", palette=palette, ax=axes[1])
    plt.tight_layout()


    plt.savefig(os.path.join(path,'plots/diff_hid_ff_gird_search_r2.pdf'))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

    # Subplot 1: Test MSE:
    sns.lineplot(data=best_model_df, x="Epoch", y="Test MSE", hue="Hiddens", palette=palette, ax=axes[0])
    axes[1].set_yscale("log")  # Log scale for better readability

    # Subplot 2: Train MSE:
    sns.lineplot(data=best_model_df, x="Epoch", y="Train MSE", hue="Hiddens", palette=palette, ax=axes[1])
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(path,'plots/diff_hid_ff_gird_search_mse.pdf'))

def prepare_long_format(df, model_name):
    """
    Reshapes the dataframe for better plotting plotting.
    """
    df = df.melt(id_vars=["Epoch"], 
                 value_vars=["Train R2", "Test R2", "Train MSE", "Test MSE"], 
                 var_name="Metric", 
                 value_name="Value")
    df["Model"] = model_name
    df["Set"] = df["Metric"].apply(lambda x: "Train" if "Train" in x else "Test")
    df["Metric"] = df["Metric"].str.replace("Train ", "").str.replace("Test ", "")  # Remove redundant text
    return df
def plot_long(cnn, ff):
    '''
    Plot R2 and MSE after longer training:
    '''
    # Put both model in same dataframe:
    cnn_long = prepare_long_format(cnn, "CNN")
    ff_long = prepare_long_format(ff, "FF")
    df_long = pd.concat([cnn_long, ff_long])

    # Plot metrics:
    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_long[df_long["Metric"] == "R2"], x="Epoch", y="Value", hue="Model", style="Set")
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_long[df_long["Metric"] == "MSE"], x="Epoch", y="Value", hue="Model", style="Set")
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'plots/longtrain.pdf'))

def plot_plot_pred():
    '''
    Plot predictions of permeability's form both models together with true values:
    '''
    # Reconstruct CNN:
    cnns = [{'out_channels': 10, 'kernel_size': 5, 'stride': 2, 'pool': 'max'},
         {'out_channels': 20, 'kernel_size': 7, 'stride': 1, 'pool': 'max'},
         {'out_channels': 40, 'kernel_size': 5, 'stride': 1, 'pool': 'max'}]
    cnn = CNN(conv_layers_params=cnns,
            hidden_sizes=[2,4],
            activation='leakyrelu',
            use_batch_norm=True,
            use_dropout=True)  # Ensure same architecture
    cnn.load_state_dict(torch.load(os.path.join(path, 'models/cnn.pth')))
    cnn.eval()
    
    # Reconstruct FFNN:
    ff = FeedForward(hidden_sizes=[32])
    ff.load_state_dict(torch.load(os.path.join(path,'models/ff.pth')), strict=False)
    ff.eval()
    
    # Load and standardize data:
    images = torch.load(os.path.join(path,'data/images.pt'),weights_only=False) # Load generated images
    params = torch.load(os.path.join(path,'data/k.pt'),weights_only=False) # Load calculated permeability
    params_mean, params_std = params.mean(), params.std()
    params = (params-params_mean)/params_std

    # Sorting data for viability:
    sorted_indices = torch.argsort(params)
    sorted_params = params[sorted_indices]
    sorted_images = images[sorted_indices]

    # Process in batches:
    pred_cnn, pred_ff = [], []
    batch_size = 512
    with torch.no_grad():  # No gradients needed for inference
        for i in range(0, len(sorted_images), batch_size):
            batch_images = sorted_images[i:i + batch_size]
            pred_cnn.append(cnn(batch_images).cpu())
            pred_ff.append(ff(batch_images).cpu())

    # Concatenate batch results:
    pred_cnn = torch.cat(pred_cnn).numpy()
    pred_ff = torch.cat(pred_ff).numpy()

    # Plot results:
    x = np.arange(len(pred_cnn))
    plt.figure(figsize=(4,4))
    plt.plot(x,pred_cnn,'.', label="CNN")
    plt.plot(x,pred_ff,'.', label="FFNN")
    plt.plot(x,sorted_params.numpy(),"-",c='r', label="True")
    plt.xlabel('Image index')
    plt.ylabel('Standardized permeability')
    plt.legend()
    plt.savefig(os.path.join(path, 'plots/preds.pdf'))




if __name__ == '__main__':
    df_cnn_gird_search = pd.read_csv(os.path.join(path,'training_data/cnn_grid_search_full.csv'))
    plot_best_r2_grid_search_cnn(df_cnn_gird_search)
    plot_best_r2_grid_search_cnn_vary_conv_hid(df_cnn_gird_search)

    df_ff_gird_search = pd.read_csv(os.path.join(path,'training_data/ff_grid_search_full.csv'))
    plot_best_r2_grid_search_ff(df_ff_gird_search)
    plot_best_r2_grid_search_ff_vary_hid(df_ff_gird_search)

    df_cnn_long = pd.read_csv(os.path.join(path,'training_data/cnn_best_long_01.csv'))
    df_ff_long = pd.read_csv(os.path.join(path,'training_data/ff_best_long_01.csv'))
    plot_long(df_cnn_long,df_ff_long)
    plot_plot_pred()

