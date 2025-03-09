'''Module for plotting with CNN'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
path = os.path.dirname(__file__)

class Plotter:


    '''
    Class for plotting the metrics of a neural network training process and show its architecture.
    '''
    def __init__(self,trainer = None, model = None):
        self.trainer = trainer
        self.model = model

    def plot_mse(self,name):
        '''
        Plotting the mean square error of the training process.
        '''
        train_mse = self.trainer.train_mse
        test_mse = self.trainer.test_mse
        fig = plt.figure(figsize=(8,6))
        plt.plot(train_mse, label= 'Train MSE')
        plt.plot(test_mse, label= 'Test MSE')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(os.path.join(path,f'plots/{name}/{name}_mse.pdf'))

    def plot_r2(self,name):
        '''
        Plotting the R2 score of the training process.
        '''
        train_r2 = self.trainer.train_r2
        test_r2 = self.trainer.test_r2
        fig = plt.figure(figsize=(8,6))
        plt.plot(train_r2, label= 'Train R2')
        plt.plot(test_r2, label= 'Test R2')
        plt.xscale('log')
        # plt.yscale('log')
        plt.ylim(-1,1)
        plt.xlabel('epoch')
        plt.ylabel('R2')
        plt.legend()
        plt.savefig(os.path.join(path,f'plots/{name}/{name}_r2.pdf'))

    def visualize_kernels_1(self,name):
        '''
        Visualizing the kernels from the first convolution layer
        '''
        weight = self.model.convolution_1.weight.data.numpy()
        out_channels= weight.shape[0]
        weight = weight[:, 0]
        vmin, vmax = weight.min(), weight.max()
        
        fig, axes = plt.subplots(4, 8, figsize=(8, 4))
        for i, ax in enumerate(axes.flat):
            if i < out_channels:  # Avoid extra subplots being used
                img = ax.imshow(weight[i], cmap="jet", vmin=vmin, vmax=vmax)
                ax.axis("off")
        fig.colorbar(img, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
        plt.savefig(os.path.join(path,f'plots/{name}/{name}_kernels_1.pdf'))

    def visualize_kernels_2(self,name):
        '''
        Visualizing the kernels from the second convolution layer
        '''
        weight = self.model.convolution_2.weight.data.numpy()
        out_channels= weight.shape[0]
        weight = weight[:, 0]
        vmin, vmax = weight.min(), weight.max() 

        fig, axes = plt.subplots(6, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < out_channels:  # Avoid extra subplots being used
                img = ax.imshow(weight[i], cmap="jet", vmin=vmin, vmax=vmax)
                ax.axis("off")
        fig.colorbar(img, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
        plt.savefig(os.path.join(path,f'plots/{name}/{name}_kernels_2.pdf'))

def plot_metrics(df_epoch_results):
    for model_name in ["CNN", "Autoencoder"]:
        df_model = df_epoch_results[df_epoch_results["Model"] == model_name].copy()
        
        # Convert Learning Rate to string to avoid hue issues
        df_model["Learning Rate"] = df_model["Learning Rate"].astype(str)

        ## ---- MSE Plot ---- ##
        g = sns.FacetGrid(df_model, col="L2 Weight Decay", col_wrap=2, height=3, sharey=False)
        g.map_dataframe(sns.lineplot, x="Epoch", y="Train MSE", hue="Learning Rate", marker="o", linestyle="-")
        g.map_dataframe(sns.lineplot, x="Epoch", y="Test MSE", hue="Learning Rate", marker="s", linestyle="--")
        g.set_axis_labels("Epoch", "MSE")
        g.set_titles(f"{model_name} - L2 = {{col_name}}")
        g.add_legend()
        plt.savefig(os.path.join(path, f"plots/training_metrics_{model_name.lower()}_mse.pdf"))
        plt.clf()

        ## ---- R² Plot ---- ##
        g = sns.FacetGrid(df_model, col="L2 Weight Decay", col_wrap=2, height=3, sharey=False)
        g.map_dataframe(sns.lineplot, x="Epoch", y="Train R2", hue="Learning Rate", marker="o", linestyle="-")
        g.map_dataframe(sns.lineplot, x="Epoch", y="Test R2", hue="Learning Rate", marker="s", linestyle="--")
        g.set_axis_labels("Epoch", "R²")
        g.set_titles(f"{model_name} - L2 = {{col_name}}")
        g.add_legend()
        plt.savefig(os.path.join(path, f"plots/training_metrics_{model_name.lower()}_r2.pdf"))
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
