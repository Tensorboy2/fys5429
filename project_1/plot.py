'''Module for plotting with CNN'''
import matplotlib.pyplot as plt

class Plotter:
    '''
    Class for plotting the metrics of a neural network training process and show its architecture.
    '''
    def __init__(self,trainer = None, model = None):
        self.trainer = trainer
        self.model = model

    def plot_mse(self):
        '''
        Plotting the mean square error of the training process.
        '''
        train_mse = self.trainer.train_mse
        test_mse = self.trainer.test_mse
        plt.plot(train_mse, label= 'Train MSE')
        plt.plot(test_mse, label= 'Test MSE')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def visualize_kernels(self):
        weight = self.model.convolution_1.weight.data.numpy()  # Move to CPU & convert to NumPy
        out_channels, in_channels, H, W = weight.shape  # (C_out, C_in, H, W)
        weight = weight[:, 0]  # Shape: (out_channels, H, W)

        # Compute grid size for the closest rectangular arrangement
        fig, axes = plt.subplots(4, 8, figsize=(8, 4))
        
        vmin, vmax = weight.min(), weight.max() # Find global min/max for normalization
        for i, ax in enumerate(axes.flat):
            if i < out_channels:  # Avoid extra subplots being used
                img = ax.imshow(weight[i], cmap="jet", vmin=vmin, vmax=vmax)
                ax.axis("off")
        fig.colorbar(img, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
        plt.show()