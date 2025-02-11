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
        fig = plt.figure(figsize=(8,6))
        plt.plot(train_mse, label= 'Train MSE')
        plt.plot(test_mse, label= 'Test MSE')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.legend()

    def plot_r2(self):
        '''
        Plotting the R2 score of the training process.
        '''
        train_r2 = self.trainer.train_r2
        test_r2 = self.trainer.test_r2
        fig = plt.figure(figsize=(8,6))
        plt.plot(train_r2, label= 'Train R2')
        plt.plot(test_r2, label= 'Test R2')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('R2')
        plt.legend()

    def visualize_kernels_1(self):
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

    def visualize_kernels_2(self):
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