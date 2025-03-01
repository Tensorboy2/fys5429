'''Main module for project 1.'''
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
path = os.path.dirname(__file__)

from cnn import CNN
from train import Trainer
from plot import Plotter
from autoencoder import Autoencoder


def grid_search():
    batch_size = 32
    lrs = [0.01,0.005,0.001,0.0005,0.0001]
    l2s = [0,0.01,0.05,0.1,0.5]
    results = []

    images = torch.load(os.path.join(path,'data/images.pt')) # Load generated data
    params = torch.load(os.path.join(path,'data/k.pt')) # Load generated data

    # Prepare data for the 
    X_train, X_test, y_train, y_test = train_test_split(images, params, test_size=0.2)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)


    for lr in lrs:
        for l2 in l2s:
            model = CNN(kernel_size_1=8,
                    stride_1=4,
                    kernel_size_2=8,
                    stride_2=4,
                    pool_1='max',
                    pool_2='max',
                    activation='sigmoid',
                    use_batch_norm=True,
                    use_dropout=True)
            optimizer = optim.Adam(params = model.parameters(), lr = lr,weight_decay=l2)

            trainer = Trainer()
            trainer.train(model,optimizer,train_data_loader,test_data_loader, num_epochs=200)
            
            train_mse = trainer.train_mse[-1]
            test_mse = trainer.train_mse[-1]
            train_r2 = trainer.train_r2[-1]
            test_r2 = trainer.train_r2[-1]
            results.append((lr, l2, train_r2, test_r2, train_mse, train_mse))
    df = pd.DataFrame(results, columns=["lr", "l2", "train_r2", "test_r2", "trains_mse", "test_mse"])
    df.to_csv(os.path.join(path,'r2.csv'))
    # Convert to DataFrame
    data = pd.read_csv(os.path.join(path,'r2.csv'))
    # Pivot for heatmap
    train_r2_pivot = data.pivot(index="lr", columns="l2", values="train_r2")
    test_r2_pivot = data.pivot(index="lr", columns="l2", values="test_r2")
    train_mse_pivot = data.pivot(index="lr", columns="l2", values="train_mse")
    test_mse_pivot = data.pivot(index="lr", columns="l2", values="test_mse")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(train_r2_pivot, annot=True, cmap="viridis", ax=axes[0])
    sns.heatmap(test_r2_pivot, annot=True, cmap="magma", ax=axes[1])
    plt.savefig(os.path.join(path,'plots/grid_search_r2.pdf'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(train_mse_pivot, annot=True, cmap="viridis", ax=axes[0])
    sns.heatmap(test_mse_pivot, annot=True, cmap="magma", ax=axes[1])
    plt.savefig(os.path.join(path,'plots/grid_search_mse.pdf'))
    # plt.show()

def main_cnn():
    '''
    Main function of project.
    '''
    # Initializing classes
    model = CNN(kernel_size_1=8,
                stride_1=2,
                kernel_size_2=8,
                stride_2=2,
                pool_1='max',
                pool_2='max',
                hidden_size=4096,
                # hidden_size=2048,
                use_batch_norm=True,
                use_dropout=True)
    model.p=0.5
    optimizer = optim.Adam(params = model.parameters(), lr = 0.001,weight_decay=0.1)
    trainer = Trainer()
    batch_size = 16

    # Dummy data
    # num_images = 10_000
    # images = torch.rand((num_images,1,128,128))
    # num_images = images.shape[0] # Get num images
    # params = torch.rand((num_images)) # Produce fake labels
    images = torch.load(os.path.join(path,'data/images.pt')) # Load generated data
    params = torch.load(os.path.join(path,'data/k.pt')) # Load generated data
    params = (params - params.min()) / (params.max() - params.min())
    # Prepare data for the 
    X_train, X_test, y_train, y_test = train_test_split(images, params, test_size=0.3)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)

    start = time.time()
    trainer.train(model,optimizer,train_data_loader,test_data_loader, num_epochs=200) # Run training
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')

    torch.save(model.state_dict(), os.path.join(path,'models/model.pth'))

    plotter = Plotter(trainer,model)
    plotter.plot_mse()
    plotter.plot_r2()
    plotter.visualize_kernels_1()
    plotter.visualize_kernels_2()
    # plt.show()


def main_autoencoder():
    '''
    Main function of project.
    '''
    # Initializing classes
    model = Autoencoder()
    optimizer = optim.Adam(params = model.parameters(), lr = 0.01,weight_decay=0.1)
    trainer = Trainer()
    batch_size = 16

    # Dummy data
    # num_images = 10_000
    # images = torch.rand((num_images,1,128,128))
    # num_images = images.shape[0] # Get num images
    # params = torch.rand((num_images)) # Produce fake labels
    images = torch.load(os.path.join(path,'data/images.pt')) # Load generated data
    params = torch.load(os.path.join(path,'data/k.pt')) # Load generated data
    params = (params - params.min()) / (params.max() - params.min())
    # Prepare data for the 
    X_train, X_test, y_train, y_test = train_test_split(images, params, test_size=0.3)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)

    start = time.time()
    trainer.train(model,optimizer,train_data_loader,test_data_loader, num_epochs=200) # Run training
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')

    torch.save(model.state_dict(), os.path.join(path,'models/model.pth'))

    plotter = Plotter(trainer,model)
    plotter.plot_mse()
    plotter.plot_r2()
    plotter.visualize_kernels_1()
    plotter.visualize_kernels_2()
    # plt.show()




if __name__ == '__main__':
    # grid_search()
    main_autoencoder()
    