'''Main module for project 1.'''
import torch
torch.manual_seed(0)
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
import os
path = os.path.dirname(__file__)

from cnn import CNN
from train import Trainer,train
from autoencoder import Autoencoder

def get_data(batch_size = 32,test_size=0.2,normalize=True):
    '''
    Function for getting data and turning them into the train and test loader.
    '''
    images = torch.load(os.path.join(path,'data/images.pt'),weights_only=False) # Load generated images
    params = torch.load(os.path.join(path,'data/k.pt'),weights_only=False) # Load calculated permeability
    if normalize:
        params_mean, params_std = params.mean(), params.std()
        params = (params-params_mean)/params_std
        # params = (params - params.min()) / (params.max() - params.min())
    # Prepare data for the 
    X_train, X_test, y_train, y_test = train_test_split(images, params, test_size=test_size,random_state=42)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)
    return train_data_loader, test_data_loader

def grid_search():
    '''
    Grid search function. 
    Preforms a grid search of given model hyper parameters and model architectures.
    '''
    results = []
    epoch_results = []
    train_data_loader, test_data_loader = get_data(batch_size=32,test_size=0.10)




    # Different convolution structures:
    cnns = [[{'out_channels': 5, 'kernel_size': 4, 'stride': 1, 'pool': 'max'},
             {'out_channels': 10, 'kernel_size': 7, 'stride': 1, 'pool': 'max'},
             {'out_channels': 20, 'kernel_size': 11, 'stride': 1, 'pool': 'max'}],
            [{'out_channels': 5, 'kernel_size': 3, 'stride': 2, 'pool': 'max'},
             {'out_channels': 10, 'kernel_size': 3, 'stride': 1, 'pool': 'max'}],
            [{'out_channels': 5, 'kernel_size': 3, 'stride': 2, 'pool': 'max'}]]
    
    hidden_sizes = [[2,4,8],
                    [8,16]]
    
    
    lrs = np.logspace(-2,-4,6)#[0.1,0.01, 0.001,0.0001,0.00001]
    l2s = [0, 0.001,0.01,0.1]
    num_epochs = 40
    for lr in lrs:
        for l2 in l2s:
            models = {'CNN': CNN(conv_layers_params=cnns[0],activation='leakyrelu',use_batch_norm=True, use_dropout=True),}
            for model_name, model in models.items():
                optimizer = optim.Adam(params = model.parameters(), lr = lr,weight_decay=0.00001)
                print(f"Training {model_name} with lr={lr}, L2={l2}")
                train_mse, test_mse, train_r2, test_r2, train_mae, test_mae = train(model,
                                                            optimizer,
                                                            train_data_loader,
                                                            test_data_loader, 
                                                            num_epochs=num_epochs,
                                                            l2=l2)
                # Store per-epoch metrics
                for epoch in range(num_epochs):
                    epoch_results.append({
                        "Model": model_name,
                        "Learning Rate": lr,
                        "L2 Weight Decay": l2,
                        "Epoch": epoch + 1,
                        "Train MSE": train_mse[epoch],
                        "Test MSE": test_mse[epoch],
                        "Train R2": train_r2[epoch],
                        "Test R2": test_r2[epoch],
                        "Train MAE": train_mae[epoch],
                        "Test MAE": test_mae[epoch]
                    })

                # Store final epoch metrics in the summary DataFrame
                results.append({
                    "Model": model_name,
                    "Learning Rate": lr,
                    "L2 Weight Decay": l2,
                    "Final Test MSE": test_mse[-1],
                    "Final Test R2": test_r2[-1],
                    "Final Test MAE": test_mae[-1]
                })

    lrs = np.logspace(-2,-4,6)#[0.1,0.01, 0.001,0.0001,0.00001]
    l2s = [0, 0.01,0.1]
    num_epochs = 40
    for lr in lrs:
        for l2 in l2s:
            models = {'Autoencoder': Autoencoder()}
            for model_name, model in models.items():
                optimizer = optim.Adam(params = model.parameters(), lr = lr,weight_decay=0)
                print(f"Training {model_name} with lr={lr}, L2={l2}")
                train_mse, test_mse, train_r2, test_r2, train_mae, test_mae = train(model,
                                                            optimizer,
                                                            train_data_loader,
                                                            test_data_loader, 
                                                            num_epochs=num_epochs,
                                                            l2=l2)
                # Store per-epoch metrics
                for epoch in range(num_epochs):
                    epoch_results.append({
                        "Model": model_name,
                        "Learning Rate": lr,
                        "L2 Weight Decay": l2,
                        "Epoch": epoch + 1,
                        "Train MSE": train_mse[epoch],
                        "Test MSE": test_mse[epoch],
                        "Train R2": train_r2[epoch],
                        "Test R2": test_r2[epoch],
                        "Train MAE": train_mae[epoch],
                        "Test MAE": test_mae[epoch]
                    })

                # Store final epoch metrics in the summary DataFrame
                results.append({
                    "Model": model_name,
                    "Learning Rate": lr,
                    "L2 Weight Decay": l2,
                    "Final Test MSE": test_mse[-1],
                    "Final Test R2": test_r2[-1],
                    "Final Test MAE": test_mae[-1]
                })

    # Convert to DataFrames
    df_results = pd.DataFrame(results)  # Final metrics summary
    df_epoch_results = pd.DataFrame(epoch_results)  # Per-epoch metrics
    df_results.to_csv(os.path.join(path,'training_data/grid_search_last.csv'))
    df_epoch_results.to_csv(os.path.join(path,'training_data/grid_search_full.csv'))

def main_cnn():
    '''
    Main function of project.
    '''
    num_epochs = 50
    lr = 0.001
    lr_step = 10
    weight_decay = 0.0001
    batch_size = 32
    # Initializing classes
    cnns = [{'out_channels': 5, 'kernel_size': 4, 'stride': 1, 'pool': 'max'},
             {'out_channels': 10, 'kernel_size': 7, 'stride': 1, 'pool': 'max'},
             {'out_channels': 20, 'kernel_size': 11, 'stride': 1, 'pool': 'max'}]
    model = CNN(conv_layers_params=cnns,
                activation='leakyrelu',
                use_batch_norm=True,
                use_dropout=True)
    optimizer = optim.AdamW(params = model.parameters(), lr = lr, weight_decay=weight_decay)
    train_data_loader, test_data_loader = get_data(batch_size=batch_size,test_size=0.1,normalize=True)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2, train_mae, test_mae = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step) # Run training
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
    torch.save(model.state_dict(),os.path.join(path,'models/cnn.pth'))
    for epoch in range(num_epochs):
                    results.append({
                        "Model": 'cnn',
                        "Learning Rate": lr,
                        "Weight Decay": weight_decay,
                        "Epoch": epoch + 1,
                        "Train MSE": train_mse[epoch],
                        "Test MSE": test_mse[epoch],
                        "Train R2": train_r2[epoch],
                        "Test R2": test_r2[epoch],
                        "Train MAE": train_mae[epoch],
                        "Test MAE": test_mae[epoch]
                    })
    
    df_results = pd.DataFrame(results)  # Final metrics summary
    df_results.to_csv(os.path.join(path,'training_data/cnn_best_long_01.csv'))
    


def main_autoencoder():
    '''
    Main function of project.
    '''
    # Initializing classes
    model = Autoencoder()
    optimizer = optim.Adam(params = model.parameters(), lr = 0.0001,weight_decay=0.001)
    trainer = Trainer()
    batch_size = 16

    # Dummy data
    # num_images = 10_000
    # images = torch.rand((num_images,1,128,128))
    # num_images = images.shape[0] # Get num images
    # params = torch.rand((num_images)) # Produce fake labels
    images = torch.load(os.path.join(path,'data/images.pt')) # Load generated data
    params = torch.load(os.path.join(path,'data/k.pt')) # Load generated data
    # params = (params - params.min()) / (params.max() - params.min())
    # Prepare data for the 
    X_train, X_test, y_train, y_test = train_test_split(images, params, test_size=0.2)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)

    start = time.time()
    trainer.train(model,optimizer,train_data_loader,test_data_loader, num_epochs=40) # Run training
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')

    torch.save(model.state_dict(), os.path.join(path,'models/autoencoder.pth'))

    plotter = Plotter(trainer,model)
    name='autoencoder'
    plotter.plot_mse(name)
    plotter.plot_r2(name)
    # plt.show()




if __name__ == '__main__':
    # grid_search()
    main_cnn()
    # main_autoencoder()
    