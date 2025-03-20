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
from train import train
from feedforward import FeedForward

def get_data(batch_size = 32,test_size=0.2,normalize=True, mask=False, grid_search=False):
    '''
    Function for getting data and turning them into the train and test loader.
    '''
    if grid_search: # Less data for grid search
        images = torch.load(os.path.join(path,'data/images.pt'),weights_only=False)[1000:] # Load generated images
        params = torch.load(os.path.join(path,'data/k.pt'),weights_only=False)[1000:] # Load calculated permeability
    else:
        images = torch.load(os.path.join(path,'data/images.pt'),weights_only=False) # Load generated images
        params = torch.load(os.path.join(path,'data/k.pt'),weights_only=False) # Load calculated permeability

    
    if mask: # Mask outliers
        mask = (params >= 0) & (params <= 0.05)
        images, params = images[mask], params[mask]

    if normalize: # Normalize
        params_mean, params_std = params.mean(), params.std()
        params = (params-params_mean)/params_std

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(images, params, test_size=test_size,random_state=42)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)
    return train_data_loader, test_data_loader

def grid_search_cnn():
    '''
    Grid search function. 
    Preforms a grid search of given model hyper parameters and model architectures.
    '''
    # results = []
    epoch_results = []
    train_data_loader, test_data_loader = get_data(batch_size=32,test_size=0.20,grid_search=True)
    # Different convolution structures:
    cnns = [[{'out_channels': 10, 'kernel_size': 5, 'stride': 2, 'pool': 'max'},
             {'out_channels': 20, 'kernel_size': 7, 'stride': 1, 'pool': 'max'}],

            [{'out_channels': 10, 'kernel_size': 5, 'stride': 2, 'pool': 'max'},
             {'out_channels': 20, 'kernel_size': 7, 'stride': 1, 'pool': 'max'},
             {'out_channels': 40, 'kernel_size': 5, 'stride': 1, 'pool': 'max'}],

            [{'out_channels': 10, 'kernel_size': 5, 'stride': 2, 'pool': 'max'},
             {'out_channels': 20, 'kernel_size': 7, 'stride': 1, 'pool': 'max'},
             {'out_channels': 40, 'kernel_size': 5, 'stride': 1, 'pool': 'max'},
             {'out_channels': 60, 'kernel_size': 3, 'stride': 1, 'pool': 'max'}],]
    
    hidden_sizes =[[2],[2,4],[2,4,8]]


    lrs = [0.01, 0.001, 0.0001]
    l2s = [0, 0.0001,0.001]
    activations = ['leakyrelu','relu','sigmoid','tanh']
    num_epochs = 10
    for lr in lrs:
        for l2 in l2s:
            for activation in activations:
                for conv in cnns:
                    for hidden_size in hidden_sizes:

                        model = CNN(conv_layers_params=conv,hidden_sizes=hidden_size,activation=activation,use_batch_norm=True, use_dropout=True)
                        
                        optimizer = optim.Adam(params = model.parameters(), lr = lr,weight_decay=l2)
                        print(f"Training CNN with: lr={lr}, weight decay={l2}, number of convs={len(conv)}, number of linears={len(hidden_size)}, activation={activation}")
                        train_mse, test_mse, train_r2, test_r2, train_mae, test_mae, train_R, test_R = train(model,
                                                                    optimizer,
                                                                    train_data_loader,
                                                                    test_data_loader, 
                                                                    num_epochs=num_epochs,
                                                                    lr_step=num_epochs)
                        # Store per-epoch metrics
                        for epoch in range(num_epochs):
                            epoch_results.append({
                                "Learning Rate": lr,
                                "L2 Weight Decay": l2,
                                "CNNs": len(conv),
                                "Hiddens": len(hidden_size),
                                "Activation": activation,
                                "Epoch": epoch + 1,
                                "Train MSE": train_mse[epoch],
                                "Test MSE": test_mse[epoch],
                                "Train R2": train_r2[epoch],
                                "Test R2": test_r2[epoch],
                                "Train MAE": train_mae[epoch],
                                "Test MAE": test_mae[epoch],
                                "Train R": train_R[epoch],
                                "Test R": test_R[epoch],
                            })

    df_epoch_results = pd.DataFrame(epoch_results)  # Per-epoch metrics
    df_epoch_results.to_csv(os.path.join(path,'training_data/cnn_grid_search_full.csv'))

def grid_search_ff():
    # results = []
    epoch_results = []
    train_data_loader, test_data_loader = get_data(batch_size=32,test_size=0.20,grid_search=True)

    lrs = [0.01, 0.001, 0.0001]
    l2s = [0, 0.0001, 0.001]
    hidden_sizes =[[32],[32,64],[32,64,128]]
    activations = ['leakyrelu','relu','sigmoid','tanh']

    num_epochs = 10
    for lr in lrs:
        for l2 in l2s:
            for hidden_size in hidden_sizes:
                for activation in activations:

                    model = FeedForward(hidden_sizes=hidden_size,activation=activation)
                    optimizer = optim.Adam(params = model.parameters(), lr = lr,weight_decay=l2)
                    print(f"Training FFNN with lr={lr}, L2={l2}, number of linears={len(hidden_size)}, activation={activation}")
                    train_mse, test_mse, train_r2, test_r2, train_mae, test_mae, train_R, test_R = train(model,
                                                                optimizer,
                                                                train_data_loader,
                                                                test_data_loader, 
                                                                num_epochs=num_epochs,
                                                                lr_step=num_epochs)
                    # Store per-epoch metrics
                    for epoch in range(num_epochs):
                        epoch_results.append({
                                    "Learning Rate": lr,
                                    "L2 Weight Decay": l2,
                                    "Hiddens": len(hidden_size),
                                    "Activation": activation,
                                    "Epoch": epoch + 1,
                                    "Train MSE": train_mse[epoch],
                                    "Test MSE": test_mse[epoch],
                                    "Train R2": train_r2[epoch],
                                    "Test R2": test_r2[epoch],
                                    "Train MAE": train_mae[epoch],
                                    "Test MAE": test_mae[epoch],
                                    "Train R": train_R[epoch],
                                    "Test R": test_R[epoch],
                                })

    df_epoch_results = pd.DataFrame(epoch_results)  # Per-epoch metrics
    df_epoch_results.to_csv(os.path.join(path,'training_data/ff_grid_search_full.csv'))

def main_cnn():
    '''
    Main function of project.
    '''
    num_epochs = 20
    lr = 0.001
    lr_step = 10
    weight_decay = 0.0001
    batch_size = 32
    
    cnns = [{'out_channels': 10, 'kernel_size': 5, 'stride': 2, 'pool': 'max'},
             {'out_channels': 20, 'kernel_size': 7, 'stride': 1, 'pool': 'max'},
             {'out_channels': 40, 'kernel_size': 5, 'stride': 1, 'pool': 'max'}]
    model = CNN(conv_layers_params=cnns,
                hidden_sizes=[2,4],
                activation='leakyrelu',
                use_batch_norm=True,
                use_dropout=True)
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay=weight_decay)
    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=True,
                                                   mask=False,
                                                   grid_search=False)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2, train_mae, test_mae, train_R, test_R = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step) # Run training
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
    torch.save(model.state_dict(),os.path.join(path,'models/cnn.pth'))
    for epoch in range(num_epochs):
        results.append({"Epoch": epoch + 1,
                        "Train MSE": train_mse[epoch],
                        "Test MSE": test_mse[epoch],
                        "Train R2": train_r2[epoch],
                        "Test R2": test_r2[epoch],
                        "Train MAE": train_mae[epoch],
                        "Test MAE": test_mae[epoch],
                        "Train R": train_R[epoch],
                        "Test R": test_R[epoch]})
    
    df_results = pd.DataFrame(results)  # Final metrics summary
    df_results.to_csv(os.path.join(path,'training_data/cnn_best_long_01.csv'))
    


def main_feedforward():
    '''
    Main function of project.
    '''
    # Initializing classes
    
    model = FeedForward(hidden_sizes=[32,64])
    optimizer = optim.Adam(params = model.parameters(), lr = 0.1,weight_decay=0.0)
    batch_size = 32
    num_epochs = 100
    lr_step = 10
    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=True,
                                                   mask=False,
                                                   grid_search=False)
    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2, train_mae, test_mae, train_R, test_R = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step) # Run training
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
    torch.save(model.state_dict(),os.path.join(path,'models/ff.pth'))
    for epoch in range(num_epochs):
        results.append({"Epoch": epoch + 1,
                        "Train MSE": train_mse[epoch],
                        "Test MSE": test_mse[epoch],
                        "Train R2": train_r2[epoch],
                        "Test R2": test_r2[epoch],
                        "Train MAE": train_mae[epoch],
                        "Test MAE": test_mae[epoch],
                        "Train R": train_R[epoch],
                        "Test R": test_R[epoch]})
    
    df_results = pd.DataFrame(results)  # Final metrics summary
    df_results.to_csv(os.path.join(path,'training_data/ff_best_long_01.csv'))

    




if __name__ == '__main__':
    # grid_search_cnn()
    grid_search_ff()
    # main_cnn()
    # main_feedforward()
    