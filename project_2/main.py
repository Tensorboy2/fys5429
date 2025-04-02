import torch
import torch.nn as nn
torch.manual_seed(0)
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
import os
path = os.path.dirname(__file__)
import torch.nn.init as init

from models.cnn import CNN
from models.simplenet import SimpleNet
# from feedforward import FeedForward
from train import train

def get_data(batch_size = 32,test_size=0.2,normalize=True, mask=False, grid_search=False, device=None):
    '''
    Function for getting data and turning them into the train and test loader.
    Optional: normalization, grid search size, mask outliers.
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
    X_train, X_test, y_train, y_test = train_test_split(images.to(device), params.to(device), test_size=test_size,random_state=42)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)
    return train_data_loader, test_data_loader

def main_cnn():
    '''
    Longer training of CNN.
    '''
    # Hyper parameters:
    num_epochs = 2000
    lr = 0.001
    lr_step = num_epochs
    weight_decay = 0.0
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnns = [{'out_channels': 10, 'kernel_size': 9, 'stride': 2, 'pool': 'max'},
             {'out_channels': 20, 'kernel_size': 9, 'stride': 1, 'pool': 'max'},
             {'out_channels': 40, 'kernel_size': 9, 'stride': 1, 'pool': 'max'}]
    model = CNN(conv_layers_params=cnns,
                hidden_sizes=[],
                activation='relu',
                use_batch_norm=True,
                use_dropout=True).to(device)
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay=weight_decay)

    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=False,
                                                   mask=True,
                                                   grid_search=False,
                                                   device = device)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2, train_mae, test_mae, train_R, test_R = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
    
    torch.save(model.state_dict(),os.path.join(path,'models/cnn.pth')) # Save model for later inference
    
    for epoch in range(num_epochs): # Store per-epoch metrics
        results.append({"Epoch": epoch + 1,
                        "Train MSE": train_mse[epoch],
                        "Test MSE": test_mse[epoch],
                        "Train R2": train_r2[epoch],
                        "Test R2": test_r2[epoch],
                        "Train MAE": train_mae[epoch],
                        "Test MAE": test_mae[epoch],
                        "Train R": train_R[epoch],
                        "Test R": test_R[epoch]})
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(path,'cnn_long.csv'))

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier works well for fully connected layers
        init.zeros_(m.bias)  # Initialize biases to zero

def main_simple():
    '''
    Longer training of CNN.
    '''
    # Hyper parameters:
    num_epochs = 2000
    lr = 0.001
    lr_step = num_epochs
    weight_decay = 0.0
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleNet().to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay=weight_decay)

    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=True,
                                                   mask=False,
                                                   grid_search=False,
                                                   device = device)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2, train_mae, test_mae, train_R, test_R = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
    
    torch.save(model.state_dict(),os.path.join(path,'models/cnn.pth')) # Save model for later inference
    
    for epoch in range(num_epochs): # Store per-epoch metrics
        results.append({"Epoch": epoch + 1,
                        "Train MSE": train_mse[epoch],
                        "Test MSE": test_mse[epoch],
                        "Train R2": train_r2[epoch],
                        "Test R2": test_r2[epoch],
                        "Train MAE": train_mae[epoch],
                        "Test MAE": test_mae[epoch],
                        "Train R": train_R[epoch],
                        "Test R": test_R[epoch]})
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(path,'simple.csv'))

if __name__ == '__main__':
    # main_cnn()
    main_simple()
