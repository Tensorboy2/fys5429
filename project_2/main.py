import torch
import torch.nn as nn
torch.manual_seed(0)
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
# import numpy as np
# import pandas as pd
import os
path = os.path.dirname(__file__)
import torch.nn.init as init

from models.cnn import CNN
from models.simplenet import SimpleNet
from models.resnet import ResNet
from models.convnext import ConvNeXt
from models.graczyk import GraczykNet
from models.bestnet import BestNet
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


def main_simple():
    '''
    Longer training of CNN.
    '''
    # Hyper parameters:
    num_epochs = 100
    lr = 1e-6
    lr_step = num_epochs
    weight_decay = 0.0
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleNet().to(device)
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay=weight_decay)

    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=True,
                                                   mask=False,
                                                   grid_search=False,
                                                   device = device)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2 = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
def main_resnet():
    '''
    Longer training of CNN.
    '''
    # Hyper parameters:
    num_epochs = 100
    lr = 1e-6
    lr_step = num_epochs
    weight_decay = 0.0
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet().to(device)
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay=weight_decay)

    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=True,
                                                   mask=False,
                                                   grid_search=False,
                                                   device = device)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2 = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
def main_convnext():
    '''
    Longer training of CNN.
    '''
    # Hyper parameters:
    num_epochs = 100
    lr = 1e-6
    lr_step = num_epochs
    weight_decay = 0.0
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNeXt().to(device)
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay=weight_decay)

    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=True,
                                                   mask=False,
                                                   grid_search=False,
                                                   device = device)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2 = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
def main_graczyknet():
    '''
    Longer training of CNN.
    '''
    # Hyper parameters:
    num_epochs = 100
    lr = 1e-2
    lr_step = 10
    weight_decay = 0.0
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraczykNet().to(device)
    optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum=0.9, weight_decay=weight_decay)

    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=True,
                                                   mask=False,
                                                   grid_search=False,
                                                   device = device)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2 = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
def main_cnn():
    '''
    Longer training of CNN.
    '''
    # Hyper parameters:
    num_epochs = 200
    lr = 0.001
    lr_step = 10
    weight_decay = 0.0
    batch_size = 32
    cnns = [{'out_channels': 10, 'kernel_size': 9, 'stride': 2, 'pool': 'max'},
             {'out_channels': 20, 'kernel_size': 7, 'stride': 1, 'pool': 'max'},
             {'out_channels': 40, 'kernel_size': 5, 'stride': 1, 'pool': 'max'}]
    model = CNN(conv_layers_params=cnns,
                hidden_sizes=[2,4],
                activation='leakyrelu',
                use_batch_norm=True,
                use_dropout=True)
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay=weight_decay)
    print(model)
    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=True,
                                                   mask=False,
                                                   grid_search=False)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2 = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')
def main_bestnet():
    '''
    Longer training of CNN.
    '''
    # Hyper parameters:
    num_epochs = 200
    lr = 0.001
    lr_step = 20
    weight_decay = 1e-5
    batch_size = 32
    model = BestNet()
    optimizer = optim.AdamW(params = model.parameters(), lr = lr, weight_decay=weight_decay)
    print(model)
    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=True,
                                                   mask=False,
                                                   grid_search=False)

    results = []
    start = time.time()
    train_mse, test_mse, train_r2, test_r2 = train(model,
                                                                        optimizer,
                                                                        train_data_loader,
                                                                        test_data_loader, 
                                                                        num_epochs=num_epochs,
                                                                        lr_step = lr_step)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')

if __name__ == '__main__':
    # main_simple()
    # main_resnet()
    # main_convnext()
    # main_graczyknet()
    # main_cnn()
    main_bestnet()
