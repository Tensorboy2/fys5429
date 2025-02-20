'''Main module for project 1.'''
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import numpy as np
import os
path = os.path.dirname(__file__)

from cnn import CNN
from train import Trainer
from plot import Plotter


# def grid_search():
#     lrs = np.logspace()#[0.01,0.005,0.001,0.0005,0.0001]
#     l2s = [0,0.01,0.05,0.1,0.5]
    
#     best_model = 
#     for lr in lrs:
#         for l2 in l2s:

def main():
    '''
    Main function of project.
    '''
    # Initializing classes
    model = CNN(kernel_size_1=8,
                stride_1=4,
                kernel_size_2=8,
                stride_2=4,
                pool_1='max',
                pool_2='max',
                use_batch_norm=True,
                use_dropout=True)
    model.p=0.5
    optimizer = optim.Adam(params = model.parameters(), lr = 0.001,weight_decay=0)
    trainer = Trainer()
    batch_size = 100

    # Dummy data
    # num_images = 10_000
    # images = torch.rand((num_images,1,128,128))
    # num_images = images.shape[0] # Get num images
    # params = torch.rand((num_images)) # Produce fake labels
    images = torch.load(os.path.join(path,'data/images.pt')) # Load generated data
    params = torch.load(os.path.join(path,'data/k.pt')) # Load generated data

    # Prepare data for the 
    X_train, X_test, y_train, y_test = train_test_split(images, params, test_size=0.2)
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
    # plotter.plot_mse()
    # plotter.plot_r2()
    # plotter.visualize_kernels_1()
    # plotter.visualize_kernels_2()
    # plt.show()




if __name__ == '__main__':
    
    main()
    