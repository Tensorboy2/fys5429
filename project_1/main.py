'''Main module for project 1.'''
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt


from cnn import CNN
from train import Trainer
from plot import Plotter

def main():
    # Initializing classes
    model = CNN(kernel_size_1=8,
                stride_1=4,
                kernel_size_2=8,
                stride_2=4,
                pool_1='max',
                pool_2='max',
                use_batch_norm=True,
                use_dropout=True)
    model.p=0.1
    optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
    trainer = Trainer()

    # Dummy data
    num_images = 256
    images = torch.rand((num_images,1,128,128))
    params = torch.rand((num_images))

    X_train, X_test, y_train, y_test = train_test_split(images, params, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)

    batch_size = 32
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)

    start = time.time()
    trainer.train(model,optimizer,train_data_loader,test_data_loader, num_epochs=5) # Run training
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')

    plotter = Plotter(trainer,model)
    plotter.plot_mse()
    plotter.plot_r2()
    plotter.visualize_kernels_1()
    plotter.visualize_kernels_2()
    plt.show()




if __name__ == '__main__':
    
    main()
    