'''Main module for project 1.'''
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split

from cnn import CNN
from train import Trainer

def main():
    # Initializing classes
    model = CNN(pool_1='max',pool_2='max',use_batch_norm=True,use_dropout=True)
    optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
    trainer = Trainer()

    # Dummy data
    num_images = 128
    images = torch.rand((num_images,1,128,128))
    params = torch.rand((num_images))

    X_train, X_test, y_train, y_test = train_test_split(images, params, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)

    batch_size = 64
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)

    
    trainer.train(model,optimizer,train_data_loader,test_data_loader) # Run training






if __name__ == '__main__':
    import time
    start = time.time()
    main()
    stop = time.time()
    print(f'Total run time: {stop-start} seconds')