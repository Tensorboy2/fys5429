from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd
import numpy as np
path = os.path.dirname(__file__)

class CustomDataset(Dataset):
    '''
    Custom dataset for lazy loading with torch data_loader.
    Gives image, image_filled and label(2 by 2 permeability tensor.)
    '''
    def __init__(self, label_path, image_path, image_filled_path, num_samples = None, transform = None, target_transform=None):
        self.images = np.load(image_path, mmap_mode='r')['images']
        self.images_filled = np.load(image_filled_path, mmap_mode='r')['images_filled']
        self.labels = np.load(label_path, mmap_mode='r')['k']

        self.num_samples = num_samples or len(self.images)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float().unsqueeze(0)
        image_filled = torch.from_numpy(self.images_filled[idx]).float().unsqueeze(0)
        label = torch.from_numpy(self.labels[idx].flatten()).float()  # shape: (2, 2)

        if self.transform:
            image = self.transform(image)
            image_filled = self.transform(image_filled)
        if self.target_transform:
            label = self.target_transform(label)

        return image_filled, label



def get_data(batch_size = 32,test_size=0.2, normalize=False, num_samples=None, num_workers=1):
    '''
    Function for getting data and turning them into the train and test loader.
    Optional: normalization, grid search size, mask outliers.
    '''
    # Paths:
    image_path = os.path.join(path, 'data/images.npz')
    image_filled_path = os.path.join(path, 'data/images_filled.npz')
    label_path = os.path.join(path, 'data/k.npz')

    dataset = CustomDataset(
        image_path=image_path,
        image_filled_path=image_filled_path,
        label_path=label_path,
        num_samples=num_samples,
        transform=None,
        target_transform=None
    )


    # Split into train and test sets:
    num_samples = len(dataset)
    train_size = int((1 - test_size) * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test datasets:
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=True, 
                                   num_workers=num_workers)
    test_data_loader = DataLoader(test_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  num_workers=num_workers)

    return train_data_loader, test_data_loader

if __name__ == "__main__":
    print("Getting data...")
    train_data_loader, test_data_loader = get_data(num_workers=4)
    print("Data loaders ready, with lazy loading.")
