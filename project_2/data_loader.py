from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd
import numpy as np
path = os.path.dirname(__file__)
from torchvision import read_image

class CustomDataset(Dataset):
    '''
    Custom dataset for lazy loading with torch data_loader.
    Gives image, image_filled and label(2 by 2 permeability tensor.)
    '''
    def __init__(self, label_dir, image_dir, image_filled_dir, num_samples = None, transform = None, target_transform=None):
        self.labels_dir = label_dir
        self.image_dir = image_dir
        self.image_filled_dir = image_filled_dir

        self.num_samples = num_samples or len(os.listdir(image_dir))

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fname = f"{idx:05d}.npy"

        image_path = os.path.join(self.image_dir, fname)
        image_filled_path = os.path.join(self.image_filled_dir, fname)
        label_path = os.path.join(self.label_dir, fname)

        image = torch.from_numpy(np.load(image_path)).float()
        image_filled = torch.from_numpy(np.load(image_filled_path)).float()
        label = torch.from_numpy(np.load(label_path)).float()  # shape: (2, 2)

        if self.transform:
            image = self.transform(image)
            image_filled = self.transform(image_filled)
        if self.target_transform:
            label = self.target_transform(label)

        return image, image_filled, label



def get_data(batch_size = 32,test_size=0.2,normalize=False, num_samples=None, device=None):
    '''
    Function for getting data and turning them into the train and test loader.
    Optional: normalization, grid search size, mask outliers.
    '''
    # Paths:
    image_dir = os.path.join(path, 'data/images')
    image_filled_dir = os.path.join(path, 'data/images_filled')
    label_dir = os.path.join(path, 'data/labels')

    dataset = CustomDataset(label_dir=label_dir, image_dir=image_dir, image_filled_dir=image_filled_dir, 
                            num_samples=num_samples, transform=None, target_transform=None)

    # TODO: Figure out standardization on permeability tensor.
    
    # if normalize: # Normalize
    #     params_mean, params_std = params.mean(), params.std()
    #     params = (params-params_mean)/params_std

    # Split into train and test sets
    num_samples = len(dataset)
    train_size = int((1 - test_size) * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test datasets
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader