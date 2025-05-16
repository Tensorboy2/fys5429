from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms.functional as tf
import os
import pandas as pd
import numpy as np
path = os.path.dirname(__file__)
import random

class DataAugmentation:
    '''Data agumentation class for flipping images and flipping label'''
    def __init__(self, hflip=True, vflip=True, rotate=True, p=0.5):
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.p = p
    
    def __call__(self, image, image_filled, label, *args, **kwds):
        K = label.clone()
        if self.hflip and random.random() < self.p:
            image = tf.hflip(image)
            image_filled = tf.hflip(image_filled)
            K[1] *= -1
            K[2] *= -1
            
        if self.vflip and random.random() < self.p:
            image = tf.vflip(image)
            image_filled = tf.vflip(image_filled)
            K[1] *= -1
            K[2] *= -1
        
        if self.rotate and random.random() < self.p:
            '''
            Rotation of image with corresponding permutation of K.
            '''
            angle = random.choice([90,180,270])
            image = tf.rotate(image, angle=angle)
            image_filled = tf.rotate(image_filled, angle=angle)

            Kxx, Kxy, Kyx, Kyy = K

            if angle == 90:
                K = torch.tensor([Kyy, -Kyx, -Kxy, Kxx])
            elif angle == 180:
                K = torch.tensor([Kxx, -Kxy, -Kyx, Kyy])
            elif angle == 270:
                K = torch.tensor([Kyy, Kyx, Kxy, Kxx])

        return image, image_filled, K
        

class CustomDataset(Dataset):
    '''
    Custom dataset for lazy loading with torch data_loader.
    Gives image, image_filled and label(2 by 2 permeability tensor.)
    '''
    def __init__(self, label_path, image_path, image_filled_path, num_samples = None, transform = None, target_transform=None):
        # self.images = np.load(image_path, mmap_mode='r')['images']
        # self.images_filled = np.load(image_filled_path, mmap_mode='r')['images_filled']
        # self.labels = np.load(label_path, mmap_mode='r')['k']
        self.image_path = image_path
        self.image_filled_path = image_filled_path
        self.label_path = label_path

        if num_samples == None:
            with np.load(label_path, mmap_mode='r') as data:
                self.num_samples = data['k'].shape[0]

        # self.num_samples = num_samples or len(self.images)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # image = torch.from_numpy(self.images[idx]).float().unsqueeze(0)
        # image_filled = torch.from_numpy(self.images_filled[idx]).float().unsqueeze(0)
        # label = torch.from_numpy(self.labels[idx].flatten()).float()  # shape: (2, 2)

        with np.load(self.image_path, mmap_mode='r') as f_img:
            image = torch.from_numpy(f_img['images'][idx]).float().unsqueeze(0)
        with np.load(self.image_filled_path, mmap_mode='r') as f_imgf:
            image_filled = torch.from_numpy(f_imgf['images_filled'][idx]).float().unsqueeze(0)
        with np.load(self.label_path, mmap_mode='r') as f_label:
            label = torch.from_numpy(f_label['k'][idx].flatten()).float()

        if self.transform:
            image, image_filled, label = self.transform(image, image_filled, label)
        if self.target_transform:
            label = self.target_transform(label)

        return image, image_filled, label



def get_data(batch_size = 32,test_size=0.2, use_hv_flip=True, num_samples=None, num_workers=2):
    '''
    Function for getting data and turning them into the train and test loader.
    Optional: normalization, grid search size, mask outliers.
    '''
    # Paths:
    image_path = os.path.join(path, 'data/images.npz')
    image_filled_path = os.path.join(path, 'data/images_filled.npz')
    label_path = os.path.join(path, 'data/k.npz')

    if use_hv_flip:
        hv_flip = DataAugmentation(hflip=True, vflip=True, p=0.5)
    else: 
        hv_flip=None

    dataset = CustomDataset(
        image_path=image_path,
        image_filled_path=image_filled_path,
        label_path=label_path,
        num_samples=num_samples,
        transform=hv_flip,
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
                                   num_workers=num_workers,
                                   persistent_workers=True, 
                                   pin_memory=True)
    
    test_data_loader = DataLoader(test_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  num_workers=num_workers,
                                  persistent_workers=True, 
                                  pin_memory=True)

    return train_data_loader, test_data_loader

if __name__ == "__main__":
    print("Getting data...")
    train_data_loader, test_data_loader = get_data(num_workers=2)
    print("Data loaders ready, with lazy loading.")
