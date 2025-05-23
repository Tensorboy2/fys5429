from torch.utils.data import DataLoader, Dataset, TensorDataset
# from sklearn.model_selection import train_test_split
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
        labels_all = np.load(label_path, mmap_mode='r')['k']
        images_all = np.load(image_path, mmap_mode='r')['images']
        images_filled_all = np.load(image_filled_path, mmap_mode='r')['images_filled']

        # Limit by num_samples
        self.num_samples = num_samples or labels_all.shape[0]
        self.labels = labels_all[:self.num_samples]
        self.images = images_all[:self.num_samples]
        self.images_filled = images_filled_all[:self.num_samples]

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float().unsqueeze(0)
        image_filled = torch.from_numpy(self.images_filled[idx]).float().unsqueeze(0)
        label = torch.from_numpy(self.labels[idx].flatten()).float()  # shape: (2, 2)

        # with np.load(self.image_path, mmap_mode='r') as f_img:
        #     image = torch.from_numpy(f_img['images'][idx]).float().unsqueeze(0)
        # with np.load(self.image_filled_path, mmap_mode='r') as f_imgf:
        #     image_filled = torch.from_numpy(f_imgf['images_filled'][idx]).float().unsqueeze(0)
        # with np.load(self.label_path, mmap_mode='r') as f_label:
        #     label = torch.from_numpy(f_label['k'][idx].flatten()).float()

        if self.transform:
            image, image_filled, label = self.transform(image, image_filled, label)
        if self.target_transform:
            label = self.target_transform(label)

        return image, image_filled, label



def get_data(batch_size = 32,
             test_size=0.2, 
             hflip=True, 
             vflip=True, 
             rotate=True, 
             num_samples=None, 
             num_workers=4):
    '''
    Function for getting data and turning them into the train and test loader.
    Optional: normalization, grid search size, mask outliers.
    '''
    # Paths:
    image_path = os.path.join(path, 'data/images.npz')
    image_filled_path = os.path.join(path, 'data/images_filled.npz')
    label_path = os.path.join(path, 'data/k.npz')

    hv_flip = DataAugmentation(hflip=hflip, 
                               vflip=vflip,
                               rotate=rotate,
                               p=0.5)


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
    print(f"Num datapoints: {num_samples}")
    train_size = int((1 - test_size) * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test datasets:
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=True, 
                                   num_workers=num_workers,
                                #    persistent_workers=True, 
                                   pin_memory=True)
    
    test_data_loader = DataLoader(test_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  num_workers=num_workers,
                                #   persistent_workers=True, 
                                  pin_memory=True)

    return train_data_loader, test_data_loader

if __name__ == "__main__":
    print("Getting data...")
    train_data_loader, test_data_loader = get_data(num_workers=4,num_samples=4000)
    print(len(train_data_loader))
    print("Data loaders ready, with lazy loading.")
