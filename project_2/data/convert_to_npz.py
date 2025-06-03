'''
convert_to_npz.py

Module for converting folders to npz format.
'''
import numpy as np
import os
from tqdm import tqdm
path = os.path.dirname(__file__)

folder = "images_filled"
all_images = []

file_list = sorted(f for f in os.listdir(folder) if f.endswith('.npy'))

for filename in tqdm(file_list):
    img = np.load(os.path.join(folder, filename))
    all_images.append(img)

all_images = np.stack(all_images)

np.savez_compressed("images_filled.npz", images_filled=all_images)