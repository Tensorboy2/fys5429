import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
import os
path = os.path.dirname(__file__)

from percolation import find_percolating_clusters, fill_non_percolating

def porous_media_generation(num_img=1, img_size=128, porosity_range=[0.1,0.5], sigma=4.0):
    """
    ## Generate set of porous media.

    This function generates a set of 2D porous media based porosity and image size. 
    The media is defined to be periodic in both axis and it ensures percolation in both directions.

    ## Params:
    - num_img (int) The number of images for generation
    - img_size (int) The size of the image 
    - img_size ([float,float]) The size of the image 

    """
    images = np.zeros((num_img,img_size,img_size),dtype=np.int32)
    for i in range(0,num_img):
        porosity = porosity_range[0] + np.random.rand()*(porosity_range[1] - porosity_range[0])
        cont = True
        while (cont):
            img = periodic_binary_blobs(shape=(img_size,img_size), blob_density=porosity, sigma=sigma,seed=np.random.randint(0,num_img))
            percolates_x = check_percolation(img)
            percolates_y = check_percolation(np.rot90(img)) # Rotate image to check percolation in y direction
            percolates = percolates_x and percolates_y
            cont = not percolates
        images[i] = img
    np.save(os.path.join(path,"images.npy"),images)



def make_img(img_size = 128, 
              blob_density = 0.5,
              sigma = 2.0,
              seed = 42):
    '''
    Making the porous media.
    '''
    img = periodic_binary_blobs(shape=(img_size,img_size), blob_density=blob_density, sigma=sigma,seed=seed)
    if not check_percolation(img = img) or not check_percolation(img = np.rot90(img)): #
        seed+=1
        img = make_img(seed=seed)
    return img.astype(np.int32)


def check_percolation(img):
    '''
    ## Checks for percolation in media.
    If the top and bottom have one idenetical label 0< at the same place then the media is percolating.
    
    ## Params:
    - img (The binary layout of the media, solid == True)

    ## Returns:
    - True or False (true is percolation which is good)
    '''
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])
    labeled_media, _ = label(np.logical_not(img), structure = structure)
    return np.any((labeled_media[0, :] > 0) & (labeled_media[0, :] == labeled_media[-1, :]))
   
def periodic_binary_blobs(shape=(128, 128), 
                          blob_density=0.2, 
                          sigma=2.0, 
                          seed=None):
    
    rng = np.random.default_rng(seed)
    noise = rng.random(shape) # Generate uniform random noise

    # Gaussian filter, using wrap mode for periodic boundary conditions:
    smooth_noise = gaussian_filter(noise, sigma=sigma, mode='wrap') 
    
    # Threshold to create binary blobs:
    threshold = np.percentile(smooth_noise, (1.0 - blob_density) * 100)
    blobs = smooth_noise > threshold
    return blobs


def compute_porosity(blobs):
    # Assuming pore space is False
    return 1-np.sum(blobs)/blobs.shape[-1]**2

def demo():
    # Parameter grid
    sigmas = [2.0, 4.0, 6.0]
    densities = [0.1, 0.3, 0.5]

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    # Generate subplots
    for i, sigma in enumerate(sigmas):
        for j, density in enumerate(densities):
            ax = axes[i, j]
            blobs = periodic_binary_blobs(shape=(128, 128), 
                                            blob_density=density, 
                                            sigma=sigma,seed=6)
            p_x, p_y, labeled = find_percolating_clusters(blobs)
            # blobs = make_img(img_size=128, blob_density=density, sigma=sigma,seed = 42)
            filled = fill_non_percolating(blobs)
            porosity = compute_porosity(filled)
            ax.imshow(filled, cmap='gray')
            ax.set_title(f"sigma={sigma}, blob_density={density}\n porosity={porosity:.2f}, px: {p_x} py: {p_y}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(path,"example_media.pdf"))
    plt.clf()

if __name__ == "__main__":
    demo()
    # num_images = 1
    # img_size = 128
    # porosity_range = [0.1,0.5]

    # porous_media_generation(num_img=num_images,
    #                         img_size=img_size,
    #                         porosity_range=porosity_range)

    # # img = np.load(os.path.join(path,"images.npy"))
    # # plt.imshow(img[0])
    # # plt.show()
    # A = np.array([[1,2],[3,4]])
    # print(A)
    # A_1 = np.rot90(A,k=1,axes=(1,0)) # Rotated 90 deg counter clockwise
    # print(A_1)
    # A_2 = np.rot90(A_1,k=-1,axes=(1,0)) # Rotated 90 deg clockwise
    # print(A_2)
    
