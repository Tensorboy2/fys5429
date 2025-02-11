import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.data import binary_blobs
from scipy.ndimage import label

def make_img(img_size = 128, 
              volume_fraction = 0.5, 
              show_plot=False):
    '''
    Making the porous media.
    '''
    img = binary_blobs(length = img_size,
                        volume_fraction= volume_fraction,
                        n_dim = 2)
    if not check_percolation(img = img):
        img = make_img()
    if show_plot:
        plt.imshow(img, cmap="binary")
        plt.axis('off')
        plt.show()
    folder_path = 'project_1/data_pipeline/images/'
    # np.save(folder_path + 'img_' + f'{0}', img)
    return img 

def check_percolation(img):
    '''
    ## Checks for percolation in media.
    
    ## Params:
    - img (The binary layout of the media)

    ## Returns:
    - True or False

    How: By inverting the binary labels, one can use the label function 
    from "scipy.ndimage" to give unique labels to all connected fluids sections.
    Then if at least one of the labels on top exists at the bottom then there exist
    a fluid path through the media. 
    '''
    
    structure = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]) # Structure of neighbors on D2Q9 Lattice Boltzmann grid 
    
    labeled_media, num_features = label(np.logical_not(img),
                                        structure = structure)
    top_labels = set(labeled_media[0,:])
    bottom_labels = set(labeled_media[-1,:])
    if top_labels - {0} & bottom_labels - {0}:
        # print('Percolation!')
        return True
    else:
        # print('No percolation!')
        return False
    
def simulate(solid, T=1000, print_step=False, plot_u=False):
    """
    Simulates a 2D Lattice Boltzmann model for fluid flow through porous media.

    Parameters:
    ----------
    solid : ndarray
        A 2D binary grid where 1 represents solid obstacles and 0 represents fluid nodes.
    T : int, optional (default=1000)
        The number of simulation time steps.
    print_step : bool, optional (default=False)
        If True, prints progress updates at regular intervals.
    plot_u : bool, optional (default=False)
        If True, generates velocity field plots during the simulation.

    Returns:
    -------
    u : ndarray
        The velocity field of the fluid at the final simulation step.
    """
    solid = solid.astype(bool)
    Nx, Ny = solid.shape # Shape pf grid

    # Lattice vectors
    c = np.array([[0,  0], [1,  0], [0,  1], [-1,  0], [0, -1],
                [1,  1], [-1,  1], [-1, -1], [1, -1]])
    
    # Lattice weights
    w = np.array([4/9] + [1/9]*4 + [1/36]*4)

    # Initialize density
    rho = np.ones((Nx, Ny)) 

    # Initialize velocity
    u = np.zeros((Nx, Ny, 2))

    
    grav = 0.0001  # Gravity constant based on the viscosity
    
    # Initialize force field
    F = np.zeros((Nx, Ny, 2)) 
    F[:, :, 1] = -grav  # Gravity acts in negative y-direction (:,:,1)

    def equilibrium_with_forcing(rho, u, F):
        """
        ## Compute equilibrium with forcing term.

        ## Params:
        - rho (Density)
        - u (Velocity)
        - F (Force field)

        ## Returns:
        - feq (Equilibrium field)
        """
        feq = np.zeros((Nx, Ny, 9))
        for i in range(9):
            eu = np.einsum('ijk,k->ij', u, c[i]) #np.dot(c[i], u.transpose(0, 2, 1))
            Fi = w[i] * (1 - 1/(2*0.6)) * (3 * (F[:, :, 0] * c[i, 0] + F[:, :, 1] * c[i, 1]))#np.dot(c[i], F.transpose(0, 2, 1)))
            feq[:, :, i] = w[i] * rho * (1 + 3 * eu + 9/2 * eu**2 - 3/2 * np.sum(u**2, axis=2)) + Fi
        return feq

    # Initialize distribution function
    f = equilibrium_with_forcing(rho, u, F)

    # Define bounce-back indices (opposite directions)
    bounce_back_pairs = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # Opposite of each direction

    omega = 0.6
    feq = np.zeros((Nx, Ny, 9))

    for step in range(T):
         
        # Collision step velocity
        # feq = equilibrium_with_forcing(rho, u, F)
        for i in range(9):
            eu = np.einsum('ijk,k->ij', u, c[i]) #np.dot(c[i], u.transpose(0, 2, 1))
            Fi = w[i] * (1 - 1/(2*0.6)) * (3 * (F[:, :, 0] * c[i, 0] + F[:, :, 1] * c[i, 1]))#np.dot(c[i], F.transpose(0, 2, 1)))
            feq[:, :, i] = w[i] * rho * (1 + 3 * eu + 9/2 * eu**2 - 3/2 * np.sum(u**2, axis=2)) + Fi

        f = f + omega * (feq - f)  # Relaxation step


        # Streaming step
        for i in range(9):
            """
            This step uses the numpy roll method. This works like this:
            For a vector in f_i at a point (x,y) the vector is moved to
            the spot in the direction of c(i). So for example vector 
            f_1 at (0,0), its value is moved, using c(1), to (1,0) as 
            the lattice vector c_1 is (1,0). 
            """
            f[:, :, i] = np.roll(f[:, :, i], shift=c[i], axis=(0, 1))
        
        # Apply bounce-back at the solid boundary
        for i in range(9):
            f[solid, i] = f[solid, bounce_back_pairs[i]]
            

        # Compute macroscopic variables
        rho = np.sum(f, axis=2) # rho is the sum of of all values at the 0'th lattice vector.
        u = np.zeros((Nx, Ny, 2))
        for i in range(9):
            u += f[:, :, i][:, :, None] * c[i]
        u /= rho[:, :, None]

        if (step % int(T/10)== 0) and print_step:
            print(f'Current step: {step}, max u: {np.max(u)}, min u: {np.min(u)}')
            
   
    p = np.sum(f, axis=2)/3
    p = np.ma.masked_array(p, mask=solid)

    return u, p
import torch
def pipeline(num_img,T,print_step):
    images = torch.zeros((num_img,1,128,128))
    for i in range(num_img):
        print(f'Making image {i}')
        img = make_img()
        images[i,0] = torch.from_numpy(img)
        print(f'Simulating...')
        u = simulate(img,T,print_step)
        print(f'Simulation done!')
    torch.save(images, 'images.pt')

if __name__ == '__main__':
    start = time.time()
    num_img = 10_000
    T = 10
    print_step = False
    pipeline(num_img, T, print_step)
    end = time.time()
    print(f'Total time for simulation: {end-start} seconds.')


"""
100 imgs with 1000 steps in 920 seconds.
"""