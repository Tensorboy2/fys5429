import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.data import binary_blobs
from scipy.ndimage import label
import torch
import os
path = os.path.dirname(__file__)
from numba import njit

def make_img(img_size = 128, 
              volume_fraction = 0.5, 
              show_plot=False):
    '''
    Making the porous media.
    '''
    img = binary_blobs(length = img_size,
                        volume_fraction= volume_fraction,
                        n_dim = 2)
    img[:,0] = True
    img[:,-1] = True
    if not check_percolation(img = img):
        img = make_img()
    if show_plot:
        plt.imshow(img, cmap="binary")
        plt.axis('off')
        plt.show()
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
                      [1, 1, 1]])
    
    labeled_media, num_features = label(np.logical_not(img),
                                        structure = structure)
    top_labels = set(labeled_media[0,:])
    bottom_labels = set(labeled_media[-1,:])
    # print(top_labels)
    # print(bottom_labels)

    # plt.imshow(labeled_media)
    # plt.show()
    if top_labels - {0} & bottom_labels - {0}:
        # print('Percolation!')
        return True
    else:
        # print('No percolation!')
        return False
    

@njit
def big_LBM(solid, T):
    """
    A monolithic Lattice Boltzmann simulation.
    This single function inlines equilibrium, collision, streaming,
    bounce-back, and macroscopic updates.
    
    # Parameters:
    - solid : 2D numpy array (Nx, Ny) of type bool
            True indicates a solid node.
    - T     : int
            Number of simulation time steps.
              
    # Returns:
    - u            : 3D numpy array (Nx, Ny, 2) velocity field.
    - p            : 2D numpy array (Nx, Ny) pressure field (rho/3).
    - fluid_rho_t  : 1D array with total fluid density at each step.
    - solid_rho_t  : 1D array with total solid density at each step.
    """
    Nx = solid.shape[0]
    Ny = solid.shape[1]
    
    # Create fluid mask: fluid = not solid.
    fluid = np.empty((Nx, Ny), dtype=np.bool_)
    for x in range(Nx):
        for y in range(Ny):
            if solid[x, y]:
                fluid[x, y] = False
            else:
                fluid[x, y] = True
    
    # Lattice vectors (9 directions)
    c = np.empty((9, 2), dtype=np.int64)
    c[0, 0] =  0; c[0, 1] =  0
    c[1, 0] =  1; c[1, 1] =  0
    c[2, 0] =  0; c[2, 1] =  1
    c[3, 0] = -1; c[3, 1] =  0
    c[4, 0] =  0; c[4, 1] = -1
    c[5, 0] =  1; c[5, 1] =  1
    c[6, 0] = -1; c[6, 1] =  1
    c[7, 0] = -1; c[7, 1] = -1
    c[8, 0] =  1; c[8, 1] = -1

    # Lattice weights
    w = np.empty(9, dtype=np.float64)
    w[0] = 4.0/9.0
    w[1] = 1.0/9.0
    w[2] = 1.0/9.0
    w[3] = 1.0/9.0
    w[4] = 1.0/9.0
    w[5] = 1.0/36.0
    w[6] = 1.0/36.0
    w[7] = 1.0/36.0
    w[8] = 1.0/36.0

    # Bounce-back mapping
    bounce_back_pairs = np.empty(9, dtype=np.int64)
    bounce_back_pairs[0] = 0
    bounce_back_pairs[1] = 3
    bounce_back_pairs[2] = 4
    bounce_back_pairs[3] = 1
    bounce_back_pairs[4] = 2
    bounce_back_pairs[5] = 7
    bounce_back_pairs[6] = 8
    bounce_back_pairs[7] = 5
    bounce_back_pairs[8] = 6

    # Initialize macroscopic variables
    rho = np.empty((Nx, Ny), dtype=np.float64)
    u   = np.zeros((Nx, Ny, 2), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            rho[x, y] = 1.0  # initial density

    # Gravity and forcing term
    grav = 0.0001
    F = np.zeros((Nx, Ny, 2), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            F[x, y, 0] = -grav  # gravity in x-direction (adjust as needed)
            F[x, y, 1] = 0.0

    # Relaxation parameter
    omega = 0.7
    relax_corr = 1.0 - 1.0/(2.0 * omega)

    # Initialize lattice distributions f using equilibrium with forcing
    f = np.empty((Nx, Ny, 9), dtype=np.float64)
    for i in range(9):
        for x in range(Nx):
            for y in range(Ny):
                # Compute dot product u·c[i]
                eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                # Square of velocity
                u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
                # Forcing term contribution
                Fi = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
                f[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi
    # Apply bounce-back at initialization
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                for d in range(9):
                    nb_x = (x + c[d, 0]) % Nx
                    nb_y = (y + c[d, 1]) % Ny
                    if solid[nb_x, nb_y]:
                        f[x, y, bounce_back_pairs[d]] = f[x, y, d]
    
    # fluid_rho_t = np.empty(T, dtype=np.float64)
    # solid_rho_t = np.empty(T, dtype=np.float64)

    # Main simulation loop
    for step in range(T):
        # Collision: compute equilibrium distribution and relax toward it
        feq = np.empty((Nx, Ny, 9), dtype=np.float64)
        for i in range(9):
            for x in range(Nx):
                for y in range(Ny):
                    eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                    u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
                    Fi = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
                    feq[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi
        for x in range(Nx):
            for y in range(Ny):
                for i in range(9):
                    f[x, y, i] = f[x, y, i] + omega * (feq[x, y, i] - f[x, y, i])
        
        # Streaming step: propagate distributions
        f_stream = np.copy(f)
        # Initialize f_stream to zero to avoid uninitialized entries.
        # for x in range(Nx):
        #     for y in range(Ny):
        #         for i in range(9):
        #             f_stream[x, y, i] = f[x, y, i]
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    for i in range(9):
                        new_x = (x + c[i, 0]) % Nx
                        new_y = (y + c[i, 1]) % Ny
                        f_stream[new_x, new_y, i] = f[x, y, i]
        f = f_stream  # update f
        
        # Re-apply bounce-back at boundaries
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    for d in range(9):
                        nb_x = (x + c[d, 0]) % Nx
                        nb_y = (y + c[d, 1]) % Ny
                        if solid[nb_x, nb_y]:
                            f[x, y, bounce_back_pairs[d]] = f[x, y, d]
        
        # Update macroscopic variables: density and velocity
        for x in range(Nx):
            for y in range(Ny):
                sum_f = 0.0
                u0 = 0.0
                u1 = 0.0
                for i in range(9):
                    sum_f += f[x, y, i]
                    u0 += f[x, y, i] * c[i, 0]
                    u1 += f[x, y, i] * c[i, 1]
                rho[x, y] = sum_f
                if sum_f != 0.0:
                    u[x, y, 0] = u0 / sum_f
                    u[x, y, 1] = u1 / sum_f
                else:
                    u[x, y, 0] = 0.0
                    u[x, y, 1] = 0.0
        
        # fluid_sum = 0.0
        # solid_sum = 0.0
        # for x in range(Nx):
        #     for y in range(Ny):
        #         if fluid[x, y]:
        #             fluid_sum += rho[x, y]
        #         else:
        #             solid_sum += rho[x, y]
        # fluid_rho_t[step] = fluid_sum
        # solid_rho_t[step] = solid_sum
        # Note: Printing inside a njitted function is not supported.
        # You could return intermediate values for logging if needed.
    
    # Compute pressure (for example, p = rho/3)
    p = np.empty((Nx, Ny), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            p[x, y] = rho[x, y] / 3.0
            
    return u, p#, fluid_rho_t, solid_rho_t

@njit
def perm(u, p, mu=1, epsilon=1e-6):
    nx,ny = p.shape
    k = 0
    for i in range(nx):
        for j in range(ny):
            # Handle boundary conditions explicitly
            if i == 0:
                dp_x = p[i+1, j] - p[i, j]  # Forward difference at left boundary
            elif i == nx - 1:
                dp_x = p[i, j] - p[i-1, j]  # Backward difference at right boundary
            else:
                dp_x = (p[i+1, j] - p[i-1, j]) / 2  # Central difference

            if j == 0:
                dp_y = p[i, j+1] - p[i, j]  # Forward difference at bottom boundary
            elif j == ny - 1:
                dp_y = p[i, j] - p[i, j-1]  # Backward difference at top boundary
            else:
                dp_y = (p[i, j+1] - p[i, j-1]) / 2  # Central difference

            # Avoid division by zero
            dp_x = dp_x if abs(dp_x) > epsilon else epsilon
            dp_y = dp_y if abs(dp_y) > epsilon else epsilon

            # Compute permeability
            k_x = mu * u[i, j, 0] / dp_x
            k_y = mu * u[i, j, 1] / dp_y
            k += np.sqrt(k_x**2 + k_y**2)  # Magnitude of permeability
    k = k/(nx*ny)
    return k

def pipeline(num_img,T):
    '''
    # Pipeline function. 
    '''
    images_tensor = torch.zeros((num_img,1,128,128))
    k_tensor = torch.zeros((num_img))
    images_numpy = np.zeros((num_img,1,128,128))
    print(f'Making images...')
    for i in range(num_img):
        print(f'Making image: {i}')
        img = make_img()
        images_numpy[i,0] = img
        images_tensor[i,0] = torch.from_numpy(img)

    print(f'Simulating...')
    for i in range(num_img):
        print(f'Simulating image: {i}.')
        start_ = time.time()
        u,p = big_LBM(images_numpy[i,0],T)
        np.save(os.path.join(path,f'data/simulation_data/velocity_{i}'),u)
        np.save(os.path.join(path,f'data/simulation_data/pressure_{i}'),p)
        k_tensor[i] = perm(u,p)
        end_ = time.time()
        print(f'Time for simulation: {end_-start_} seconds.')
    torch.save(images_tensor, os.path.join(path,'data/images.pt'))
    torch.save(k_tensor, os.path.join(path,'data/k.pt'))


if __name__ == '__main__':
    num_samples = 100  # Total number of samples
    simulation_time = 10_000
    
    start = time.time()
    pipeline(num_samples, simulation_time)
    end = time.time()
    print(f'Total time for pipeline: {end-start} seconds.')

"""
100 imgs with 1000 steps in 920 seconds.
new numba code: 100 imgs with 10_000 steps in 1149.8 seconds.
"""