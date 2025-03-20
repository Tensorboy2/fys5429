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
    return img.astype(np.int32)


def check_percolation(img):
    '''
    ## Checks for percolation in media.
    If the top and bottom have one idenetical label 0< at the same place then the media is percolating.
    
    ## Params:
    - img (The binary layout of the media, solid == True)

    ## Returns:
    - True or False
    '''
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])
    labeled_media, num_features = label(np.logical_not(img), structure = structure)
    return np.any((labeled_media[0, :] > 0) & (labeled_media[0, :] == labeled_media[-1, :]))
    

@njit(fastmath=True)
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
            if fluid[x,y]:
                rho[x, y] = 1.0  # initial density

    # Gravity and forcing term
    grav = 0.00001
    F = np.zeros((Nx, Ny, 2), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            F[x, y, 0] = -grav  # gravity in x-direction (adjust as needed)
            F[x, y, 1] = 0.0

    # Relaxation parameter
    omega = 0.6
    relax_corr = 1.0 - 1.0/(2.0 * omega)

    # Initialize lattice distributions f using equilibrium with forcing
    f = np.empty((Nx, Ny, 9), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x,y]:
                # Square of velocity
                u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
                for i in range(9):
                    # Compute dot product u·c[i]
                    eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                    # Forcing term contribution
                    Fi = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
                    f[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi
    
    Fi = np.empty((Nx, Ny, 9), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                for i in range(9):
                    Fi[x, y, i] = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
    # Main simulation loop
    for step in range(T):

        # Collision: compute equilibrium distribution and relax toward it
        feq = np.empty((Nx, Ny, 9), dtype=np.float64)
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x,y]:
                    u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
                    for i in range(9):
                        eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                        # Fi = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
                        feq[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi[x, y, i]
                        f[x, y, i] = f[x, y, i] + omega * (feq[x, y, i] - f[x, y, i])
        
        # for x in range(Nx):
        #     for y in range(Ny):
        #         if fluid[x,y]:
        #             for i in range(9):
        #                 f[x, y, i] = f[x, y, i] + omega * (feq[x, y, i] - f[x, y, i])
        
        # Streaming step: propagate distributions
        f_stream = np.copy(f)
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    for i in range(9):
                        new_x = (x + c[i, 0]) % Nx
                        new_y = (y + c[i, 1]) % Ny
                        if fluid[new_x, new_y]:
                            f_stream[new_x, new_y, i] = f[x, y, i]
                        else:
                            f_stream[x, y, bounce_back_pairs[i]] = f[x, y, i]
        f = f_stream  # update f
        
        # Update macroscopic variables: density and velocity
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x,y]:
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
    u_x = 0.0
    tot_rho = 0.0
    num = 0.0
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x,y]:
                u_x += u[x, y,0] 
                tot_rho += rho[x, y]
                num += 1.0
    avg_u_x = u_x/num
    avg_rho = tot_rho/num
    mu = (relax_corr-1/2)/3
    k = avg_u_x*mu/(avg_rho*grav)
    return u, k

def plot_info(img,u,k,i):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title(f'mask, k={k:.5f}')
    plt.colorbar()
    plt.subplot(1,2,2)
    u_ = np.linalg.norm(u,axis=2)
    plt.imshow(u_)
    plt.title('velocity')
    plt.colorbar()
    plt.savefig(f'arb{i}.pdf')
    # plt.savefig(f'clear_channel.pdf')

def pipeline(start_index,stop_index,T,rank):
    '''
    # Pipeline function. 
    '''
    num_img = stop_index-start_index
    images_tensor = torch.zeros((num_img,1,128,128))
    k_tensor = torch.zeros((num_img))
    for i in range(num_img):
        img = make_img()
        # img = np.zeros((128,128),dtype=np.bool)
        # # img[30:100,0:30] = True
        # # img[30:100,100:127] = True
        # img[:,0] = True
        # img[:,-1] = True
        images_tensor[i,0] = torch.from_numpy(img)
        u,k = big_LBM(img,T)
        # plot_info(img,u,k,start_index+i)
        k_tensor[i] = k
    
    # torch.save(images_tensor, os.path.join(path,'data/images.pt'))
    # torch.save(k_tensor, os.path.join(path,'data/k.pt'))
    return images_tensor, k_tensor



import sys
from mpi4py import MPI
if __name__ == '__main__':
    num_samples = 16*63  # Total number of samples
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Process ID
    size = comm.Get_size() # Total number of processes
    chunk_size = num_samples // size  # Each process gets an equal chunk
    start_index = rank * chunk_size
    stop_index = start_index + chunk_size if rank != size - 1 else num_samples # Last process takes remaining
    simulation_time = 10_000
    
    start = time.time()
    images_local, k_local = pipeline(start_index,stop_index, simulation_time,rank)
    end = time.time()

    # Gather the data at rank 0
    all_images = comm.gather(images_local, root=0)
    all_k = comm.gather(k_local, root=0)

    comm.Barrier()
    if (rank==0):
        print(f'Total time for pipeline: {end-start} seconds.')
        # Concatenate tensors from all ranks
        images_tensor = torch.cat(all_images, dim=0)
        k_tensor = torch.cat(all_k, dim=0)

        # Save only on rank 0
        torch.save(images_tensor, os.path.join(path,'data/images_8.pt'))
        torch.save(k_tensor, os.path.join(path,'data/k_8.pt'))
        print(f"Saved tensors. Total samples: {images_tensor.shape[0]}")
    MPI.Finalize()
"""
mpirun -np 8 python3 project_1/data_pipeline.py   # Use this one, more efficient and hwthread is not faster on authors computer.
mpirun --use-hwthread-cpus -np 16 python3 project_1/data_pipeline.py
"""