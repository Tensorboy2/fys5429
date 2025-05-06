import numpy as np
import os
path = os.path.dirname(__file__)
from numba import njit

@njit
def initialize_lattice():
    """
    ## Initialize lattice vectors

    ## Returns
    - c (NDArray) lattice vectors
    - W (NDArray) lattice weights
    - bounce_back_paris (NDArray) index pairs for bounce back
    """
    c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int64)
    w = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=np.float64)
    bounce_back_pairs = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
    return c, w, bounce_back_pairs

@njit
def initialize_fluid_mask(solid):
    """
    ## Quick function for inverting solid array to fluid

    Might be slower than: 
    ```python
    fluid =!solid
    ```
    
    ## Params
    - solid (NDArray) binary array (true = solid)
    
    ## Returns
    - fluid (NDArray) inverted solid array
    """
    Nx, Ny = solid.shape
    fluid = np.empty((Nx, Ny), dtype=np.bool_)
    for x in range(Nx):
        for y in range(Ny):
            fluid[x, y] = not solid[x, y]
    return fluid

@njit
def initialize_macros(fluid, grav=0.00001):
    """
    ## Initialize density, velocity and force field

    ## Params
    - fluid (NDArray) fluid binary array (true = fluid)
    - grav (float) gravity constant

    ## Returns
    - rho (NDArray) fluid density
    - u (NDArray) fluid velocity field
    - F (NDArray) body force
    """
    Nx, Ny = fluid.shape
    rho = np.empty((Nx, Ny), dtype=np.float64)
    u = np.zeros((Nx, Ny, 2), dtype=np.float64)
    F = np.zeros((Nx, Ny, 2), dtype=np.float64)
    # grav = 0.00001
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                rho[x,y] = 1.0
                F[x, y, 0] = -grav
    return rho, u, F

@njit
def compute_equilibrium(rho, u, c, w, Fi):
    """
    ## Equilibrium computation

    ## Params
    - rho (NDArray) fluid density
    - u (NDArray) fluid velocity field

    ## Returns
    - feq (NDArray) equilibrium lattice distribution  
    """
    Nx, Ny = rho.shape
    feq = np.zeros((Nx, Ny, 9), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            u_sq = u[x, y, 0]**2 + u[x, y, 1]**2
            for i in range(9):
                eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                feq[x, y, i] = w[i]*rho[x, y]*(1 + 3*eu + 4.5*eu**2 - 1.5*u_sq) + Fi[x, y, i]
    return feq
@njit
def collision(f, feq, omega):
    return f + omega * (feq - f)

@njit
def streaming(f, fluid, c, bounce_back_pairs):
    Nx, Ny, _ = f.shape
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
    return f_stream

@njit
def update_macros(f, c, fluid):
    Nx, Ny, _ = f.shape
    rho = np.zeros((Nx, Ny), dtype=np.float64)
    u = np.zeros((Nx, Ny, 2), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                for i in range(9):
                    rho[x, y] += f[x, y, i]
                    u[x, y, 0] += f[x, y, i] * c[i, 0]
                    u[x, y, 1] += f[x, y, i] * c[i, 1]
                if rho[x, y] != 0:
                    u[x, y, 0] /= rho[x, y]
                    u[x, y, 1] /= rho[x, y]
    return rho, u

@njit(fastmath=True)
def big_LBM(solid, T):
    Nx, Ny = solid.shape
    c, w, bounce_back_pairs = initialize_lattice()
    fluid = initialize_fluid_mask(solid)
    rho, u, F, grav = initialize_macros(fluid)
    omega = 0.6
    relax_corr = 1.0 - 1.0/(2.0 * omega)

    Fi = np.zeros((Nx, Ny, 9), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            for i in range(9):
                Fi[x, y, i] = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])

    f = compute_equilibrium(rho, u, c, w, Fi)

    for _ in range(T):
        feq = compute_equilibrium(rho, u, c, w, Fi)
        f = collision(f, feq, omega)
        f = streaming(f, fluid, c, bounce_back_pairs)
        rho, u = update_macros(f, c, fluid)

    last_u_max = np.amax(u)
    t = T
    stop_u = 1e-8
    while t < 100_000:
        t += 1
        feq, _ = compute_equilibrium(rho, u, c, w, F, relax_corr, Nx, Ny, fluid)
        collision(f, feq, omega, fluid, Nx, Ny)
        f = streaming(f, c, bounce_back_pairs, fluid, Nx, Ny)
        rho, u = update_macros(f, c, fluid, Nx, Ny)
        current_u_max = np.amax(u)
        if abs(current_u_max - last_u_max) < stop_u:
            break
        last_u_max = current_u_max

    return u, rho, fluid, relax_corr, grav

@njit
def k_tensor(u_x, u_y, fluid, relax_corr, grav):
    u_xx, u_xy = u_x[:,:,0],u_x[:,:,1]
    u_y = np.rot90(u_y)
    u_yx, u_yy = u_y[:,:,0],u_y[:,:,1]

    u_tensor  = np.zeros((2,2),dtype=np.float64)
    k_tensor  = np.zeros((2,2),dtype=np.float64)

    u_x = 0.0
    tot_rho = 0.0
    count = 0.0
    Nx, Ny = fluid.shape
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                u_tensor[0,0] += u_xx[x, y]
                u_tensor[0,1] += u_xy[x, y]
                count += 1.0
    avg_u_x = u_x / count
    mu = (relax_corr - 0.5) / 3
    return avg_u_x * mu / (grav)

# @njit(fastmath=True)
# def big_LBM(solid, T):
#     """
#     A Lattice Boltzmann simulation.
#     This single function inlines equilibrium, collision, streaming,
#     bounce-back, and macroscopic updates to optimize njit performance.
    
#     # Parameters:
#     - solid : 2D numpy array (Nx, Ny) of type bool
#             True indicates a solid node.
#     - T     : int
#             Number of simulation time steps.
              
#     # Returns:
#     - u            : 3D numpy array (Nx, Ny, 2) velocity field.
#     - k            : Permeability calculated from velocity field and density.
#     """
#     Nx = solid.shape[0]
#     Ny = solid.shape[1]
    
#     # Create fluid mask: fluid = not solid:
#     fluid = np.empty((Nx, Ny), dtype=np.bool_)
#     for x in range(Nx):
#         for y in range(Ny):
#             if solid[x, y]:
#                 fluid[x, y] = False
#             else:
#                 fluid[x, y] = True
    
#     # Lattice vectors (9 directions):
#     c = np.empty((9, 2), dtype=np.int64)
#     c[0, 0] =  0; c[0, 1] =  0
#     c[1, 0] =  1; c[1, 1] =  0
#     c[2, 0] =  0; c[2, 1] =  1
#     c[3, 0] = -1; c[3, 1] =  0
#     c[4, 0] =  0; c[4, 1] = -1
#     c[5, 0] =  1; c[5, 1] =  1
#     c[6, 0] = -1; c[6, 1] =  1
#     c[7, 0] = -1; c[7, 1] = -1
#     c[8, 0] =  1; c[8, 1] = -1

#     # Lattice weights:
#     w = np.empty(9, dtype=np.float64)
#     w[0] = 4.0/9.0
#     w[1] = 1.0/9.0
#     w[2] = 1.0/9.0
#     w[3] = 1.0/9.0
#     w[4] = 1.0/9.0
#     w[5] = 1.0/36.0
#     w[6] = 1.0/36.0
#     w[7] = 1.0/36.0
#     w[8] = 1.0/36.0

#     # Bounce-back mapping:
#     bounce_back_pairs = np.empty(9, dtype=np.int64)
#     bounce_back_pairs[0] = 0
#     bounce_back_pairs[1] = 3
#     bounce_back_pairs[2] = 4
#     bounce_back_pairs[3] = 1
#     bounce_back_pairs[4] = 2
#     bounce_back_pairs[5] = 7
#     bounce_back_pairs[6] = 8
#     bounce_back_pairs[7] = 5
#     bounce_back_pairs[8] = 6

#     # Initialize macroscopic variables:
#     rho = np.empty((Nx, Ny), dtype=np.float64)
#     u   = np.zeros((Nx, Ny, 2), dtype=np.float64)
#     for x in range(Nx):
#         for y in range(Ny):
#             if fluid[x,y]:
#                 rho[x, y] = 1.0  # initial density

#     # Gravity and forcing term:
#     grav = 0.00001
#     F = np.zeros((Nx, Ny, 2), dtype=np.float64)
#     for x in range(Nx):
#         for y in range(Ny):
#             F[x, y, 0] = -grav  # gravity in x-direction (adjust as needed)
#             F[x, y, 1] = 0.0

#     # Relaxation parameter:
#     omega = 0.6
#     relax_corr = 1.0 - 1.0/(2.0 * omega)

#     # Initialize lattice distributions f using equilibrium with forcing:
#     f = np.empty((Nx, Ny, 9), dtype=np.float64)
#     for x in range(Nx):
#         for y in range(Ny):
#             if fluid[x,y]:
#                 # Square of velocity:
#                 u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
#                 for i in range(9):
#                     # Compute dot product u·c[i]:
#                     eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
#                     # Forcing term contribution:
#                     Fi = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
#                     f[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi
    
#     Fi = np.empty((Nx, Ny, 9), dtype=np.float64)
#     for x in range(Nx):
#         for y in range(Ny):
#             if fluid[x, y]:
#                 for i in range(9):
#                     Fi[x, y, i] = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
#     # Main simulation loop
    
#     for _ in range(T):

#         # Collision: compute equilibrium distribution and relax toward it
#         feq = np.empty((Nx, Ny, 9), dtype=np.float64)
#         for x in range(Nx):
#             for y in range(Ny):
#                 if fluid[x,y]:
#                     u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
#                     for i in range(9):
#                         eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
#                         feq[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi[x, y, i]
#                         f[x, y, i] = f[x, y, i] + omega * (feq[x, y, i] - f[x, y, i])
        
        
#         # Streaming step: propagate distributions
#         f_stream = np.copy(f)
#         for x in range(Nx):
#             for y in range(Ny):
#                 if fluid[x, y]:
#                     for i in range(9):
#                         new_x = (x + c[i, 0]) % Nx
#                         new_y = (y + c[i, 1]) % Ny
#                         if fluid[new_x, new_y]:
#                             f_stream[new_x, new_y, i] = f[x, y, i]
#                         else:
#                             f_stream[x, y, bounce_back_pairs[i]] = f[x, y, i]
#         f = f_stream  # update f
        
#         # Update macroscopic variables: density and velocity
#         for x in range(Nx):
#             for y in range(Ny):
#                 if fluid[x,y]:
#                     sum_f = 0.0
#                     u0 = 0.0
#                     u1 = 0.0
#                     for i in range(9):
#                         sum_f += f[x, y, i]
#                         u0 += f[x, y, i] * c[i, 0]
#                         u1 += f[x, y, i] * c[i, 1]
#                     rho[x, y] = sum_f
#                     if sum_f != 0.0:
#                         u[x, y, 0] = u0 / sum_f
#                         u[x, y, 1] = u1 / sum_f
#                     else:
#                         u[x, y, 0] = 0.0
#                         u[x, y, 1] = 0.0


#     last_u_max = np.amax(u)
#     current_u_max = last_u_max+1
#     stop_u = 1e-8
#     t = T
#     while (t<100_000) and (abs(last_u_max - current_u_max)>stop_u):
#         t+=1
#         last_u_max = current_u_max 
#         # Collision: compute equilibrium distribution and relax toward it
#         feq = np.empty((Nx, Ny, 9), dtype=np.float64)
#         for x in range(Nx):
#             for y in range(Ny):
#                 if fluid[x,y]:
#                     u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
#                     for i in range(9):
#                         eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
#                         feq[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi[x, y, i]
#                         f[x, y, i] = f[x, y, i] + omega * (feq[x, y, i] - f[x, y, i])
        
        
#         # Streaming step: propagate distributions
#         f_stream = np.copy(f)
#         for x in range(Nx):
#             for y in range(Ny):
#                 if fluid[x, y]:
#                     for i in range(9):
#                         new_x = (x + c[i, 0]) % Nx
#                         new_y = (y + c[i, 1]) % Ny
#                         if fluid[new_x, new_y]:
#                             f_stream[new_x, new_y, i] = f[x, y, i]
#                         else:
#                             f_stream[x, y, bounce_back_pairs[i]] = f[x, y, i]
#         f = f_stream  # update f
        
#         # Update macroscopic variables: density and velocity
#         for x in range(Nx):
#             for y in range(Ny):
#                 if fluid[x,y]:
#                     sum_f = 0.0
#                     u0 = 0.0
#                     u1 = 0.0
#                     for i in range(9):
#                         sum_f += f[x, y, i]
#                         u0 += f[x, y, i] * c[i, 0]
#                         u1 += f[x, y, i] * c[i, 1]
#                     rho[x, y] = sum_f
#                     if sum_f != 0.0:
#                         u[x, y, 0] = u0 / sum_f
#                         u[x, y, 1] = u1 / sum_f
#                     else:
#                         u[x, y, 0] = 0.0
#                         u[x, y, 1] = 0.0
        
#         current_u_max = np.amax(u)
    
#     return u, rho

# def k_tensor(u, fluid, Nx, Ny, relax_corr, grav):
#     u_x = 0.0
#     tot_rho = 0.0
#     num = 0.0
#     for x in range(Nx):
#         for y in range(Ny):
#             if fluid[x,y]:
#                 u_x += u[x, y,0] 
#                 tot_rho += rho[x, y]
#                 num += 1.0
#     avg_u_x = u_x/num
#     avg_rho = tot_rho/num
#     mu = (relax_corr-1/2)/3
#     k = avg_u_x*mu/(avg_rho*grav)
#     return k