# Data
Inside this folder the data generation is done. 

The full pipeline requires a few executions. The first is *make_images.py*, which is set to generate 20 000 images. This image is then filled using flood fill to remove disconected pores that can lead to instabilities in the simulation later. This "image$\_$filled" is then checked for percolation on a binary dilution of that image. The reason for dilution is that if a channel is one pixel in width, it can lead to numerical instability while percolating. Dilution fills these tight channels so that only percolating images that percolate with minimum 3 pixels in width truly percolates. The image, the filled image is then stored in folders *images/*, *images_filled/* and *porosities/* with each image having a 5 digit id on the name of the saved npy object. 

Once the images are stored, they can be converted to *.npz* format for use of lazy loading later. 

The *data_pipeline_2.py* script is based on the one from project 1 : *data_pipeline.py*. This new version works in the same way with MPI to distribute the work. The files are stored now as checkpoints in *output_checkpoints/*. This is convenient as it allows the script to be stopped and continued if resources change or something happens. The simulation logic follows a basic Lattice-Boltzmann simulation. There is however slight deviations for optimizations of loops. 
Execution can be done with
```bash
mpirun -np 4 data_pipeline.py
```
Use more or less cores depending on hardware. *map_by* and *bind_by* can also be used to enforce distributed memory to individual cpu's.

As of *15.05.2025* there are not physical units in place, but this will be worked out in the following matter:
````python
L_phys = 1e-3  # physical domain size in meters (e.g., 1 mm)
U_phys = 1e-4  # physical velocity in m/s
nu_phys = 1e-6 # physical kinematic viscosity in m^2/s (water)
R_e = (U_phys*L_phys)/nu_phys # = (1e-4*1e-3)/1e-6 (((m/s)*m)/(m^2/s)) = 1e-1 (dimensionless)
Nx = solid.shape[0]  # Number of lattice points in x (128)
Ny = solid.shape[1]  # Number of lattice points in y (128)

dx = L_phys / Nx     # physical spacing between lattice nodes, 1e-3m/128 = 7.8125 * 10^-6 m
dt = dx / U_phys     # physical time step (so U_lattice = 1)  7.8125*10^-6/1e-3 m/(m/s) = 7.8125 * 10^-3 (s)

U_lattice = 1.0  
nu_lattice = nu_phys * dt / (dx ** 2) # 
omega = 1.0 / (3 * nu_lattice + 0.5)

omega = 1.0 / (3.0 * nu_lattice + 0.5)

g_phys = 9.81  # m/s² or any other value
g_lattice = g_phys * dt**2 / dx # 

grav = g_lattice
```