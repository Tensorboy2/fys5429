import numpy as np
import matplotlib.pyplot as plt
nx = 10
ny = 10
mu = 1
u = np.random.rand(nx,ny,2)
p = np.random.rand(nx,ny)
def perm(u, p, mu=1, epsilon=1e-6):
    # k_x = np.zeros((nx, ny))
    # k_y = np.zeros((nx, ny))
    k = 0#np.zeros((nx, ny))

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
    print(k)
    # return k_x, k_y, k
perm(u,p)

# plt.subplot(2,3,1)
# plt.imshow(u[:,:,0])
# plt.title(f'u_x')
# plt.colorbar()
# plt.subplot(2,3,2)
# plt.imshow(u[:,:,1])
# plt.colorbar()
# plt.title(f'u_y')
# plt.subplot(2,3,3)
# plt.imshow(p)
# plt.title(f'p')
# plt.colorbar()
# plt.subplot(2,3,4)
# plt.imshow(k_x)
# plt.title(f'k_x')
# plt.colorbar()
# plt.subplot(2,3,5)
# plt.imshow(k_y)
# plt.title(f'k_y')
# plt.colorbar()
# plt.subplot(2,3,6)
# plt.title(f'K = {np.mean(k):.2f}')
# plt.imshow(k)
# plt.colorbar()

# plt.show()