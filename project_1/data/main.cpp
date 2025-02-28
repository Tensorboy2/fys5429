#include <iostream>
#include "cnpy.h"
#include <vector>
#include <array>
#include <tuple>
#include <omp.h>
#include <string>
#include <chrono>

// Create fluid mask: fluid = not solid.
// (Here, if solid[x][y] != 0 then the node is solid, so fluid is false.)
std::vector<std::vector<bool>> fluid_function(const std::vector<std::vector<int>>& solid,int Nx, int Ny){
    std::vector<std::vector<bool>> fluid(Nx, std::vector<bool>(Ny, false));
    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
        fluid[x][y] = (solid[x][y] == 0);
        }
    }
    return fluid;
}
    


std::tuple<
    std::vector<std::vector<std::array<double, 2>>>, // velocity field u (Nx x Ny x 2)
    std::vector<std::vector<double>>               // pressure field p (Nx x Ny)
>
big_LBM(const std::vector<std::vector<int>>& solid, int T) {
    int Nx = solid.size();
    int Ny = solid[0].size();
    
    std::vector<std::vector<bool>> fluid = fluid_function(solid,Nx,Ny);
    
    
    // Lattice vectors (9 directions)
    static std::array<std::array<int, 2>, 9> c = {{
        {{ 0,  0}},
        {{ 1,  0}},
        {{ 0,  1}},
        {{-1,  0}},
        {{ 0, -1}},
        {{ 1,  1}},
        {{-1,  1}},
        {{-1, -1}},
        {{ 1, -1}}
    }};
    
    // Lattice weights
    static std::array<double, 9> w = {{
        4.0/9.0,
        1.0/9.0,
        1.0/9.0,
        1.0/9.0,
        1.0/9.0,
        1.0/36.0,
        1.0/36.0,
        1.0/36.0,
        1.0/36.0
    }};
    
    // Bounce-back mapping (i.e. which direction is the opposite of which)
    static std::array<int, 9> bounce_back_pairs = {{0, 3, 4, 1, 2, 7, 8, 5, 6}};
    
    // Initialize macroscopic variables.
    // Initial density (rho) is set to 1 everywhere.
    std::vector<std::vector<double>> rho(Nx, std::vector<double>(Ny, 1.0));
    
    // Velocity field u (Nx x Ny x 2), initially zero.
    std::vector<std::vector<std::array<double, 2>>> u(
        Nx, std::vector<std::array<double, 2>>(Ny, {0.0, 0.0})
    );
    
    // Gravity and forcing term F (Nx x Ny x 2)
    double grav = 0.0001;
    std::vector<std::vector<std::array<double, 2>>> F(
        Nx, std::vector<std::array<double, 2>>(Ny, {0.0, 0.0})
    );
    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            F[x][y][0] = -grav; // gravity in x-direction
            F[x][y][1] = 0.0;
        }
    }
    
    // Relaxation parameter and its correction factor.
    double omega = 0.7;
    double relax_corr = 1.0 - 1.0/(2.0 * omega);
    
    // Initialize the lattice distributions f (Nx x Ny x 9) using equilibrium with forcing.
    std::vector<std::vector<std::array<double, 9>>> f(
        Nx, std::vector<std::array<double, 9>>(Ny)
    );
    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            for (int i = 0; i < 9; i++) {
                double eu   = u[x][y][0]*c[i][0] + u[x][y][1]*c[i][1];
                double u_sq = u[x][y][0]*u[x][y][0] + u[x][y][1]*u[x][y][1];
                double Fi   = w[i] * relax_corr * 3.0 * (F[x][y][0]*c[i][0] + F[x][y][1]*c[i][1]);
                f[x][y][i] = w[i] * rho[x][y] * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi;
            }
        }
    }
    
    // Apply bounce-back at initialization.
    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            if (fluid[x][y]) {
                for (int d = 0; d < 9; d++) {
                    int nb_x = (x + c[d][0] + Nx) % Nx;
                    int nb_y = (y + c[d][1] + Ny) % Ny;
                    // If the neighbor is solid, apply bounce-back.
                    if (solid[nb_x][nb_y] != 0) {
                        f[x][y][bounce_back_pairs[d]] = f[x][y][d];
                    }
                }
            }
        }
    }
    
    // Main simulation loop.
    for (int step = 0; step < T; step++) {
        // Collision: compute the equilibrium distribution (feq).
        std::vector<std::vector<std::array<double, 9>>> feq(
            Nx, std::vector<std::array<double, 9>>(Ny)
        );
        for (int x = 0; x < Nx; x++) {
            for (int y = 0; y < Ny; y++) {
                for (int i = 0; i < 9; i++) {
                    double eu   = u[x][y][0]*c[i][0] + u[x][y][1]*c[i][1];
                    double u_sq = u[x][y][0]*u[x][y][0] + u[x][y][1]*u[x][y][1];
                    double Fi   = w[i] * relax_corr * 3.0 * (F[x][y][0]*c[i][0] + F[x][y][1]*c[i][1]);
                    feq[x][y][i] = w[i] * rho[x][y] * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi;
                }
            }
        }
        // Relaxation: relax the distribution toward equilibrium.
        for (int x = 0; x < Nx; x++) {
            for (int y = 0; y < Ny; y++) {
                for (int i = 0; i < 9; i++) {
                    f[x][y][i] = f[x][y][i] + omega * (feq[x][y][i] - f[x][y][i]);
                }
            }
        }
        
        // Streaming: propagate the distributions.
        auto f_stream = f; // create a copy of f
        for (int x = 0; x < Nx; x++) {
            for (int y = 0; y < Ny; y++) {
                if (fluid[x][y]) {
                    for (int i = 0; i < 9; i++) {
                        int new_x = (x + c[i][0] + Nx) % Nx;
                        int new_y = (y + c[i][1] + Ny) % Ny;
                        if (fluid[new_x][new_y]) {
                            f_stream[new_x][new_y][i] = f[x][y][i];
                        }
                        else {
                            f_stream[x][y][bounce_back_pairs[i]] = f[x][y][i];
                            
                        }
                        
                    }
                }
            }
        }
        f = f_stream;  // update f with the streamed values
        
        // Re-apply bounce-back at boundaries.
        // for (int x = 0; x < Nx; x++) {
        //     for (int y = 0; y < Ny; y++) {
        //         if (fluid[x][y]) {
        //             for (int d = 0; d < 9; d++) {
        //                 int nb_x = (x + c[d][0] + Nx) % Nx;
        //                 int nb_y = (y + c[d][1] + Ny) % Ny;
                        
        //             }
        //         }
        //     }
        // }
        
        // Update macroscopic variables: density and velocity.
        for (int x = 0; x < Nx; x++) {
            for (int y = 0; y < Ny; y++) {
                double sum_f = 0.0;
                double u0 = 0.0;
                double u1 = 0.0;
                for (int i = 0; i < 9; i++) {
                    sum_f += f[x][y][i];
                    u0 += f[x][y][i] * c[i][0];
                    u1 += f[x][y][i] * c[i][1];
                }
                rho[x][y] = sum_f;
                if (sum_f != 0.0) {
                    u[x][y][0] = u0 / sum_f;
                    u[x][y][1] = u1 / sum_f;
                } else {
                    u[x][y][0] = 0.0;
                    u[x][y][1] = 0.0;
                }
            }
        }
    }
    
    // Compute pressure (for example, p = rho/3).
    std::vector<std::vector<double>> p(Nx, std::vector<double>(Ny, 0.0));
    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            p[x][y] = rho[x][y] / 3.0;
        }
    }
    
    return std::make_tuple(u, p);
}


int main() {

    static int T = 10000;
    int Nx = 128, Ny = 128;

    // Access the data


    const int N = 1;
    int results[N];

    // Parallelized for loop using OpenMP
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        int thread_id = omp_get_thread_num();
        // Load the .npy file
        std::string path = "../simulation_data/geometry_" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr = cnpy::npy_load(path);
        // int* solid = arr.data<int>();
        int* solid_data = arr.data<int>();
        size_t rows = arr.shape[0];  // Assuming arr.shape[0] exists
        size_t cols = arr.shape[1];  // Assuming arr.shape[1] exists

        std::vector<std::vector<int>> solid(rows, std::vector<int>(cols));
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                solid[r][c] = solid_data[r * cols + c];  // Flattened index conversion
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Data loading time for thread: " << thread_id << " time: " << duration.count() << " milliseconds" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        std::tuple<std::vector<std::vector<std::array<double, 2>>>,std::vector<std::vector<double>>> result = big_LBM(solid,T);
        end = std::chrono::high_resolution_clock::now();
        // cnpy::npy_save("../cpp_simulation_data/velocity_" + std::to_string(i) + ".npy", result, "w");
        // Calculate the duration
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Output the duration in milliseconds
        std::cout << "Simulation time for thread: " << thread_id << " time: " << duration.count() << " milliseconds" << std::endl;
        // std::cout << "Thread " << thread_id << std::endl;
    }



    return 0;
}

// g++ -std=c++11 -o main cnpy.cpp main.cpp -Iinclude -lz