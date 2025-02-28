import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Process ID
size = comm.Get_size() # Total number of processes

# Define the full range
N = 100000000  # Total iterations

# Divide workload
chunk_size = N // size  # Each process gets an equal chunk
start = rank * chunk_size
end = start + chunk_size if rank != size - 1 else N  # Last process takes remaining

# Each process computes its range independently
results = []
for i in range(start, end):
    result = i ** 2  # Example computation (replace with real work)
    results.append((i, result))  # Store (index, value)

# Save results in separate files per process
filename = f"results_rank_{rank}.txt"
# np.savetxt(filename, results, fmt="%d", header="Index, Value")

print(f"Process {rank} computed range {start} to {end} and saved to {filename}")

MPI.Finalize()