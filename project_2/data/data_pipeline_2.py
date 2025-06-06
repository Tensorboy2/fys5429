import numpy as np
import time
import torch
import os
from mpi4py import MPI
path = os.path.dirname(__file__)
from simulation.lbm import big_LBM

out_dir = os.path.join(path, "output_checkpoints")
os.makedirs(out_dir, exist_ok=True)

def image_already_done(index):
    '''Function for chacking if images already is simulated.'''
    return (
        os.path.exists(os.path.join(out_dir, f"u_x_{index:05d}.npy")) and
        os.path.exists(os.path.join(out_dir, f"u_y_{index:05d}.npy")) and
        os.path.exists(os.path.join(out_dir, f"k_{index:05d}.npy"))
    )

def save_result(index, u_x, u_y, k):
    '''Function for saving simulated results.'''
    np.save(os.path.join(out_dir, f"u_x_{index:05d}.npy"), u_x)
    np.save(os.path.join(out_dir, f"u_y_{index:05d}.npy"), u_y)
    np.save(os.path.join(out_dir, f"k_{index:05d}.npy"), k)

def main_mpi_from_dir():
    '''
    MPI version of data pipeline for simulating and calculating permeability.
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    T = 10_000
    image_dir = os.path.join(path, "images_filled")

    if rank == 0:
        print("[INFO] Scanning image filenames...")
        all_filenames = sorted(os.listdir(image_dir))
        all_filenames = [f for f in all_filenames if f.endswith(".npy")]
        N = len(all_filenames)
        print(f"[INFO] Found {N} image files.")
    else:
        all_filenames = None
        N = None

    # Broadcast total number of images:
    N = comm.bcast(N if rank == 0 else None, root=0)

    # Ensure even distribution:
    local_N = N // size
    remainder = N % size
    if rank < remainder:
        local_start = rank * (local_N + 1)
        local_end = local_start + local_N + 1
    else:
        local_start = rank * local_N + remainder
        local_end = local_start + local_N

    if rank == 0:
        filenames_chunks = []
        for r in range(size):
            if r < remainder:
                s = r * (local_N + 1)
                e = s + local_N + 1
            else:
                s = r * local_N + remainder
                e = s + local_N
            filenames_chunks.append(all_filenames[s:e])
    else:
        filenames_chunks = None

    # Scatter filenames:
    local_filenames = comm.scatter(filenames_chunks, root=0)

    for i, fname in enumerate(local_filenames):
        global_idx = local_start + i

        if image_already_done(global_idx):
            print(f"[RANK {rank}] [SKIP] Image {global_idx+1}/{N} already processed.")
            continue

        print(f"\n[RANK {rank}] [INFO] Processing image {global_idx+1}/{N}...")
        image_start = MPI.Wtime()

        image_path = os.path.join(image_dir, fname)
        image = np.load(image_path)

        print(f"[RANK {rank}] [INFO] Running LBM in x-direction...")
        u_x, k_xx, k_xy = big_LBM(image, T, 0)

        print(f"[RANK {rank}] [INFO] Running LBM in y-direction...")
        u_y, k_yx, k_yy = big_LBM(image, T, 1)

        k = np.array([[k_xx, k_xy], [k_yx, k_yy]])

        save_result(global_idx, u_x, u_y, k)

        elapsed = MPI.Wtime() - image_start
        print(f"[RANK {rank}] [DONE] Image {global_idx+1} completed in {elapsed:.2f} seconds.")

    comm.Barrier()
    if rank == 0:
        print("[DONE] All ranks finished processing.")


if __name__ == '__main__':
    main_mpi_from_dir()