import numpy as np
import time
import torch
import os
from mpi4py import MPI
path = os.path.dirname(__file__)
from simulation.lbm import big_LBM



def main():
    """
    Serial version with timing and logging info
    """
    print("[INFO] Loading images...")
    start_total = time.time()

    images = np.load(os.path.join(path, "data_generation/images_filled.npy"))
    N, W, H = images.shape
    T = 10_000

    print(f"[INFO] Loaded {N} images of size {W}x{H}")
    u_x_all = np.zeros((N, W, H, 2))
    u_y_all = np.zeros((N, W, H, 2))
    k_all = np.zeros((N, 2, 2))


    for i in range(N):
        print(f"\n[INFO] Processing image {i+1}/{N}...")
        image_start = time.time()

        print("[INFO] Running LBM in x-direction...")
        u_x, k_xx, k_xy = big_LBM(images[i], T, 0)

        print("[INFO] Running LBM in y-direction...")
        u_y, k_yx, k_yy = big_LBM(images[i], T, 1)

        u_x_all[i] = u_x
        u_y_all[i] = u_y
        k_all[i] = np.array([[k_xx, k_xy], [k_yx, k_yy]])

        elapsed = time.time() - image_start
        print(f"[DONE] Image {i+1} completed in {elapsed:.2f} seconds.")

    print("\n[INFO] Saving results to disk...")
    np.save(os.path.join(path, "u_x.npy"), u_x_all)
    np.save(os.path.join(path, "u_y.npy"), u_y_all)
    np.save(os.path.join(path, "k.npy"), k_all)

    total_time = time.time() - start_total
    print(f"[DONE] All images processed in {total_time:.2f} seconds.")


out_dir = os.path.join(path, "output_checkpoints")
os.makedirs(out_dir, exist_ok=True)

def image_already_done(index):
    return (
        os.path.exists(os.path.join(out_dir, f"u_x_{index:05d}.npy")) and
        os.path.exists(os.path.join(out_dir, f"u_y_{index:05d}.npy")) and
        os.path.exists(os.path.join(out_dir, f"k_{index:05d}.npy"))
    )

def save_result(index, u_x, u_y, k):
    np.save(os.path.join(out_dir, f"u_x_{index:05d}.npy"), u_x)
    np.save(os.path.join(out_dir, f"u_y_{index:05d}.npy"), u_y)
    np.save(os.path.join(out_dir, f"k_{index:05d}.npy"), k)

def main_2():
    """
    Serial version with checkpointing and per-image saving
    """
    print("[INFO] Loading images...")
    start_total = time.time()

    images = np.load(os.path.join(path, "data_generation/images_filled.npy"))
    N, W, H = images.shape
    T = 10_000

    print(f"[INFO] Loaded {N} images of size {W}x{H}")
    
    for i in range(N):
        if image_already_done(i):
            print(f"[SKIP] Image {i+1}/{N} already processed.")
            continue

        print(f"\n[INFO] Processing image {i+1}/{N}...")
        image_start = time.time()

        print("[INFO] Running LBM in x-direction...")
        u_x, k_xx, k_xy = big_LBM(images[i], T, 0)

        print("[INFO] Running LBM in y-direction...")
        u_y, k_yx, k_yy = big_LBM(images[i], T, 1)

        k = np.array([[k_xx, k_xy], [k_yx, k_yy]])

        save_result(i, u_x, u_y, k)

        elapsed = time.time() - image_start
        print(f"[DONE] Image {i+1} completed in {elapsed:.2f} seconds.")

    total_time = time.time() - start_total
    print(f"\n[DONE] All available images processed in {total_time:.2f} seconds.")

def main_mpi_from_dir():
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

    # Broadcast total number of images
    N = comm.bcast(N if rank == 0 else None, root=0)

    # Ensure even distribution
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

    # Scatter filenames
    local_filenames = comm.scatter(filenames_chunks, root=0)

    for i, fname in enumerate(local_filenames):
        global_idx = local_start + i

        if image_already_done(global_idx):
            print(f"[RANK {rank}] [SKIP] Image {global_idx+1}/{N} already processed.")
            continue

        print(f"\n[RANK {rank}] [INFO] Processing image {global_idx+1}/{N}...")
        # image_start = time.time()
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
    # main_2()
    main_mpi_from_dir()