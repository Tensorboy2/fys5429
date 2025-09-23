#!/bin/bash
#SBATCH --job-name=convnextsmall_hflip_vflip_rotate_dataaug
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py "configs/split_jobs/convnextsmall_hflip_vflip_rotate_metrics_dataaugmentation_test.yaml"
