#!/bin/bash
#SBATCH --job-name=convnext-diffnum
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py "configs/diff_num_datapoints.yaml"
