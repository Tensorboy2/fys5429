#!/bin/bash
#SBATCH --job-name=convnextsmall_18000_diffnum
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py "configs/split_jobs/convnextsmall_18000_metrics_diff_num_datapoints.yaml"
