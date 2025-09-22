#!/bin/bash
#SBATCH --job-name=convnextsmall_16000_diffnum
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

python main.py "configs/split_jobs/convnextsmall_16000_metrics_diff_num_datapoints.yaml"
