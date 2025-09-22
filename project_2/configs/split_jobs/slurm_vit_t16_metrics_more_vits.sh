#!/bin/bash
#SBATCH --job-name=vit_t16_more_vits
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py "configs/split_jobs/vit_t16_metrics_more_vits.yaml"
