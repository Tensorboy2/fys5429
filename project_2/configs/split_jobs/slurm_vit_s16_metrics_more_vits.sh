#!/bin/bash
#SBATCH --job-name=vit_s16_more_vits
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

python main.py "configs/split_jobs/vit_s16_metrics_more_vits.yaml"
