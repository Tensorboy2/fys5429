#!/bin/bash
#SBATCH --job-name=vit_t16_extralong
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py "configs/split_jobs/vit_t16_extralong.yaml"
