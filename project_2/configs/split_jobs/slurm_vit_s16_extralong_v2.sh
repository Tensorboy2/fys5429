#!/bin/bash
#SBATCH --job-name=vit_s16_extralong
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py "configs/split_jobs/vit_s16_extralong_v2.yaml"
