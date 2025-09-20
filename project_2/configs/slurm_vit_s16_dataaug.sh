#!/bin/bash
#SBATCH --job-name=vit-s16-dataaug
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py "configs/vit_s16_dataaugmentation_test.yaml"
