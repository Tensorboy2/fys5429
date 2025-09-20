#!/bin/bash
#SBATCH --job-name=convnextsmall-dataaug
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py "configs/dataaugmentation_test.yaml"
