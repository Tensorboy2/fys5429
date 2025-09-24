#!/bin/bash
#SBATCH --job-name=Speed_test_vit_b16_extralong
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py "configs/split_jobs/vit_b16_extralong.yaml"
