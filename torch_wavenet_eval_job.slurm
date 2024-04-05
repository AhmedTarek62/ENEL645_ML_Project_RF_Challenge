#!/bin/bash
#==============================================
#SBATCH --partition=gpu-a100
#SBATCH --time=0:40:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sigsep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ogechukwukanu@gmail.com
#==============================================

source ~/software/init-conda

conda activate rfchallenge

python KUTII_WaveNet_eval_script.py --soi_type=QPSK --batch_size=8 --ckpt_path=checkpoints/KUTII_WaveNet_QPSK-32_0.16428813977167012.pt
