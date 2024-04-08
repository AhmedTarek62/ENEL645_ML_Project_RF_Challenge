#!/bin/bash
#==============================================
#SBATCH --partition=gpu-a100
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --job-name=sigsep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ogechukwukanu@gmail.com
#==============================================

source ~/software/init-conda

conda activate rfchallenge

python KUTII_WaveNet_trainer_script.py --epoch=50 --batch_size=8 --soi_type=QPSK --loss=softdemod --dataset_path=rf_datasets/train_set_mixed/datasets/QPSK_20240406_141746
