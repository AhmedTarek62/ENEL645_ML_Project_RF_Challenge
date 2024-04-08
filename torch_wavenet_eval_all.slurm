#!/bin/bash
#==============================================
#SBATCH --partition=gpu-a100
#SBATCH --time=4:00:00
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

# =======================================================================================================
# QPSK signal
# =======================================================================================================
# for QPSK + CommSignal2
echo 'checkpoints/Final_QPSK_UNet_model_epoch_15_val_loss_0.0084.pt
checkpoints/KUTII_WaveNet_QPSK-21_0.503670459613204.pt' | python all_models_eval_script.py --num_models=2 --baseline_model_path=checkpoints/torchmodels/dataset_qpsk_commsignal2_mixture_wavenet/weights-206000.pt --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame

# for QPSK + CommSignal3
echo 'checkpoints/Final_QPSK_UNet_model_epoch_15_val_loss_0.0084.pt
checkpoints/KUTII_WaveNet_QPSK-21_0.503670459613204.pt' | python all_models_eval_script.py --num_models=2 --baseline_model_path=checkpoints/torchmodels/dataset_qpsk_commsignal3_mixture_wavenet/weights-206000.pt --interference_type=CommSignal3 --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame

# for QPSK + CommSignal5G1
echo 'checkpoints/Final_QPSK_UNet_model_epoch_15_val_loss_0.0084.pt
checkpoints/KUTII_WaveNet_QPSK-21_0.503670459613204.pt' | python all_models_eval_script.py --num_models=2 --baseline_model_path=checkpoints/torchmodels/dataset_qpsk_commsignal5g1_mixture_wavenet/weights-276000.pt --interference_type=CommSignal5G1 --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame

# for QPSK + EMISignal1
echo 'checkpoints/Final_QPSK_UNet_model_epoch_15_val_loss_0.0084.pt
checkpoints/KUTII_WaveNet_QPSK-21_0.503670459613204.pt' | python all_models_eval_script.py --num_models=2 --baseline_model_path=checkpoints/torchmodels/dataset_qpsk_emisignal1_mixture_wavenet/weights-278000.pt --interference_type=EMISignal1 --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame

# =======================================================================================================
# QPSK_OFDM signal
# =======================================================================================================
# for QPSP_OFDM + CommSignal2
echo 'checkpoints/Final_OFDM_UNet_model_epoch_14_val_loss_0.0106.pt
checkpoints/KUTII_WaveNet_QPSK_OFDM-15_0.5915907843410969.pt' | python all_models_eval_script.py --num_models=2 --baseline_model_path=checkpoints/torchmodels/dataset_ofdmqpsk_commsignal2_mixture_wavenet/weights-204000.pt --soi_type=QPSK_OFDM --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame

# for QPSP_OFDM + CommSignal3
echo 'checkpoints/Final_OFDM_UNet_model_epoch_14_val_loss_0.0106.pt
checkpoints/KUTII_WaveNet_QPSK_OFDM-15_0.5915907843410969.pt' | python all_models_eval_script.py --num_models=2 --baseline_model_path=checkpoints/torchmodels/dataset_ofdmqpsk_commsignal3_mixture_wavenet/weights-318000.pt --interference_type=CommSignal3 --soi_type=QPSK_OFDM --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame

# for QPSP_OFDM + CommSignal5G1
echo 'checkpoints/Final_OFDM_UNet_model_epoch_14_val_loss_0.0106.pt
checkpoints/KUTII_WaveNet_QPSK_OFDM-15_0.5915907843410969.pt' | python all_models_eval_script.py --num_models=2 --baseline_model_path=checkpoints/torchmodels/dataset_ofdmqpsk_commsignal5g1_mixture_wavenet/weights-270000.pt --interference_type=CommSignal5G1 --soi_type=QPSK_OFDM --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame

# for QPSP_OFDM + EMISignal1
echo 'checkpoints/Final_OFDM_UNet_model_epoch_14_val_loss_0.0106.pt
checkpoints/KUTII_WaveNet_QPSK_OFDM-15_0.5915907843410969.pt' | python all_models_eval_script.py --num_models=2 --baseline_model_path=checkpoints/torchmodels/dataset_ofdmqpsk_emisignal1_mixture_wavenet/weights-208000.pt --interference_type=EMISignal1 --soi_type=QPSK_OFDM --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame
