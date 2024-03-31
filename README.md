# ICASSP24_RF_Challenge

An MIT challenge that requires developing an engine for signal separation of radio-frequency (RF) waveforms. At inference time, a superposition of a signal of interest (SOI) and an interfering signal will be fed to the engine, which should recover the SOI by performing a sophisticated interference calculation. For more details visit [https://rfchallenge.mit.edu/icassp24-single-channel/](https://rfchallenge.mit.edu/icassp24-single-channel/)

## Getting Started

It is highly recommended that a virtual environment setup is used. The following guide uses conda and pip (conda for python version and pip for dependencies)

```bash
conda create -n rfchallenge python=3.11
conda activate rfchallenge
python -m pip install --upgrade pip
python -m pip install tensorflow[and-cuda]
python -m pip install torch torchvision torchaudio
python -m pip install sionna
python -m pip install -r requirments.txt
```

Download interference files
```bash
wget -O  dataset.zip "https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0"
mkdir -p rf_datasets/train_test_set_unmixed
unzip  dataset.zip -d rf_datasets/train_test_set_unmixed
rm dataset.zip
```

Test Default_Torch_WaveNet (baseline) model

```bash
# download model weights
wget -O  reference_models.zip "https://www.dropbox.com/scl/fi/890vztq67krephwyr0whb/reference_models.zip?rlkey=6yct3w8rx183f0l3ok2my6rej&dl=0"
unzip  reference_models.zip
rm reference_models.zip

# run inference model on QPSK signal and CommSignal2 interference
python Default_Torch_WaveNet_inference.py --model_path=torchmodels/dataset_qpsk_commsignal2_mixture_wavenet/weights-206000.pt --soi_type=QPSK --num_batches=50 --batch_size=200 --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame
```

## Project Files

The project files are arranged as follows
```bash
.
|-- Default_Torch_WaveNet.py
|-- Default_Torch_WaveNet_inference.py
|-- README.md
|-- comm_utils
|   |-- __init__.py
|   |-- ofdm_helper.py
|   |-- qpsk_helper.py
|   |-- rrc_helper.py
|   |-- sig_helper.py
|-- data_manipulation_utils
|   |-- __init__.py
|   |-- preprocessing_helpers.py
|   |-- visualization_helpers.py
|-- dataset_utils
|   |-- SigSepDataset.py
|   |-- __init__.py
|   |-- generate_train_mixture.py
|-- eval_utils
|   |-- __init__.py
|   |-- postprocessing_helpers.py
|-- notebooks
|-- requirements.txt
```
| Directory/File | Description |
| --- | --- |
| `comm_utils` | utils for digital communication techniques such as modulation, pulse shaping, demodulation and so on |
| `data_manipulation_utils` | utils for ... |
| `dataset_utils` | utils for generating and loading training datasets |
| `eval_utils` | utils for evaluating models |
| `Default_Torch_WaveNet.py` | Baseline model design |
| `Default_Torch_WaveNet_inference.py` | a script to load baseline weights and perform inference on test dataset (Test1Mixture) |
