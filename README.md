# ICASSP24_RF_Challenge

An MIT challenge that requires developing an engine for signal separation of radio-frequency (RF) waveforms. At inference time, a superposition of a signal of interest (SOI) and an interfering signal will be fed to the engine, which should recover the SOI by performing a sophisticated interference calculation. For more details visit [https://rfchallenge.mit.edu/icassp24-single-channel/](https://rfchallenge.mit.edu/icassp24-single-channel/)

## Getting Started

It is highly recommended that a virtual environment setup is used. The following guide uses conda and pip (conda for python version and pip for dependencies)

```bash
conda create -n rfchallenge python=3.11
conda activate rfchallenge
conda install -c conda-forge llvm-tools llvmdev
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade ipykernel jupyterlab
python -m pip install sionna
python -m pip install tensorflow-cpu==2.15.1
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
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
python wave_net_inference.py --model_path=torch_models/dataset_qpsk_commsignal2_mixture_wavenet/weights-206000.pt --soi_type=QPSK --batch_size=4 --interference_dir_path=rf_datasets/train_test_set_unmixed/dataset/testset1_frame
```

## Project Files

The project files are arranged as follows
```bash
.
|-- wave_net_inference.py
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
|   |-- generate_competition_eval_mixture.py
|   `-- generate_train_mixture.py
|-- eval_utils
|   |-- __init__.py
|   |-- competition_eval_helpers.py
|   `-- postprocessing_helpers.py
|-- models
|   |-- UNet.py
|   |-- __init__.py
|-- notebooks
|   |-- RFC_Default_Torch_WaveNet_inference.ipynb
|   `-- train_and_eval_basic_UNet.ipynb
|-- requirements.txt
|-- src
|   |-- DefaultTorchWaveNet.py
|   |-- __init__.py
|   |-- config_torchwavenet.py
|   `-- configs
|       `-- wavenet.yaml
`-- training_utils
    |-- __init__.py
    |-- train_helpers.py
    `-- validation_helpers.py
```
| Directory/File | Description |
| --- | --- |
| `comm_utils` | utils for digital communication techniques such as modulation, pulse shaping, demodulation and so on |
| `data_manipulation_utils` | utils for ... |
| `dataset_utils` | utils for generating and loading training datasets |
| `training_utils` | utils for training a pytorch model |
| `eval_utils` | utils for evaluating models |
| `src.Default_Torch_WaveNet.py` | Baseline model design |
| `Default_Torch_WaveNet_inference.py` | a script to load baseline weights and perform inference on test dataset (Test1Mixture) |
