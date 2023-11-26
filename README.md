# MSSPlace: Multi-Sensor Place Recognition with Visual and Text Semantics

This repository contains the code for the paper "MSSPlace: Multi-Sensor Place Recognition with Visual and Text Semantics".

# Installation

We highly recommend using provided Dockerfile ([docker/Dockerfile.cuda](./docker/Dockerfile.cuda)) to install all the dependencies. You can build the docker image using the following command:

```bash
# from the root directory of the repository
bash docker/build.sh
```

To start the container, run the following command. `DATASETS_DIR` is the path to the directory containing the datasets. This directory will be mounted inside the container at `/home/docker_mssplace/Datasets`.

```bash
bash docker/start.sh [DATASETS_DIR]
```

To enter the container, run the following command:

```bash
bash docker/into.sh
```

# Training

We use Hydra for managing configurations. You can find the configuration files in the [`configs/`](./configs/) directory. The default configuration is [`configs/train_unimodal.yaml`](./configs/train_unimodal.yaml). You can override any of the parameters in the configuration file from the command line. All neccessary config files are provided.

You can use the [`train_unimodal.py`](./train_unimodal.py) script for training. For example, to train lidar-only model on the NCLT dataset, run the following command:

```bash
python train_unimodal.py dataset=nclt/lidar dataset.dataset_root=/home/docker_mssplace/Datasets/NCLT_preprocessed model=lidar exp_name=nclt_lidar_exp
```

# Testing

## Single-Modality

To test a single-modality model, you can use the [`test.py`](./test.py) script. For example, to test the lidar-only model trained on the NCLT dataset, run the following command:

```bash
python test.py --checkpoint outputs/nclt_lidar_exp_2023-11-16-12-25-41/checkpoints/best.pth --model_config configs/model/lidar.yaml --dataset_config configs/dataset/nclt/lidar.yaml --dataset_dir /home/docker_mssplace/Datasets/OpenPlaceRecognition/NCLT_preprocessed --batch_size 32 --device cuda
```

### Parsing results from WandB

We use [WandB](https://wandb.ai/) for logging the training and testing results. You can use the [`parse_wandb.ipynb`](./parse_wandb.ipynb) notebook to parse the results from WandB.

## Multi-Modality

As we do not train multi-modality models end-to-end, we use the following notebook to stack multiple single-modality models and test them: [`test_multimodal.ipynb`](./test_multimodal.ipynb).

Note that in the above notebook we use the checkpoints that was downloaded from WandB, saved together with corresponding config files and stored in the [`checkpoints/`](./checkpoints/) directory. If you want to use our checkpoints, you can download them from [Google Drive](https://drive.google.com/drive/folders/1KmIfUXtfkU1Qs4wDM-MAC86PClkn_cwB?usp=drive_link).
