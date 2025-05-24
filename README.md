# MSSPlace: Multi-Sensor Place Recognition with Visual and Text Semantics

This repository contains the code for the paper "MSSPlace: Multi-Sensor Place Recognition with Visual and Text Semantics".

## Installation

Initialize submodules and build the Docker environment:

```bash
git submodule update --init --recursive
bash docker/build.sh
bash docker/start.sh [DATASETS_DIR]  # DATASETS_DIR will be mounted at /home/docker_mssplace/Datasets
bash docker/into.sh
```

## Quick Start

Evaluate pre-trained models on Oxford RobotCar or NCLT datasets:

```bash
# Download checkpoints and datasets first (see sections below)
python evaluate_checkpoints.py --dataset oxford --model mssplace-li
python evaluate_checkpoints.py --dataset nclt --model mssplace-list --verbose
```

## Evaluation

### Model Variants

| Model | Modalities | Description |
|-------|------------|-------------|
| `mssplace-li` | LiDAR + Images | Basic multimodal |
| `mssplace-lis` | LiDAR + Images + Semantic | Adds semantic segmentation |
| `mssplace-lit` | LiDAR + Images + Text | Adds text descriptions |
| `mssplace-list` | LiDAR + Images + Semantic + Text | Complete multimodal |

### Pre-trained Checkpoints

⚠️ **Work in Progress**: Checkpoint download links will be updated soon. Please check back later for access to pre-trained models.

### Datasets

⚠️ **Work in Progress**: Preprocessed datasets will be made publicly available for download soon. Please check back later for dataset access.

### Directory Structure

```
/home/docker_mssplace/
├── MSSPlace/                    # This repository
│   ├── evaluate_checkpoints.py
│   └── checkpoints/             # Downloaded checkpoints
└── Datasets/                    # Dataset directory (configurable with --datasets-dir)
    ├── pnvlad_oxford_robotcar/
    └── NCLT_preprocessed/
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | *Required* | `oxford` or `nclt` |
| `--model` | *Required* | Model variant (see table above) |
| `--datasets-dir` | `/home/docker_mssplace/Datasets` | Path to datasets |
| `--batch-size` | `32` | Batch size |
| `--verbose` | `False` | Detailed logging |

## Training (Optional)

⚠️ **Work in Progress**: Training documentation and scripts will be updated soon. Please check back later for training instructions.

## Troubleshooting

- **Missing checkpoints**: Download all `.pth` files to `checkpoints/`
- **Dataset errors**: Verify directory structure matches expected format
- **CUDA memory**: Reduce `--batch-size` if out-of-memory
- **Dependencies**: Use provided Docker environment
