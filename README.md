# Simulation-VFPS

A simulation code for paper **VF-PS: How to Select Important Participants in Vertical Federated Learning, Efficiently and Securely?**

This project implements dynamic participant selection for Vertical Federated Learning (VFL) using mutual information estimation with group testing.

## Reference

- **Paper**: [VF-PS: How to Select Important Participants in Vertical Federated Learning, Efficiently and Securely?](https://arxiv.org/abs/2205.12731)
- **Original Repository**: [Dynamic-VFPS](https://github.com/r-gheda/Dynamic-VFPS)

## Versions

| Version | File | Description |
|---------|------|-------------|
| **Legacy** | `test.py` | Old version implementation modified from the reference repository |
| **New** | `test_gpu.py` | Updated version with CUDA/GPU support, compatible with newer PyTorch versions |

## Installation

```bash
# Create conda environment
conda env create -f environment_cuda.yml -y
conda activate vfps-gpu
```

## Usage

### Basic Usage

```bash
# Fashion-MNIST (default)
python test_gpu.py

# CIFAR-10
python test_gpu.py --dataset cifar-10

# Custom parameters
python test_gpu.py --epochs 50 --clients 10 --selected 6

# With encryption
python test_gpu.py --encryption tenseal
python test_gpu.py --encryption paillier

# Static MI mode (select clients once before training)
python test_gpu.py --mi-mode static --mi-ratio 0.111

# CIFAR-10 with custom settings
python test_gpu.py --dataset cifar-10 --epochs 100 --lr 0.01
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | fashion-mnist | Dataset: `fashion-mnist` or `cifar-10` |
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch-size` | 256 | Batch size |
| `--local-epochs` | 1 | Local iterations per batch |
| `--clients` | 10 | Number of total clients |
| `--selected` | 6 | Number of selected clients |
| `--n-tests` | 5 | Number of group tests |
| `--k-nn` | 3 | KNN k value for MI estimation |
| `--mi-mode` | dynamic | MI mode: `dynamic` or `static` |
| `--mi-ratio` | 0.111 | Data ratio for MI estimation (static mode) |
| `--encryption` | plaintext | Encryption: `plaintext`, `paillier`, `tenseal` |
| `--bandwidth` | 300 | Bandwidth in Mbps |

## Time Statistics

The implementation provides detailed time breakdown:

| Component | Description |
|-----------|-------------|
| **Train** | Forward + backward computation time |
| **Comm** | Communication time (plaintext activation + gradient) |
| **MI Compute** | MI estimation computation time |
| **MI Comm** | Encrypted data transmission time for MI estimation |

### Communication Flow

1. **Client Selection Phase**:
   - Clients send **encrypted** raw data to server
   - Server computes MI and selects participants

2. **Model Training Phase**:
   - Forward: Clients send **plaintext** activations to server
   - Backward: Server sends **plaintext** gradients to clients

## Project Structure

```
Simulation-VFPS/
├── test.py                          # Legacy version (PySyft-based)
├── test_gpu.py                      # New version (GPU support, recommended)
├── run.sh                           # Run script examples
├── requirements.txt                 # Pip dependencies
├── environment_cuda.yml             # Conda environment config
│
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── config.py                    # Training configuration
│   ├── splitnn.py                   # Split neural network core
│   ├── evaluation.py                # Model evaluation
│   │
│   ├── models/                      # Neural network models
│   │   ├── resnet.py                # ResNet18 implementation
│   │   └── split_resnet.py          # Split ResNet for VFL
│   │
│   ├── data/                        # Data processing
│   │   └── distributor.py           # Vertical data distribution
│   │
│   ├── communication/               # Communication module
│   │   └── estimator.py             # Communication time estimator
│   │
│   ├── transmission/                # Encryption implementations
│   │   ├── base.py                  # Base transmission class
│   │   ├── plaintext.py             # Plaintext transmission
│   │   ├── paillier/                # Paillier encryption
│   │   └── tenseal/                 # TenSEAL/CKKS encryption
│   │
│   └── utils/                       # Utility functions
│       ├── helpers.py               # Helper functions (digamma, args)
│       └── split_data.py            # Data split utilities
│
├── datasets/                        # Dataset storage
│   ├── mnist/                       # MNIST data
│   └── fashion_mnist/               # Fashion-MNIST data
│
└── README.md                        # This file
```

## Datasets

Supports two datasets:

| Dataset | Image Size | Channels | Classes | Training Samples |
|---------|-----------|----------|---------|------------------|
| Fashion-MNIST | 28x28 | 1 (grayscale) | 10 | 60,000 |
| CIFAR-10 | 32x32 | 3 (RGB) | 10 | 50,000 |

### Data Partition

Images are vertically partitioned by columns with **complete data distribution** (no data loss):

**Fashion-MNIST (28x28, 10 clients)**:
- First 8 clients: 3 columns each (28×3 = 84 pixels)
- Last 2 clients: 2 columns each (28×2 = 56 pixels)
- Total: 8×3 + 2×2 = 28 columns ✓

**CIFAR-10 (32x32, 10 clients)**:
- First 2 clients: 4 columns each (32×4 = 128 pixels)
- Last 8 clients: 3 columns each (32×3 = 96 pixels)
- Total: 2×4 + 8×3 = 32 columns ✓

## Model Architecture

- **Client Model**: ResNet18 (adapted for small images, supports variable input widths)
- **Server Model**: Fully connected layers for classification
- **Feature Dim**: 256 per client

## Time Statistics
