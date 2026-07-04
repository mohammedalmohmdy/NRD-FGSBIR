# Neural Retrieval Dynamics (NRD)

A complete, research-grade PyTorch implementation of **Neural Retrieval Dynamics: Learning Continuous Cross-Modal Trajectories for Fine-Grained Sketch-Based Image Retrieval**.

## Overview

This project implements the NRD framework for Fine-Grained Sketch-Based Image Retrieval (FG-SBIR) with the following key components:

- **Dual Representation Encoder**: Separate encoders for sketches and images with shared embedding space
- **Continuous Cross-Modal Trajectory Learning**: Learnable trajectories bridging modalities
- **Dynamic Retrieval Interaction Module**: Cross-modal attention for enhanced interaction
- **Progressive Similarity Optimization**: Multi-stage similarity computation
- **Complete Training Pipeline**: With AMP, multi-GPU support, checkpointing, and logging

## Project Structure

```
NRD/
├── datasets/              # Dataset loaders
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── sketchy_dataset.py
│   ├── shoev2_dataset.py
│   └── chairv2_dataset.py
├── models/                # Model components
│   ├── __init__.py
│   ├── encoder.py        # Dual Representation Encoder
│   ├── trajectory.py     # Cross-Modal Trajectory Learning
│   ├── interaction.py    # Dynamic Interaction Module
│   ├── similarity.py     # Progressive Similarity Optimization
│   ├── loss.py          # All loss functions
│   └── nrd.py           # Main NRD model
├── trainer/              # Training and validation
│   ├── __init__.py
│   ├── train.py         # Training loop
│   ├── validate.py      # Validation metrics
│   └── engine.py        # Training engine
├── utils/                # Utilities
│   ├── __init__.py
│   ├── config.py        # Configuration management
│   ├── seed.py          # Reproducibility
│   ├── logger.py        # Logging
│   └── metrics.py       # Metric computation
├── configs/              # Configuration files
│   ├── default.yaml
│   ├── sketchy.yaml
│   ├── shoev2.yaml
│   └── chairv2.yaml
├── inference.py          # Inference module
├── train.py             # Main training script
├── test.py              # Main testing script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU support)

### Setup

1. Clone the repository:
```bash
cd NRD
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch; print(torch.__version__)"
```

## Dataset Preparation

### Supported Datasets

- **Sketchy Extended**: Extended version of Sketchy dataset
- **ShoeV2**: Shoe dataset for sketch-based retrieval
- **ChairV2**: Chair dataset for sketch-based retrieval

### Dataset Structure

Organize your data as follows:

```
data/
├── sketchy_extended/
│   ├── sketches/
│   │   ├── category1/
│   │   │   ├── sketch1.png
│   │   │   └── ...
│   │   └── ...
│   ├── photos/
│   │   ├── category1/
│   │   │   ├── photo1.jpg
│   │   │   └── ...
│   │   └── ...
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── shoev2/
│   ├── sketches/
│   ├── photos/
│   └── splits/
└── chairv2/
    ├── sketches/
    ├── photos/
    └── splits/
```

### Split File Format

Each split file should contain one entry per line:
```
category/sketch_name.jpg category/photo_name.jpg label
```

## Configuration

Configuration files are in YAML format located in the `configs/` directory.

### Key Configuration Sections

- **model**: Model architecture parameters
- **training**: Training hyperparameters
- **validation**: Evaluation settings
- **checkpoint**: Checkpoint saving
- **logging**: TensorBoard and console logging
- **hardware**: Device and precision settings
- **augmentation**: Data augmentation parameters

### Example Configuration

```yaml
model:
  sketch_encoder:
    backbone: "resnet50"
    pretrained: true
    feature_dim: 2048
    dropout: 0.5

training:
  dataset: "sketchy_extended"
  data_root: "./data/sketchy_extended"
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.0001
```

## Training

### Basic Training

Train on Sketchy Extended dataset:

```bash
python train.py --config configs/sketchy.yaml
```

### Training with Custom Parameters

```bash
python train.py \
    --config configs/default.yaml \
    --dataset sketchy_extended \
    --data_root ./data/sketchy_extended \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 0.0001 \
    --device cuda \
    --num_gpus 1
```

### Resume Training

```bash
python train.py \
    --config configs/sketchy.yaml \
    --resume ./checkpoints/checkpoint_epoch_50.pth
```

### Training Options

- `--config`: Path to configuration file (required)
- `--dataset`: Dataset name (sketchy_extended, shoev2, chairv2)
- `--data_root`: Data root directory
- `--batch_size`: Batch size
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--device`: Device (cuda/cpu)
- `--num_gpus`: Number of GPUs
- `--mixed_precision`: Enable automatic mixed precision
- `--resume`: Path to checkpoint to resume from
- `--seed`: Random seed for reproducibility

## Testing and Evaluation

### Basic Testing

```bash
python test.py \
    --config configs/sketchy.yaml \
    --checkpoint ./checkpoints/best_model.pth
```

### Testing on Specific Split

```bash
python test.py \
    --config configs/sketchy.yaml \
    --checkpoint ./checkpoints/best_model.pth \
    --split test
```

### Testing with Per-Class Metrics

```bash
python test.py \
    --config configs/sketchy.yaml \
    --checkpoint ./checkpoints/best_model.pth \
    --per_class \
    --output_dir ./results
```

### Testing Options

- `--config`: Path to configuration file (required)
- `--checkpoint`: Path to model checkpoint (required)
- `--dataset`: Dataset name
- `--data_root`: Data root directory
- `--split`: Dataset split (val/test)
- `--batch_size`: Batch size
- `--device`: Device (cuda/cpu)
- `--output_dir`: Directory to save results
- `--save_features`: Save extracted features
- `--per_class`: Compute per-class metrics
- `--seed`: Random seed

## Inference

### Running Inference

```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pth \
    --config configs/sketchy.yaml \
    --query ./query_sketch.png \
    --gallery ./data/gallery \
    --top_k 10
```

### Programmatic Inference

```python
from inference import NRDInference
from utils import load_config

# Load configuration
config = load_config('configs/sketchy.yaml')

# Load inference module
inference = NRDInference.from_checkpoint(
    checkpoint_path='./checkpoints/best_model.pth',
    config_path='configs/sketchy.yaml'
)

# Load and encode sketch
sketch = inference.load_image('query_sketch.png')
sketch_features = inference.encode_sketch(sketch)

# Retrieve top-K images
top_k_paths, top_k_scores = inference.retrieve(
    sketch=sketch,
    gallery_features=gallery_features,
    gallery_paths=gallery_paths,
    top_k=10
)
```

## Model Architecture

### Dual Representation Encoder

The encoder consists of two separate CNN backbones (ResNet-50 by default) for sketches and images, each with a projection head to map to a shared embedding space.

**Equation:**
```
f_s = E_s(x_s)
f_i = E_i(x_i)
```

### Continuous Cross-Modal Trajectory Learning

Learns continuous trajectories between modalities using a parameterized evolution function.

**Equation:**
```
z(t) = (1 - t) * f_s + t * f_i + Δ(t)
```

### Dynamic Retrieval Interaction Module

Multi-head cross-modal attention enables interaction between modalities.

**Equation:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

### Progressive Similarity Optimization

Computes similarity at multiple trajectory stages with learnable weights.

**Equation:**
```
S_progressive = Σ_{k=1}^{K} α_k * S_k
```

## Loss Functions

### Triplet Ranking Loss

Encourages anchor-positive pairs to be closer than anchor-negative pairs.

**Equation:**
```
L_triplet = max(0, d(a, p) - d(a, n) + margin)
```

### Trajectory Alignment Loss

Ensures bidirectional trajectories are aligned.

**Equation:**
```
L_align = Σ_{t=0}^{T} ||z_s→i(t) - z_i→s(1-t)||^2
```

### Smoothness Loss

Encourages smooth trajectory evolution.

**Equation:**
```
L_smooth = Σ_{t=1}^{T-1} ||z(t) - z(t-1)||^2
```

### L2 Regularization

Prevents overfitting by penalizing large weights.

**Equation:**
```
L_l2 = λ * Σ ||θ||^2
```

## Evaluation Metrics

The model is evaluated using standard retrieval metrics:

- **mAP@200**: Mean Average Precision at 200
- **Precision@K**: Precision at K (K=1, 5, 10)
- **Recall@K**: Recall at K (K=1, 5, 10)
- **Top-K Accuracy**: Top-K accuracy (K=1, 5, 10)

## Features

### Training Features

- **Automatic Mixed Precision (AMP)**: Faster training with reduced memory usage
- **Multi-GPU Support**: Data parallelism for multiple GPUs
- **Cosine Learning Rate Scheduler**: Smooth learning rate decay
- **Checkpoint Saving**: Automatic checkpoint saving with cleanup
- **Early Stopping**: Stop training when metric plateaus
- **TensorBoard Logging**: Real-time training visualization
- **Reproducibility**: Configurable random seeds

### Validation Features

- **Comprehensive Metrics**: mAP, Precision, Recall, Top-K
- **Per-Class Analysis**: Detailed per-class metrics
- **Feature Extraction**: Save extracted features for analysis

## Hyperparameters

### Default Hyperparameters

- **Backbone**: ResNet-50 (pretrained on ImageNet)
- **Feature Dimension**: 2048
- **Trajectory Steps**: 10
- **Hidden Dimension**: 512
- **Attention Heads**: 8
- **Interaction Layers**: 2
- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Triplet Margin**: 0.3
- **Loss Weights**:
  - Triplet Ranking: 1.0
  - Trajectory Alignment: 0.5
  - Smoothness: 0.1
  - L2 Regularization: 0.001

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size
- Enable mixed precision: `--mixed_precision`
- Use gradient accumulation

### Slow Training

- Enable mixed precision
- Increase number of data loading workers
- Use multiple GPUs

### Poor Performance

- Check data preprocessing
- Verify dataset splits
- Adjust learning rate
- Increase training epochs
- Try different backbone architectures

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nrd2024,
  title={Neural Retrieval Dynamics: Learning Continuous Cross-Modal Trajectories for Fine-Grained Sketch-Based Image Retrieval},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- The FG-SBIR research community for dataset and benchmark contributions

## Contact

For questions and issues, please open an issue on the GitHub repository.
