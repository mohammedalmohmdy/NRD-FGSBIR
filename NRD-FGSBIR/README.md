# Neural Retrieval Dynamics (NRD)

 **Neural Retrieval Dynamics: Learning Continuous Cross-Modal Trajectories for Fine-Grained Sketch-Based Image Retrieval**.

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
  
  📂 Datasets linkes

- **ShoeV2 / ChairV2**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/1frltfiEd9ymnODZFHYrbg741kfys1rq1/view)

- **Sketchy**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view)

- **TU-Berlin**  
  [TU-Berlin Official Website](https://www.tu-berlin.de/)  
  [Google Drive Download](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view)

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

Citation: If you use this code, please cite:

title = {Neural Retrieval Dynamics: Learning Continuous Cross-Modal Trajectories for Fine-Grained Sketch-Based Image Retrieval)},

author = {Mohammed A. S. Al-Mohamadi and Prabhakar C. J.},

journal = {...........}, year = {2026} }

Contact: almohmdy30@gmail.com GitHub: https://github.com/mohammedalmohmdy

