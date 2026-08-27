# AeroWF: Geometric-Aware Spectral-Temporal Dual-Stream Learning for Airport Weather Forecasting

## Overview

AeroWF is a novel geometric-aware spectral-temporal dual-stream learning framework designed for accurate and reliable aerodrome weather forecasting. The framework addresses the challenges of multi-scale dynamics and structural heterogeneity inherent in airport meteorological data, which involves variable numbers of runways, high-frequency sensor streams, and low-frequency environmental contexts.

## Key Contributions

1. **Macroscopic Context Injection**: A module that explicitly grounds transient, micro-scale sensor observations within slow-varying, macro-scale meteorological reports (METAR), ensuring physical consistency across temporal scales.

2. **Spectral-Temporal Dual-Stream Encoder**: Disentangles transient temporal patterns and global spectral periodicities through:
   - A transformer-based temporal branch for capturing temporal dynamics
   - A complex-valued frequency-domain branch for capturing spectral periodicities

3. **Structure-Adaptive Hierarchical Aggregator**: Unifies variable-cardinality runway observations through bidirectional interaction, without relying on predefined topological assumptions.

4. **Geometric-Consistent Pre-training**: Combines hybrid masked reconstruction with intrinsic metric alignment, ensuring learned representations preserve both local fidelity and the underlying manifold structure of weather evolution.

## Model Architecture

AeroWF comprises four main components:

1. **Temporal Branch**: Transformer-based encoder capturing short-term dynamics in sensor observations
2. **Spectral Branch**: Complex-valued FFT-based encoder (FreTS) capturing long-term periodicities
3. **Context Injection**: METAR integration module for cross-scale consistency
4. **Hierarchical Aggregator**: Structure-adaptive GNN unifying variable-cardinality runway data

The dual-stream design allows the model to learn disentangled representations of transient and periodic patterns, which are then fused for downstream tasks.

## Dataset

We evaluate AeroWF on a curated real-world multi-airport dataset:

| Dataset | Samples | Runways | Features | Airports |
|---------|---------|---------|----------|----------|
| Processed Data | 248K | 2-4 | 11 | ZBAA, ZBAD, ZSPD, ZSSS |

**Data Format**:
- **Temporal Data**: `(batch_size, num_runways, time_steps, channels)`
- **METAR Data** (optional): Categorical and continuous meteorological variables
- **Processed Location**: `processed/` directory with `*.npy` files


## Project Structure

```
AeroWF/
├── main.py                                 # Entry point
├── README.md                               # This file
├── models/
│   ├── model_factory.py                   # Model factory
│   ├── optimizers.py                      # Optimizer definitions
│   ├── unified_runner.py                  # Training pipeline
│   └── AirFM/                             # Core AeroWF implementations
│       ├── __init__.py
│       ├── unified_model.py               # Main AeroWF model
│       ├── unified_trainer.py             # Trainer class
│       ├── HierarchicalGNN.py             # Hierarchical aggregator
│       ├── Attention.py                   # Attention modules
│       ├── fft_filter.py                  # FFT utilities
│       ├── loss.py                        # Loss functions
│       ├── masked.py                      # Masking utilities
│       ├── physics_distance.py            # Distance metrics
│       ├── soft_dtw_cuda.py              # Soft-DTW implementation
│       ├── encoders/
│       │   ├── __init__.py
│       │   ├── exogenous_encoder.py      # METAR encoder
│       │   ├── frets_encoder.py          # Spectral branch
│       │   ├── positional_encoding.py    # Positional embeddings
│       │   └── transformer_encoder.py    # Temporal branch
│       └── fusion/
│           ├── __init__.py
│           └── dual_stream_fusion.py     # Fusion strategies
├── processed/                             # Data directory
│   ├── global_weather_config.json        # Metadata
│   ├── ZBAD_train.npy
│   ├── ZSPD_train.npy
│   ├── ZSSS_train.npy
│   └── ZBAA_train.npy
└── utils/
    ├── args.py                           # Argument parser
    ├── analysis.py                       # Analysis utilities
    └── utils.py                          # Helper functions
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd AeroWF
```

2. **Install dependencies** (Python 3.8+)
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
```

## Quick Start

### 1. Basic Training

```python
from models.unified_runner import unified_train
from utils.args import parse_args

# Configure model and training
config = {
    'Model_Type': ['AeroWF'],
    'task_type': 'unified_pretrain',  # Unified pre-training
    'lambda_recon': 1.0,               # Reconstruction loss weight
    'lambda_contrast': 0.5,            # Contrastive loss weight
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-3,
}

# Run training
best_metrics, all_metrics = unified_train(config, data)
```

### 2. Command Line

```bash
# Unified pre-training (recommended)
python main.py --task_type unified_pretrain --lambda_recon 1.0 --lambda_contrast 0.5

# Masked reconstruction only
python main.py --task_type masked_recon

# Contrastive learning only
python main.py --task_type pretrain
```

### 3. Pre-training Tasks

AeroWF supports three pre-training strategies:

| Task | Description | Use Case |
|------|-------------|----------|
| `pretrain` | Contrastive learning via distance metrics (DTW + FFT) | Limited data |
| `masked_recon` | Hybrid masking (random + causal) with reconstruction | General |
| `unified_pretrain` | Joint optimization of reconstruction + contrastive learning | Large datasets ✓ |

### 4. Using with METAR Data (Optional)

```python
# Configure exogenous variables
config['exo_config'] = {
    'categorical': {
        'weather_code_id': {'vocab_size': 25},
        'sky_condition': {'vocab_size': 6},
    },
    'continuous': ['visibility', 'cloud_height']
}

# During training, provide METAR data
exo_cat = {...}  # Categorical variables
exo_cont = {...}  # Continuous variables
best_metrics = unified_train(config, data, exo_categorical=exo_cat, exo_continuous=exo_cont)
```

## Performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank the four major international airport hubs (ZBAA, ZBAD, ZSPD, ZSSS) for providing the meteorological data used in this research.
