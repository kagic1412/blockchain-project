# Elliptic++ AML Detection

A comparative study of Graph Neural Networks (GNN) and tree-based models for Anti-Money Laundering (AML) detection in blockchain transactions.

## Overview

This project explores the effectiveness of GraphSAGE neural networks versus LightGBM tree models in detecting illicit transactions on the Elliptic++ dataset. The study analyzes why GNN underperforms tree models in this specific task and provides hyperparameter optimization using Optuna.

## Dataset

**Note**: The data files are not included in this repository due to size constraints. Please download from the official source.

### Download

1. Download from [Elliptic++ dataset repository](https://github.com/git-disl/EllipticPlusPlus/tree/main/Transactions%20Dataset)
2. Create the following directory structure in the project root:
   ```
   data/
   ├── raw/
   ├── processed/
   └── output/
   ```
3. Place the downloaded files into `data/raw/` with these names:
   - `txs_classes.csv`
   - `txs_edgelist.csv`
   - `txs_features.csv`

### Dataset Overview

The [Elliptic++](https://www.elliptic.co/) dataset contains 203,769 Bitcoin transactions with:
- **166 features** per transaction (local and aggregate features)
- **Graph structure** showing transaction flows (234,355 edges)
- **Temporal split**: Training (timestep ≤ 34), Testing (timestep > 34)
- **Class distribution**: 1=Illicit (4,545), 2=Licit (42,019), 3=Unknown (157,205)

## Project Structure

```
blockchain/
├── data/
│   ├── raw/                      # Raw data files
│   │   ├── txs_classes.csv       # Transaction labels
│   │   ├── txs_edgelist.csv     # Transaction graph edges
│   │   └── txs_features.csv      # Transaction features
│   ├── processed/                # Processed data
│   │   ├── txs_merge_with_network.csv  # Final dataset
│   │   └── ...
│   └── output/                   # Model outputs
│
├── scripts/
│   ├── data_processing/          # Data processing
│   │   ├── extract_network_features.py
│   │   ├── merge_data.py
│   │   └── feature_analysis.py
│   └── models/                   # Model training
│       ├── baseline_model.py     # LightGBM baseline
│       ├── graphsage_aml.py      # GraphSAGE GNN
│       ├── graphsage_optuna.py    # Hyperparameter tuning
│       └── summary.py             # Model comparison
│
├── analysis/                      # Analysis outputs
├── docs/                         # Documentation
├── results/                       # Experiment results
├── README.md                     # This file
└── .gitignore
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/elliptic-aml.git
cd elliptic-aml

# Create conda environment
conda create -n pytorch_env python=3.9
conda activate pytorch_env

# Install PyTorch
pip install torch torchvision torchaudio

# Install PyTorch Geometric (requires PyTorch to be installed first)
pip install pyg-nightly torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

```bash
# Merge features with labels
python scripts/data_processing/merge_data.py

# Extract network features
python scripts/data_processing/extract_network_features.py

# Merge network features
python scripts/data_processing/merge_network_features.py
```

### 2. Run Models

```bash
# LightGBM Baseline
python scripts/models/baseline_model.py

# GraphSAGE GNN (F1-optimized parameters)
python scripts/models/graphsage_aml.py

# GNN with Optuna hyperparameter search
python scripts/models/graphsage_optuna.py
```

### 3. Model Comparison

```bash
python scripts/summary.py
```

## Performance Results

| Model | Recall | Precision | F1-Score |
|-------|--------|-----------|----------|
| LightGBM (189 features) | **0.73** | **0.90** | **0.81** |
| GraphSAGE (50 features, F1-optimized) | 0.62 | 0.65 | 0.63 |
| GraphSAGE (50 features, Recall-optimized) | 0.87 | 0.16 | 0.27 |

### Key Finding

Tree-based models significantly outperform GNN on this task. Key issues identified:

1. **Unknown node interference**: 77% of nodes have unknown labels, contaminating message passing
2. **Temporal distribution shift**: Graph structure evolves over time
3. **Training strategy**: No validation set, no early stopping

## Model Configurations

### LightGBM Baseline
```python
LGBMClassifier(
    is_unbalance=True,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    num_leaves=31
)
```

### GraphSAGE (F1-optimized)
```python
GraphSAGE(
    in_channels=50,      # Top 50 features
    hidden_channels=64,
    out_channels=2,
    dropout=0.476,
    aggr='max'
)
# weight_1 = 4.56
# lr = 0.039
# epochs = 250
```

## Documentation

- English: See [`README.md`](README.md)
- 中文文档: See [`README_CN.md`](README_CN.md)

## License

This project is for research purposes. The Elliptic++ dataset is provided by Elliptic under their own license.

## Acknowledgments

- [Elliptic++](https://www.elliptic.co/) dataset
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
