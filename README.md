# inzva DLSG #10 - Homework 2: GNN Graph Classification

## Overview

This project implements a controlled empirical study comparing **Feature-Only** vs **Message Passing** graph neural network models for graph classification.

## Models

### Feature-Only (no message passing)
- **LinearBaseline**: Global pool → Linear classifier
- **MLPBaseline**: Global pool → MLP classifier  
- **DeepSets**: Per-node MLP → Global pool → MLP classifier

### Message Passing
- **GCN**: Graph Convolutional Network
- **GraphSAGE**: GraphSAGE
- **GAT**: Graph Attention Network
- **GIN**: Graph Isomorphism Network

## Datasets
- **PROTEINS** - Biological network classification (Accuracy)
- **IMDB-MULTI** - Movie collaboration networks (Accuracy)
- **REDDIT-BINARY** - Reddit thread classification (Accuracy)
- **ogbg-molhiv** - Molecular property prediction (ROC-AUC)

## Usage

### Google Colab (Recommended)
Open `GNN_Homework2.ipynb` in Google Colab and run all cells.

### CLI
```bash
pip install -r requirements.txt

# Single experiment
python main.py --dataset PROTEINS --model gcn --pool mean --hidden 64 --layers 4 --seeds 0 1 2

# All experiments
bash run_experiments.sh
```

## Configuration
- Hidden dimension: 64
- Layers: 4
- Dropout: 0.5
- Learning rate: 0.001
- Batch size: 64
- Epochs: 100 (early stopping, patience=20)
- Seeds: 0, 1, 2

## Project Structure
```
├── GNN_Homework2.ipynb   # Complete Colab notebook
├── main.py               # CLI entry point
├── models.py             # All 7 model definitions
├── data.py               # Dataset loading & preprocessing
├── train.py              # Training loop & evaluation
├── utils/
│   ├── __init__.py
│   └── utilities.py      # EarlyStopping, logging, helpers
├── run_experiments.sh     # Batch experiment runner
└── requirements.txt      # Dependencies
```
