#!/bin/bash
# =============================================================================
# Run all experiments for the GNN Graph Classification Homework
# =============================================================================

# Datasets
DATASETS=("PROTEINS" "IMDB-MULTI" "REDDIT-BINARY" "ogbg-molhiv")

# Feature-Only Models
FEATURE_ONLY=("linear" "mlp" "deepsets")

# Message Passing Models
MSG_PASSING=("gcn" "graphsage" "gat" "gin")

# All models
ALL_MODELS=("${FEATURE_ONLY[@]}" "${MSG_PASSING[@]}")

# Common parameters
HIDDEN=64
LAYERS=4
DROPOUT=0.5
LR=0.001
BS=64
EPOCHS=100
PATIENCE=20
SEEDS="0 1 2"
POOL="mean"

echo "=========================================="
echo "GNN Graph Classification Experiments"
echo "=========================================="

# Part 1: Feature-Only Models on PROTEINS and REDDIT-BINARY
echo ""
echo "--- Part 1: Feature-Only Models ---"
for dataset in "PROTEINS" "REDDIT-BINARY"; do
    for model in "${FEATURE_ONLY[@]}"; do
        echo ""
        echo "Running: $dataset / $model"
        python main.py \
            --dataset "$dataset" \
            --model "$model" \
            --pool $POOL \
            --hidden $HIDDEN \
            --layers $LAYERS \
            --dropout $DROPOUT \
            --lr $LR \
            --batch_size $BS \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            --seeds $SEEDS \
            --save_plots
    done
done

# Part 2: All Models on All Datasets
echo ""
echo "--- Part 2: All Models on All Datasets ---"
for dataset in "${DATASETS[@]}"; do
    for model in "${ALL_MODELS[@]}"; do
        echo ""
        echo "Running: $dataset / $model"
        python main.py \
            --dataset "$dataset" \
            --model "$model" \
            --pool $POOL \
            --hidden $HIDDEN \
            --layers $LAYERS \
            --dropout $DROPOUT \
            --lr $LR \
            --batch_size $BS \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            --seeds $SEEDS \
            --save_plots
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
