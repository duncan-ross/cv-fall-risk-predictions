#! /bin/bash

echo "Logistic Regression, Training Results:\n"
python3 src/evaluation.py \
    --predictions_file "predictions/logistic-predictions-training.csv" \
    --loss "weighted_ce"
echo "---\n"

echo "Logistic Regression, Test Results:\n"
python3 src/evaluation.py \
    --predictions_file "predictions/logistic-predictions.csv" \
    --loss "weighted_ce"
echo "---\n"

echo "Survey NN, Test Results:\n"
python3 src/evaluation.py \
    --predictions_file "predictions/survey-nn-predictions.csv" \
    --loss "weighted_ce"
echo "---\n"

echo "Video-Only Transformer, Test Results:\n"
python3 src/evaluation.py \
    --predictions_file "predictions/transformer/transformer-predictions.csv" \
    --loss "weighted_ce"
echo "---\n"

echo "Fusion Model, Test Results:\n"
python3 src/evaluation.py \
    --predictions_file "predictions/fusion_mc_preds.csv" \
    --loss "weighted_ce"
echo "---\n"