#! /bin/bash

python src/evaluation.py --predictions_file # Insert predictions file here

python3 src/evaluation.py \
    --predictions_file "predictions/logistic-predictions.csv" \
    --loss "weighted_ce"

python3 src/evaluation.py \
    --predictions_file "predictions/logistic-predictions-training.csv" \
    --loss "weighted_ce"

python3 src/evaluation.py \
    --predictions_file "predictions/survey-nn-predictions.csv" \
    --loss "weighted_ce"