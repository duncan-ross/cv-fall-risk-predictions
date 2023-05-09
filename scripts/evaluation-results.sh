#! /bin/bash

python src/evaluation.py --predictions_file # Insert predictions file here
python3 src/evaluation.py \
    --predictions_file "predictions/logistic-predictions.csv" \
    --loss "weighted_ce"
