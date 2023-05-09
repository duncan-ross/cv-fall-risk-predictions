#! /bin/bash

python src/evaluation.py --predictions_file # Insert predictions file here
python src/evaluation.py --predictions_file --loss "weighted_ce" --weight_file