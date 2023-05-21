#! /bin/bash

python src/01-preprocess-data.py --data_dir data/ --seed 7654321
python src/03-survey-encoding.py