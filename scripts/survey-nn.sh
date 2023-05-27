python3 src/run_survey_model.py --function "train" --writing_params_path "parameters/survey-nn.params" --reading_params_path "parameters/survey-nn.params"

python3 src/run_survey_model.py --function "evaluate" --writing_params_path "parameters/survey-nn.params" --reading_params_path "parameters/survey-nn.params" --outputs_path "predictions/survey-nn-predictions.csv"