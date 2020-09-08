import random
import argparse
import numpy as np
from top500models import train_model

#read in command line inputs
parser = argparse.ArgumentParser(description='Read settings and features input via the command line, train an appropriate machine learning model, and print the prediction.\n'
        + 'Additional instructions for using this script are provided in the README.md file')
parser.add_argument('model_eval_method', metavar = 'model_eval_method', type=str, help='(str) Model evaluation methodology (ToP or ToA)')
parser.add_argument('dependent_variable', metavar = 'dependent_variable', type=str, help='(str) Dependent/target variable ("Log(Rmax)" or "Log(Efficiency)")')
parser.add_argument('architecture', metavar = 'architecture', type=str, help='(str) Feature specifying the computer architecture')
parser.add_argument('microarchitecture', metavar = 'uarch', type=str, help='(str) Feature specifying the computer microarchitecture')
parser.add_argument('year', metavar = 'year', type=int, help='(int) Feature specifying the computer year')
parser.add_argument('clockspeed', metavar = 'clockspeed', type=float, help='(float) Feature specifying the computer clockspeed')
parser.add_argument('total_cores', metavar = 'total_cores', type=int, help='(int) Feature specifying the computer core count')
parser.add_argument('frac_accel_cores', metavar = 'frac_accel_cores', type=float, help='(float) Feature specifying the fraction of cores that are accelerators')
parser.add_argument('--seed', metavar = 'seed', type=int, action='store', default=10, help='(int) Optional value to seed the random number generator (default=10)')
argsset = parser.parse_args()

#echo inputs
if argsset.model_eval_method not in ['ToP', 'ToA']: raise ValueError('The model_eval_method parameter must take one of two values: "ToP" or "ToA"')
print(f'Using the model that best predicted {argsset.dependent_variable} during the {"train-on-past (ToP)" if argsset.model_eval_method == "ToP" else "train-on-all (ToA)"} case study\n')
print(f'The theoretical supercomputer observation will have the following features:')
print(f'Architecture: {argsset.architecture}')
print(f'Microarchitecture: {argsset.microarchitecture}')
print(f'Year: {argsset.year}')
print(f'Clockspeed: {argsset.clockspeed} GHz')
print(f'Total Cores: {argsset.total_cores}')
print(f'Fraction of Cores that are Accelerators: {argsset.frac_accel_cores}')
print()

# argsset contains all args parsed in the beginning of the file

# identify which model to use
model_evaluation_methodology = argsset.model_eval_method
dependent_variable = argsset.dependent_variable

# specify the feature values
features = {
    "Architecture": argsset.architecture, 
    "Microarchitecture": argsset.microarchitecture,
    "Year": argsset.year,
    "Clockspeed": argsset.clockspeed,
    "Total Cores": argsset.total_cores,
    "Fraction of Cores that are Accelerators": argsset.frac_accel_cores
}

# seed the random number generator
seed = argsset.seed
random.seed(seed)

predictor = train_model(model_evaluation_methodology, dependent_variable)
prediction = predictor(features)
is_rmax = dependent_variable == "Log(Rmax)"
print(f'{dependent_variable} prediction: {round(prediction, 3)}')
print(f'This corresponds to a {"Rmax" if is_rmax else "Efficiency"} value of {round(np.exp(prediction), 3)} {"TFlops" if is_rmax else "GFlops/Watt"}')
