import random
random.seed(10)

import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score

from sklearn.base import BaseEstimator

from top500models import get_clean_datasets_rmax, get_clean_datasets_efficiency, no_dupes, all_datasets, normalize_and_split
from train_set_select_strats import * #okay because I know everything that's in this file

"""
Calculates the validation results from the train-on-past (ToP) case
study of the experiment. Models are trained on 
@param alg_hyp_set_combo: an untrained instance of a scikit-learn class with its hyperparameters
set in the constructor. For instance, to use a RandomForest with max_depth=5, pass in RandomForest(max_depth=5).
For dnn1 and dnn2, use the class names DNN1 and DNN2 respectively. Don't forget to import these.
@param scaler_class: A scikit-learn class representing the feature scaler to use, either StandardScaler, RobustScaler,
or MinMaxScaler. Do not pass in an instance such as StandardScaler() or a string such as "StandardScaler"
@param dependent_variable: str with the value "Log(Rmax)" or "Log(Efficiency)", specifies which
benchmark measurement to predict
@param train_set_select_strat: training set selection strategy (e.g. one_prev). Import the train_set_select_strats file and choose a strategy from the variable ALL_TRAIN_SET_SELECT_STRATS.
@return: (avg, std) of R^2 prediction scores for all the validation train-test splits.
"""
def calc_ToP_avg_val_score(alg_hyp_set_combo, scaler_class, dependent_variable, train_set_select_strat):
    if train_set_select_strat not in ALL_TRAIN_SET_SELECT_STRATS:
        raise ValueError(f"train_set_select_strat that was input, {train_set_select_strat}, must be one of the following functions from the train_set_select_strats.py file: {[strat.__name__ for strat in ALL_TRAIN_SET_SELECT_STRATS]}")
    r2_results = []
    for val_set_idx in range(2, 18):
        train_set_size = train_set_select_strat(val_set_idx)
        score = calc_ToP_result(alg_hyp_set_combo, scaler_class, dependent_variable, val_set_idx - train_set_size, val_set_idx)
        r2_results.append(score)
    return (np.mean(r2_results), np.std(r2_results))

"""
Calculates an individual result from the train-on-past (ToP) case
study of the experiment
@param alg_hyp_set_combo: an untrained instance of a scikit-learn class with its hyperparameters
set in the constructor. For instance, to use a RandomForest with max_depth=5, pass in RandomForest(max_depth=5).
For dnn1 and dnn2, use the class names DNN1 and DNN2 respectively. Don't forget to import these.
@param scaler_class: A scikit-learn class representing the feature scaler to use, either StandardScaler, RobustScaler,
or MinMaxScaler. Do not pass in an instance such as StandardScaler() or a string such as "StandardScaler"
@param dependent_variable: str with the value "Log(Rmax)" or "Log(Efficiency)", specifies which
benchmark measurement to predict
@param first_train_idx: int from 1-17 inclusive, specifies the list # of the first TOP500
list in the training set. Must be less than test_idx.
@param test_idx: int from 2-18 inclusive, specifies the list # of the TOP500 list to use
in the testing set. For example, if first_train_idx is 3 and test_idx is 6, lists 3, 4, and 5
will be trained on, while list 6 will be the testing set.
@return: R^2 prediction score for this train-test split.
"""
def calc_ToP_result(alg_hyp_set_combo, scaler_class, dependent_variable, first_train_idx, test_idx):
    if not isinstance(alg_hyp_set_combo, BaseEstimator):
        raise TypeError("alg_hyp_set_combo must be an instance of BaseEstimator, but was a {type(alg_hyp_set_combo).__name__}")
    if scaler_class not in [StandardScaler, RobustScaler, MinMaxScaler]:
        raise ValueError("scaler_class must be either the StandardScaler, RobustScaler, or MinMaxScaler classes, but was")
    if not isinstance(dependent_variable, str):
        raise TypeError(f"dependent_variable must be a str, but was a {type(dependent_variable).__name__}")
    elif dependent_variable not in ["Log(Rmax)", "Log(Efficiency)"]:
        raise ValueError(f"dependent_variable must be either 'Log(Rmax)' or 'Log(Efficiency)', but was '{dependent_variable}'")
    if not isinstance(first_train_idx, int):
        raise TypeError(f"first_train_idx must be an int, but was a {type(first_train_idx).__name__}")
    elif first_train_idx not in range(1, 18):
        raise ValueError(f"first_train_idx must be between 1 and 17 inclusive, but was {first_train_idx}")
    if not isinstance(test_idx, int):
        raise TypeError(f"test_idx must be an int, but was a {type(test_idx).__name__}")
    elif test_idx not in range(2, 19):
        raise ValueError(f"test_idx must be between 2 and 18 inclusive, but was {test_idx}")
    if test_idx <= first_train_idx:
        raise ValueError(f"test_idx must be greater than first_train_idx, but test_idx was {test_idx} and first_train_idx was {first_train_idx}")
    
    clean_train = None
    clean_test = None

    if dependent_variable == "Log(Rmax)":
        clean_train_dep_var_rmax, clean_test_dep_var_rmax = get_clean_datasets_rmax(all_datasets, test_idx, range(first_train_idx, test_idx))
        clean_test_dep_var_rmax_no_dupes = no_dupes(clean_train_dep_var_rmax, clean_test_dep_var_rmax)
        clean_train = clean_train_dep_var_rmax
        clean_test = clean_test_dep_var_rmax_no_dupes
    else:
        clean_train_dep_var_efficiency, clean_test_dep_var_efficiency = get_clean_datasets_efficiency(all_datasets, test_idx, range(first_train_idx, test_idx))
        clean_test_dep_var_efficiency_no_dupes = no_dupes(clean_train_dep_var_efficiency, clean_test_dep_var_efficiency)
        clean_train = clean_train_dep_var_efficiency
        clean_test = clean_test_dep_var_efficiency_no_dupes

    random.seed(10)

    train_x, train_y, test_x, test_y = normalize_and_split(clean_train, clean_test, normalizer=scaler_class)

    model = alg_hyp_set_combo
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    return r2_score(test_y, pred_y)
