import random
random.seed(10)
import numpy as np
np.random.seed(10)

import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

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
from sklearn.base import clone

from top500models import (get_clean_datasets_rmax, get_clean_datasets_efficiency, clean_data_dep_var_rmax, clean_data_dep_var_efficiency, 
no_dupes, all_datasets, normalize_and_split, standardize_dataset)

"""
Calculates the validation or holdout results from the train-on-past (ToA) case
study of the experiment
@param alg_hyp_set_combo: an untrained instance of a scikit-learn class with its hyperparameters
set in the constructor. For example, to use a RandomForest with max_depth=5, pass in a RandomForest(max_depth=5) instance.
For dnn1 and dnn2, use the class names DNN1 and DNN2 respectively. Don't forget to import these from top500models.py.
@param scaler_class: A scikit-learn class representing the feature scaler to use, either StandardScaler, RobustScaler,
or MinMaxScaler. Do not pass in an instance such as StandardScaler() or a string such as "StandardScaler"
@param dependent_variable: str with the value "Log(Rmax)" or "Log(Efficiency)", specifies which
benchmark measurement to predict
@param is_holdout: if true, train the model on the entire non-holdout set and test on the holdout set. If false,
perform 20 train-validation splits on the non-holdout set and average the results
@return: (avg, std) of average R^2 scores if is_holdout is false; otherwise, (score, None) where score is the R^2
score on the holdout set
"""
def calc_ToA_result(alg_hyp_set_combo, scaler_class, dependent_variable, is_holdout):
  if not isinstance(alg_hyp_set_combo, BaseEstimator):
    raise TypeError("alg_hyp_set_combo must be an instance of BaseEstimator, but was a {type(alg_hyp_set_combo).__name__}")
  if scaler_class not in [StandardScaler, RobustScaler, MinMaxScaler]:
    raise ValueError("scaler_class must be either the StandardScaler, RobustScaler, or MinMaxScaler classes, but was {scaler_class}")
  if not isinstance(dependent_variable, str):
    raise TypeError(f"dependent_variable must be a str, but was a {type(dependent_variable).__name__}")
  elif dependent_variable not in ["Log(Rmax)", "Log(Efficiency)"]:
    raise ValueError(f"dependent_variable must be either 'Log(Rmax)' or 'Log(Efficiency)', but was '{dependent_variable}'")
  
  non_holdout, holdout = holdout_split(all_datasets)

  if is_holdout:
      score = train_test_predict(non_holdout, holdout, alg_hyp_set_combo, dep_var=dependent_variable, scaler=scaler_class)
      return (score, None)
  else:
      return random_subsample_val(non_holdout, alg_hyp_set_combo, dep_var=dependent_variable, scaler=scaler_class)

"""
Given a list of raw dataframes, combines all the data
and splits it into a holdout set and a non-holdout set
@param datasets_more_features: raw TOP500 lists with more features
@param frac_test: fraction of data to be used in the holdout set
@return: (non_holdout, holdout) where non_holdout is training/validation
data and holdout is test data
"""
def holdout_split(datasets_more_features, frac_test=0.1):
  all_lists = pd.DataFrame()
  for dataset in datasets_more_features:
    all_lists = all_lists.append(standardize_dataset(dataset))
  non_holdout, holdout = train_test_split(all_lists, test_size=frac_test, random_state=10)
  non_holdout = non_holdout.reset_index(drop=True)
  holdout = holdout.reset_index(drop=True)
  return (non_holdout, holdout)

"""
Given a training set and a test set, trains the model
on the training set and evaluates it on the testing set.
Excludes test set observations that also appear in the training set
@param train: raw, unnormalized training set
@param test: raw, unnormalized testing set
@param model: model to evaluate
@param dep_var: dependent variable, either 'Log(Rmax)' or 'Log(Efficiency)'
@param scaler: class of scikit-learn scaling transformer to use on the x data
@return: R^2 score of prediction
"""
def train_test_predict(train, test, model, dep_var="Log(Rmax)", scaler=StandardScaler):
  raw_train = train.copy()
  if dep_var == "Log(Rmax)":
    train = clean_data_dep_var_rmax(train, raw_train, is_training_set=True, exclude_dupes=True)
    test = clean_data_dep_var_rmax(test, raw_train, is_training_set=False, exclude_dupes=True)
  elif dep_var == "Log(Efficiency)":
    train = clean_data_dep_var_efficiency(train, raw_train, is_training_set=True, exclude_dupes=True)
    test = clean_data_dep_var_efficiency(test, raw_train, is_training_set=False, exclude_dupes=True)
  else:
    raise ValueError(f"{dep_var} is not a valid dependent variable name")

  train_x, train_y, test_x, test_y = normalize_and_split(train, test, normalizer=scaler)
  model = clone(model) #clone() wipes any previous data from the model if it was trained already
  model.fit(train_x, train_y)
  pred_y = model.predict(test_x)
  return r2_score(test_y, pred_y)

"""
Implements the 'Random Subsampling Validation' technique
described in 'Predicting New Workload or CPU Performance 
by Analyzing Public Datasets' by Yu Wang, Victor Lee, 
Gu-Yeon Wei, and David Brooks.
Given a raw dataset, splits the set into a training and validation section,
training the model on the training set and scoring it on the validation set.
Repeats this process many times with different train/val splits each time.
@param non_holdout: non heldout raw, standardized data, combined into one dataframe
@param model: untrained model
@param iterations: number of times to repeat the process
@param frac_val: fraction of the set that will be used for validation
@param scaler: class of scikit-learn scaling transformer to use on the x data
@param dep_var: dependent variable, either 'Log(Rmax)' or 'Log(Efficiency)'
@return: (avg, std) tuple listing the mean and standard deviation R^2 scores 
across all iterations
"""
def random_subsample_val(non_holdout, model, iterations=20, frac_val=0.1, dep_var="Log(Rmax)", scaler=StandardScaler, new_feats=None):
  random.seed(10)
  scores_by_iteration = []
  for iteration in range(iterations):
    train, val = train_test_split(non_holdout, test_size=frac_val)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    
    score = train_test_predict(train, val, model, dep_var=dep_var, scaler=scaler)
    scores_by_iteration.append(score)
  return (np.mean(scores_by_iteration), np.std(scores_by_iteration))
