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

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from ToPResultsReplicator import calc_ToP_result, calc_ToP_avg_val_score
from ToAResultsReplicator import calc_ToA_result
from top500models import DNN1, DNN2
from train_set_select_strats import one_prev, two_prev, three_prev, four_prev, all_prev, half_prev, third_prev

print("Example Usage for Train-on-Past (ToP) Case Study Results")
print("-" * 75)

print("Doing a train-on-past run with a RandomForestRegressor with max_depth 5, scaled with StandardScaler, predicting Log(Rmax) in list #14 having trained on lists #12 and 13")
print("R^2 Score: %.3f" % calc_ToP_result(RandomForestRegressor(max_depth=5), StandardScaler, "Log(Rmax)", 12, 14, random_state=7))

print("Doing a train-on-past run with a default KNeighborsRegressor, scaled with RobustScaler, predicting Log(Efficiency) in list #7 having trained on lists #1-6")
print("R^2 Score: %.3f" % calc_ToP_result(KNeighborsRegressor(), RobustScaler, "Log(Efficiency)", 1, 7))

print("Doing a train-on-past run with dnn1, scaled with MinMaxScaler, predicting Log(Rmax) in list #18 having trained on list #17")
print("R^2 Score: %.3f" % calc_ToP_result(DNN1(), MinMaxScaler, "Log(Rmax)", 17, 18))

print("Doing a train-on-past run with dnn2, scaled with StandardScaler, predicting Log(Rmax) in list #4 having trained on lists #2 and 3")
print("R^2 Score: %.3f" % calc_ToP_result(DNN2(), StandardScaler, "Log(Rmax)", 2, 4))

print("Doing a train-on-past run with a default LGBMRegressor, scaled with MinMaxScaler, predicting Log(Rmax) in list #18 having trained on list #17")
print("R^2 Score: %.3f" % calc_ToP_result(LGBMRegressor(), MinMaxScaler, "Log(Rmax)", 17, 18))

print("Getting the train-on-past validation scores with dnn2, scaled with RobustScaler, predicting Log(Rmax) using all_prev")
avg_r2, std_r2 = calc_ToP_avg_val_score(DNN2(), RobustScaler, "Log(Rmax)", all_prev, random_state=7)
print("Avg. R^2 Score: %.3f Std. R^2 Score: %.3f" % (avg_r2, std_r2))

print("Doing a train-on-past run with a default GradientBoostingRegressor, scaled with RobustScaler, predicting Log(Rmax) in list #11 having trained on lists #9 and 10")
print("R^2 Score: %.3f" % calc_ToP_result(GradientBoostingRegressor(), RobustScaler, "Log(Rmax)", 9, 11))

print("Getting the train-on-past validation scores for a LinearRegression, scaled with RobustScaler, predicting Log(Efficiency) with one_prev")
avg_r2, std_r2 = calc_ToP_avg_val_score(LinearRegression(), RobustScaler, "Log(Efficiency)", one_prev)
print("Avg. R^2 Score: %.3f, Std. R^2 Score: %.3f" % (avg_r2, std_r2))

print()
print("Example Usage for Train-on-All (ToA) Case Study Results")
print("-" * 75)

print("Finding train-on-all avg. validation phase scores with a KNeighborsRegressor with p = 1, scaled with MinMaxScaler, predicting Log(Efficiency)")
avg_r2, std_r2 = calc_ToA_result(KNeighborsRegressor(p=1), MinMaxScaler, "Log(Efficiency)", is_holdout=False)
print("Avg. R^2 Score: %.3f, Std. R^2 Score: %.3f" % (avg_r2, std_r2))

print("Finding train-on-all avg. validation phase scores with an XGBRegressor, scaled with RobustScaler, predicting Log(Rmax)")
avg_r2, std_r2 = calc_ToA_result(XGBRegressor(), RobustScaler, "Log(Rmax)", is_holdout=False, random_state=7)
print("Avg. R^2 Score: %.3f, Std. R^2 Score: %.3f" % (avg_r2, std_r2))

print("Finding train-on-all holdout set scores with a RandomForestRegressor with n_estimators = 1000, scaled with RobustScaler, predicting Log(Rmax)")
r2, _ = calc_ToA_result(RandomForestRegressor(n_estimators=1000), RobustScaler, "Log(Rmax)", is_holdout=True)
print("R^2 Score: %.3f" % r2)

print("Finding train-on-all holdout set scores with a default LGBMRegressor, scaled with MinMaxScaler, predicting Log(Efficiency)")
r2, _ = calc_ToA_result(LGBMRegressor(), MinMaxScaler, "Log(Efficiency)", is_holdout=True)
print("R^2 Score: %.3f" % r2)
