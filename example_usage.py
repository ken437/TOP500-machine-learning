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

from ToPResultsReplicator import calc_ToP_result
from top500models import DNN1, DNN2

print("Doing a train-on-past run with a RandomForestRegressor with max_depth 5, scaled with StandardScaler, predicting Log(Rmax) in list #14 having trained on lists #12 and 13")
print("R^2 Score: %.3f" % calc_ToP_result(RandomForestRegressor(max_depth=5), StandardScaler, "Log(Rmax)", 12, 14))

# print("Doing a train-on-past run with a default KNeighborsRegressor, scaled with RobustScaler, predicting Log(Efficiency) in list #7 having trained on lists #1-6")
# print("R^2 Score: %.3f" % calc_ToP_result(KNeighborsRegressor(), RobustScaler, "Log(Efficiency)", 1, 7))

print("Doing a train-on-past run with dnn1, scaled with MinMaxScaler, predicting Log(Rmax) in list #18 having trained on list #17")
print("R^2 Score: %.3f" % calc_ToP_result(DNN1(), MinMaxScaler, "Log(Rmax)", 17, 18))

print("Doing a train-on-past run with dnn2, scaled with StandardScaler, predicting Log(Rmax) in list #4 having trained on lists #2 and 3")
print("R^2 Score: %.3f" % calc_ToP_result(DNN2(), StandardScaler, "Log(Rmax)", 2, 4))

print("Doing a train-on-past run with a default LGBMRegressor, scaled with MinMaxScaler, predicting Log(Rmax) in list #18 having trained on list #17")
print("R^2 Score: %.3f" % calc_ToP_result(LGBMRegressor(), MinMaxScaler, "Log(Rmax)", 17, 18))
