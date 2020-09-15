Overview
--------
This repository contains three important files that are useful for testing our models and replicating our results: ``make_predictions.py``, `` ToAResultsReplicator.py``, and ``ToPResultsReplicator.py``. The former file is for using our best models to make predictions about hypothetical supercomputers, and the latter two files are for evaluating our models on TOP500 data. Below we will describe how to use these files:

``make_predictions.py`` Description
-----------------------------------
This repository contains a script called ``make_predictions.py`` that can be used to train any of our experiment's four best models and use it to make predictions. This script takes a number of command line arguments: two to specify which model to use, six to specify the feature values to use when making a performance/power efficiency prediction, and one optional argument to set the random seed. For more information run ``python3 make_predictions.py -h`` in the same directory as the ``make_predictions.py`` script.

There are two arguments specifying which model to use. The first, ``model_evaluation_methodology``, takes two values: ``ToP`` or ``ToA``. This argument specifies which case study to look at when deciding what model to use. The second argument, ``dependent_variable``, takes either ``"Log(Rmax)"`` or ``"Log(Efficiency)"``. Make sure to surround this argument with quotes to prevent the parentheses from causing problems. This argument specifies the target variable that the model will try to predict, and also further specifies which model to use when making predictions. To get an understanding of how these two arguments work together, it may be helpful to point out an example. If the make_predictions.py script is run with ``ToP`` and ``"Log(Efficiency)"`` as its first two arguments, the script will train the model that best predicted Log(Efficiency) in the train on past (ToP) case study of our experiment. This model will be trained to predict Log(Efficiency) and will output a Log(Efficiency) prediction.

The next six arguments can be used to specify the features of a theoretical supercomputer to simulate. The feature arguments appear in the following order:
1. architecture
1. microarchitecture (uarch)
1. year
1. clockspeed
1. total cores
1. fraction of cores that are accelerators (frac_accel_cores)

Remember to surround multi-word microarchitecture arguments with quotation marks to clarify that they are a single argument.

The final, optional, argument is a value that can be used to seed the random number generator. It takes a default value of 10, but setting it to different values can be useful for determining the effect of chance on the model predictions.

``make_predictions.py`` Restrictions on Argument Values
-------------------------------------------------------
Command line argument values are restricted in the following ways:
* model_evaluation_methodology must be either the string "ToP" or the string "ToA"
* dependent_variable must be either the string "Log(Rmax)" or "Log(Efficiency)"
* architecture must be one of the strings "Cluster", "MPP", or "Constellations"
* microarchitecture must be one of the following strings: 'AMD Zen (Naples)', 'AMD Zen-2 (Rome)', 'Bulldozer', 'CBEA', 'IBM A2', 'Intel Broadwell', 'Intel Cascade lake', 'Intel Core', 'Intel Haswell', 'Intel IvyBridge', 'Intel Nehalem', 'Intel SandyBridge', 'Intel Skylake', 'Intel Westmere', 'K10', 'K8', 'Knights Landing', 'Many Integrated Cores', 'Montecito', 'POWER5', 'POWER6', 'POWER7', 'POWER9', 'Piledriver', 'PowerPC 970', 'SBSA', 'SPARC64 IXfx', 'SPARC64 VII', 'SPARC64 VIIIfx', 'SPARC64 XIfx', 'SW26010', 'Unknown', 'Vulcan', or 'Zen'
* year must be an int
* clockspeed must be a positive float
* total cores must be a positive int
* fraction of cores that are accelerators must be an int or float in the interval [0, 1]
* seed, if used, should be an int

``make_predictions.py`` Training Set Information
------------------------------------------------
* ToP models (models found to be best suited for the ToP model evaluation methodology) were trained as though they were trying to predict November 2020 data. For example, the best model for predicting Log(Rmax) when using the ToP methodology was a gradient boosted ensemble with the training set selection strategy 'two_prev.' This model was trained on the November 2019 and June 2020 TOP500 lists, since those are the two lists released immediately before the November 2020 lists.
* ToA models (models found to be best suited for the ToA model evaluation methodology) were simply trained on all the TOP500 data used in our experiment, i.e. all TOP500 lists between the November 2011 and June 2020 lists inclusive.

``ToAResultsReplicator.py`` and ``ToPResultsReplicator.py`` Description
-----------------------------------------------------------------------

Unlike ``make_predictions.py``, which is a script that is meant to be run from the command line, these two files are designed to be imported. When using these files, we recommend creating a Python file in the root directory of this repository and running the following imports:

```Python
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
```

The important functions for replicating our results are the ``calc_ToP_result``, ``calc_ToP_avg_val_score``, and ``calc_ToA_result`` functions. These functions can replicate the result of an individual train-test split in the train-on-past section of our experiment, replicate the average and standard deviation validation-phase model scores in the train-on-past section of our experiment, and replicate the validation or holdout results in the train-on-all section of our experiment, respectively. Detailed information on these functions is found in their documentation within the ``ToAResultsReplicator.py`` and ``ToPResultsReplicator.py`` fileslu.

Examples of Using ``ToAResultsReplicator.py`` and ``ToPResultsReplicator.py``
-----------------------------------------------------------------------------

The file ``example_usage.py`` contains examples of how to use the functions in these two files. When running the script with ``python3 example_usage.py``, the program will print a message describing which of the experimental results it is going to calculate, and then it will calculate that result, printing the result to standard output. For example:

```
Finding train-on-all avg. validation phase scores with a KNeighborsRegressor with p = 1, scaled with MinMaxScaler, predicting Log(Efficiency)
Avg. R^2 Score: 0.704, Std. R^2 Score: 0.079
```

This will be repeated for many different experimental results.

If the script is run on a system without the proper accelerators, Tensorflow warning messages such as the following may appear:

```
2020-09-14 07:58:35.982993: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-09-14 07:58:35.983041: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
```

To resolve these warnings, copy the code to a system that is equipped with accelerators and try again.

Library Versions
----------------

Library versions needed for this project can be found in the ``requirements.txt`` file.
To install these Python libraries, you need the `virtualenv` package on your system
which you should install using your system's package manager.

Then you can run:

```
virtualenv -p python3 venv
. venv/bin/activate
pip install -r requirements.txt
```

Python 3.6.9 or more recent is required.
Tested on Ubuntu 18.04-5 LTS and Ubuntu 20.04.

