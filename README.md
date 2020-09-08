Overview
--------
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

Restrictions on Argument Values
-------------------------------
Command line argument values are restricted in the following ways:
* model_evaluation_methodology must be either the string "ToP" or the string "ToA"
* dependent_variable must be either the string "Log(Rmax)" or "Log(Efficiency)"
* architecture must be one of the strings "Cluster", "MPP", or "Constellations"
* microarchitecture must be one of the following strings: 'AMD Zen (Naples)', 'AMD Zen-2 (Rome)', 'Bulldozer', 'CBEA', 'IBM A2', 'Intel Broadwell', 'Intel Cascade lake', 'Intel Core', 'Intel Haswell', 'Intel IvyBridge', 'Intel Nehalem', 'Intel SandyBridge', 'Intel Skylake', 'Intel Westmere', 'K10', 'K8', 'Knights Landing', 'Many Integrated Cores', 'Montecito', 'POWER5', 'POWER6', 'POWER7', 'POWER9', 'Piledriver', 'PowerPC 970', 'SBSA', 'SPARC64 IXfx', 'SPARC64 VII', 'SPARC64 VIIIfx', 'SPARC64 XIfx', 'SW26010', 'Unknown', 'Vulcan', or 'Zen'
* year must be an int or float
* clockspeed must be a positive int or float
* total cores must be a positive int
* fraction of cores that are accelerators must be an int or float in the interval [0, 1]
* seed, if used, shoudl be an int

Training Set Information
------------------------
* ToP models (models found to be best suited for the ToP model evaluation methodology) were trained as though they were trying to predict November 2020 data. For example, the best model for predicting Log(Rmax) when using the ToP methodology was a gradient boosted ensemble with the training set selection strategy 'two_prev.' This model was trained on the November 2019 and June 2020 TOP500 lists, since those are the two lists released immediately before the November 2020 lists.
* ToA models (models found to be best suited for the ToA model evaluation methodology) were simply trained on all the TOP500 data used in our experiment, i.e. all TOP500 lists between the November 2011 and June 2020 lists inclusive.

Library Versions
----------------
Library versions needed for this project can be found in the ``requirements.txt`` file.
