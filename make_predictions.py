import pandas as pd
import numpy as np
import re
import math
import random
import argparse

#machine learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

#read in command line inputs
parser = argparse.ArgumentParser(description='Read settings and features input via the command line, train an appropriate machine learning model, and print the prediction')
parser.add_argument('model_eval_method', metavar = 'model_eval_method', type=str, help='(str) Model evaluation methodology (ToP or ToA)')
parser.add_argument('dependent_variable', metavar = 'dependent_variable', type=str, help='(str) Dependent/target variable (LogRmax or LogEfficiency)')
parser.add_argument('architecture', metavar = 'architecture', type=str, help='(str) Feature specifying the computer architecture')
parser.add_argument('microarchitecture', metavar = 'uarch', type=str, help='(str) Feature specifying the computer microarchitecture')
parser.add_argument('year', metavar = 'year', type=int, help='(int) Feature specifying the computer year')
parser.add_argument('clockspeed', metavar = 'clockspeed', type=float, help='(float) Feature specifying the computer clockspeed')
parser.add_argument('total_cores', metavar = 'total_cores', type=int, help='(int) Feature specifying the computer core count')
parser.add_argument('frac_accel_cores', metavar = 'frac_accel_cores', type=float, help='(float) Feature specifying the fraction of cores that are accelerators')
parser.add_argument('--seed', metavar = 'seed', type=int, action='store', default=10, help='(int) Optional value to seed the random number generator (default=10)')
argsset = parser.parse_args()

# read in the files (assumes TOP500 files are stored in a directory called 'TOP500_files' in the same directory as this script

top500_dir_path = "TOP500_files/"
month_strs = ["06", "11"]
filenames = ["TOP500_" + str(year) + month + ".xls" for year in range(2012, 2020) for month in month_strs]
filenames = ["TOP500_201111.xls"] + filenames
filenames.append("TOP500_202006.xlsx")
all_datasets = []

for filename in filenames:
  curr_dataset = pd.read_excel(top500_dir_path + filename, header=0)
  all_datasets.append(curr_dataset)

"""
Takes a dataset and makes it conform to the same format as
all other datasets (same column names, units etc.)

NOTE: this function does NOT give each feature a mean of zero
and standard deviation of 1. It instead makes it safe to concatenate
different datasets into one dataframe by fixing column and unit
discrepancies.

NOTE: this function is limited to preprocessing steps that
do not cause data leakage because the result applying this function to any
given data observation is completely unaffected by the other
data observations in the dataset. Thus, applying it to the train
and test functions pre-split should give identical results to applying
it post-split. To test this, I applied this function to the entire 11th
more features dataset, and I also separated the 11th dataset, applied
the function to both halves individually, and appended the halves.
The resulting dataframes matched exactly. Thus, it does not matter
whether this function is used before or after splitting into training
and test sets and therefore this function can be used on the unsplit
data without the risk of data leakage.

@param dataset: the dataset that needs to be standardized
@return: the same dataset, but formatted in a consistent manner
"""
def standardize_dataset(dataset):
  dataset_copy = dataset.copy()

  #standardize units
  if "Mflops/Watt" in dataset_copy.columns:
    for row in dataset_copy.index:
      dataset_copy.at[row, "Mflops/Watt"] /= 1000

  #Rmax (without explicit units) is actually in GFlop/s!
  if "Rmax" in dataset_copy.columns:
    for row in dataset_copy.index:
      dataset_copy.at[row, "Rmax"] /= 1000
    dataset_copy = dataset_copy.rename({"Rmax": "Rmax [TFlop/s]"}, axis=1)
  if "RMax" in dataset_copy.columns:
    for row in dataset_copy.index:
      dataset_copy.at[row, "RMax"] /= 1000
    dataset_copy = dataset_copy.rename({"RMax": "Rmax [TFlop/s]"}, axis=1)

  #rename columns
  renaming_mapping = {"Cores": "Total Cores",
                      "Rmax [TFlop/s]": "Log(Rmax)",
                      "Power Efficiency [GFlops/Watts]": "Power Effeciency [GFlops/Watts]",
                      "Mflops/Watt": "Power Effeciency [GFlops/Watts]",
                      "Accelerator Cores": "Accelerator/Co-Processor Cores"}
  dataset_copy = dataset_copy.rename(renaming_mapping, axis=1)

  #calculate power efficiency from power and rmax if necessary
  efficiency_data_present = False
  for efficiency_col in ["Mflops/Watt", "Power Effeciency [GFlops/Watts]"]:
    if efficiency_col in dataset_copy.columns:
      efficiency_data_present = True

  if "Power" in dataset_copy.columns and not efficiency_data_present:
    for row in dataset_copy.index:
      #the Log(Rmax) column has not actually had the log transform applied to it yet
      #Rmax is in TFlops
      #Power is in Watts
      #Power Efficiency (Rmax / Power) is in GFlops/Watt
      dataset_copy.at[row, "Power Effeciency [GFlops/Watts]"] = \
      dataset_copy.at[row, "Log(Rmax)"] / (dataset_copy.at[row, "Power"] * 1000)
    dataset_copy = dataset_copy.drop(columns=["Power"])

  #transform rmax
  for row in dataset.index:
    dataset_copy.at[row, "Log(Rmax)"] = math.log(dataset_copy.at[row, "Log(Rmax)"])

  #transform power efficiency
  for row in dataset.index:
    dataset_copy.at[row, "Log(Efficiency)"] = round(math.log(dataset_copy.at[row, "Power Effeciency [GFlops/Watts]"]), 3)

  #NOTE: 'Efficiency' in Log(Efficiency) is in GFlops/Watts, but
  #'Rmax' in 'Log(Rmax)' is in TFlops. Keep this in mind if you want
  #to use these quantities to calculate the processor's power.

  return dataset_copy

import re

"""
Given a dataframe containing raw processor data, adds an additional
column to the data listing the microarchitecture name of each processor
@param dataframe: processor data dataframe. Must not have been cleaned and
must contain both a "Processor" and "Processor Technology" column.
@return: same as dataframe, but with a microarchitecture column.
"""
def find_microarchitectures(dataframe):
  if "Processor" not in dataframe.columns:
    raise ValueError("Input missing necessary column: Processor")
  elif "Processor Technology" not in dataframe.columns:
    raise ValueError("Input missing necessary column: Processor Technology")

  #ma in these variable names means "microarchitecture"
  already_mas = ['AMD Zen (Naples)', 'AMD Zen-2 (Rome)', 'Bulldozer', 'CBEA',
       'IBM A2', 'Intel Broadwell', 'Intel Cascade lake', 'Intel Core',
       'Intel Haswell', 'Intel IvyBridge', 'Intel Nehalem',
       'Intel SandyBridge', 'Intel Skylake', 'Intel Westmere', 'K10',
       'K8', 'Knights Landing', 'Many Integrated Cores', 'Montecito',
       'POWER5', 'POWER6', 'POWER7', 'POWER9', 'Piledriver',
       'PowerPC 970', 'SBSA', 'SPARC64 IXfx', 'SPARC64 VII',
       'SPARC64 VIIIfx', 'SPARC64 XIfx', 'SW26010', 'Unknown', 'Vulcan',
       'Zen']
  #This dictionary maps a processor name
  #pattern to the microarchitecture corresponding to that name pattern.
  non_mas_to_mas = {
      #AMD x86_64
      "Opteron": "K8",
      "Opteron 23[0-9]{2}": "K10",
      "Opteron 41[0-9]{2}": "K10",
      "Opteron 61[0-9]{2}": "K10",
      "Opteron 62[0-9]{2}": "Bulldozer",
      "Opteron 63[0-9]{2}": "Piledriver",
      "Opteron O-6376": "Piledriver",
      "Opteron .* Core": "K8",
      "Hygon Dhyana": "Zen",
      #Cavium
      "Cavium ThunderX2 CN99[0-9]+-[0-9]+": "Vulcan",
      #Fujitsu ARM
      "(?:Fujitsu )?A64FX": "SBSA",
      #Intel EM64T
      "Xeon [A-Z]55[0-9]{2}": "Intel Nehalem",
      "Xeon [A-Z]56[0-9]{2}": "Intel Westmere",
      "Xeon E7-4870": "Intel Westmere",
      #Intel IA-64
      "Itanium 2 (?:Montecito)|(?:Dual Core)": "Montecito",
      #Intel MIC
      "Intel Xeon Phi 5120D": "Many Integrated Cores",
      #Intel Xeon Phi
      "Intel Xeon Phi 72[0-9]{2}[A-Z]?": "Knights Landing",
      #Others/ShenWei
      "(?:Sunway)|(?:ShenWei)(?: processor)? SW1[0-9]{3}": "SW-1",
      "(?:Sunway)|(?:ShenWei)(?: processor)? SW2[0-9]{3}": "SW-2",
      "(?:Sunway)|(?:ShenWei)(?: processor)? SW3[0-9]{3}": "SW-3",
      "(?:Sunway)|(?:ShenWei)(?: processor)? SW260101": "SW26010",
      #Power
      "(?:IBM )?POWER5": "POWER5",
      "(?:IBM )?POWER6": "POWER6",
      "(?:IBM )?POWER7": "POWER7",
      "(?:IBM )?POWER8": "POWER8",
      "(?:IBM )?POWER8+": "POWER8+",
      "(?:IBM )?POWER9": "POWER9",
      "(?:IBM )?POWER10": "POWER10",
      "(?:IBM )?PowerPC 4[0-9]{2}": "PowerPC 4xx",
      "(?:IBM )?PowerPC 6[0-9]{2}": "PowerPC 6xx",
      "(?:IBM )?PowerPC 7[0-9]{2}": "PowerPC 7xx",
      "(?:IBM )?PowerPC 970": "PowerPC 970",
      "(?:IBM )?PowerXCell 8i": "CBEA",
      #PowerPC
      "Power BQC": "IBM A2",
      #Sparc
      "SPARC64 VII": "SPARC64 VII",
      "SPARC64 VIIIfx": "SPARC64 VIIIfx",
      "SPARC64 IXfx": "SPARC64 IXfx",
      "SPARC64 XIfx": "SPARC64 XIfx",
      #ThunderX2
      "Marvell ThunderX2 CN99[0-9]{2}-[0-9]{4}": "Vulcan"
  }

  dataframe_copy = dataframe.copy()
  mas_column = [None] * len(dataframe_copy.index)
  for row in dataframe_copy.index:
    curr_processor_technology = dataframe_copy.at[row, "Processor Technology"]
    if curr_processor_technology in already_mas:
      mas_column[row] = curr_processor_technology

    else:
      curr_processor = dataframe_copy.at[row, "Processor"]
      remove_numbers_at_end = re.compile("^(.*) (?:(?:[0-9]|\.|-)+C)? (?:[0-9]|\.|-)+G|MHz$")
      curr_name = remove_numbers_at_end.findall(curr_processor)
      if len(curr_name) != 1:
        raise ValueError(f"Processor name {curr_processor} did not match expected regex")
      curr_name = curr_name[0]

      found_name_match = False
      for processor_name_pattern in non_mas_to_mas:
        pattern = re.compile(processor_name_pattern)
        if pattern.match(curr_name) != None:
          microarchitecture = non_mas_to_mas[processor_name_pattern]
          mas_column[row] = microarchitecture
          found_name_match = True
      if not found_name_match:
        mas_column[row] = "Unknown"
  dataframe_copy["Microarchitecture"] = mas_column
  return dataframe_copy

"""
Finds a unique float ID number for a given row
"""
def get_row_id(dataframe, row_index, mode="float"):
  if mode == 'str':
    id = ""
    for column in dataframe.columns:
      id += str(round(dataframe.at[row_index, column], 3))
    return id
  elif mode == 'float':
    id = 0.0
    curr_multiplier = 1.0
    for column in dataframe.columns:
      id += curr_multiplier * round(dataframe.at[row_index, column], 3)
      curr_multiplier *= 2
    return id

"""
Given (cleaned) training and test datasets returns
a version of the test set, but with any observations
also occuring in the training set removed
@param training: training set (dataframe)
@param testing: testing set (dataframe)
@param verbose: if true, identify which rows were included and which were not
@param mode: 'str' or 'float', specifies how the row id is calculated
@return: modified version of the testing set
"""
def no_dupes(training, testing, verbose=False, mode='float'):
  output = pd.DataFrame(columns=testing.columns)
  training_row_ids = set([])
  for row in training.index:
    training_row_ids.add(get_row_id(training, row, mode=mode))
  for row in testing.index:
    curr_row_dupe = get_row_id(testing, row, mode=mode) in training_row_ids
    if not curr_row_dupe:
      output = output.append(testing.loc[row].copy(), ignore_index=True)
    if verbose:
      print(f"Is the row at index {row} unique? {not curr_row_dupe}")
  return output

import re

"""
Given a column name, encodes the data in that column using one-
hot encoding
@param df: dataframe containing the column to be encoded
@param colname: name of a column with categorical data
@param training_df: dataframe used for training (used to fit the encoder)
@return: a new dataframe with the column one-hot encoded
"""
def one_hot_encode(df, colname, training_df, dep_var):
  #if df has a "Microarchitecture" column, training_df must have one too
  if "Microarchitecture" in df.columns:
    training_df = find_microarchitectures(training_df)

  #find unique entries in training_df[colname]
  unique_list = []
  for row in training_df.index:
    if (training_df.at[row, colname] not in unique_list) and (not np.isnan(training_df.at[row, dep_var])):
      unique_list.append(training_df.at[row, colname])

  column_names_for_encoding = unique_list.copy()
  #append colname to the beginning of all entries in column_names_for_encoding to avoid name collisions
  for i in range(len(column_names_for_encoding)):
    column_names_for_encoding[i] = colname + ": " + column_names_for_encoding[i]
  #use column_names_for_encoding as columns in a new dataframe
  encoded_df = pd.DataFrame(index=df.index, columns=column_names_for_encoding)
  encoded_df = encoded_df.fillna(0)
  #iterate through df[colname], putting 1s in encoded_df in the correct spots
  for row in df.index:
    thisVal = df.at[row, colname]
    if thisVal in unique_list:
      encoded_df.at[row, colname + ": " + thisVal] = 1
  #concatenate encoded_df to the left side of df
  df = pd.concat([encoded_df, df], axis=1)
  #drop original column
  df = df.drop([colname], axis=1)
  #avoid the dummy variable trap by dropping a dummy variable column
  df = df.drop([column_names_for_encoding[0]], axis=1)
  return df

def _find_idx(list_to_search, elem):
  for i in range(len(list_to_search)):
    if list_to_search[i] == elem:
      return i
  return None

"""
Given a column name, encodes the data in that column using label encoding
@param df: dataframe containing the column to be encoded
@param colname: name of a column with categorical data
@param training_df: dataframe used for training (used to fit the encoder)
@return: a new dataframe with the column label encoded
"""
def label_encode(df, colname, training_df, dep_var):
  #if df has a "Microarchitecture" column, training_df must have one too
  if "Microarchitecture" in df.columns:
    training_df = find_microarchitectures(training_df)

  #find unique entries in training_df[colname]
  unique_list = []
  for row in training_df.index:
    if (training_df.at[row, colname] not in unique_list) and (not np.isnan(training_df.at[row, dep_var])):
      unique_list.append(training_df.at[row, colname])

  encoded_col = [None] * len(df.index)
  #iterate through df[colname], putting the right numbers in encoded_col
  for row in df.index:
    thisVal = df.at[row, colname]
    thisVal_idx = _find_idx(unique_list, thisVal)
    if thisVal_idx is None: #reserve 0 for unknown labels (not in training set)
      encoded_col[row] = 0
    else:
      encoded_col[row] = thisVal_idx + 1
  #replace original column
  df_cpy = df.copy()
  df_cpy[colname] = encoded_col
  return df_cpy

def _clean_data(df_to_clean, training_df, numerical_features, categorical_features, dependent_variable, \
                exclude_dupes, is_training_set, cores_count_log=True, oh_encode=True):
  df_to_clean = find_microarchitectures(df_to_clean)
  #only include important columns
  if "Clockspeed" in numerical_features:
    #clockspeed is a special feature because it is extracted from the "Processor" column
    clockspeeds_in_GHz_arr = df_to_clean["Processor"].str.match(r'^.* (.*)GHz$')
    clockspeeds_in_MHz_arr = df_to_clean["Processor"].str.match(r'^.* (.*)MHz$')
    future_clockspeed_column = []
    for row in range(len(df_to_clean.index)):
      curr_processor_name = df_to_clean.at[row, "Processor"]
      if clockspeeds_in_GHz_arr[row]:
        clockspeed = float(re.findall(r"^.* (.*)GHz$", curr_processor_name)[0])
        future_clockspeed_column.append(clockspeed)
      elif clockspeeds_in_MHz_arr[row]:
        clockspeed = float(re.findall(r"^.* (.*)MHz$", curr_processor_name)[0]) / 1000
        future_clockspeed_column.append(clockspeed)
      else:
        raise ValueError("Processor name must mention clockspeed in either GHz or MHz")
    df_to_clean["Clockspeed"] = future_clockspeed_column

  columns = numerical_features + categorical_features + dependent_variable # Y variable MUST be last!!!
  clean_df = df_to_clean[columns].copy()
  #one hot encode categorical columns
  for col in categorical_features:
    if oh_encode:
      clean_df = one_hot_encode(clean_df, col, training_df, dependent_variable[0])
    else:
      clean_df = label_encode(clean_df, col, training_df, dependent_variable[0])

  if "Accelerator/Co-Processor Cores" in clean_df.columns:
    #missing values in accelerator cores always represent 0 and cannot be interpolated
    clean_df["Accelerator/Co-Processor Cores"] = clean_df["Accelerator/Co-Processor Cores"].fillna(value=0)

  #deal with missing values
  clean_df = clean_df.dropna()
  clean_df = clean_df.reset_index(drop=True)

  #take logarithm of cores and find fraction of cores that are accelerators, if desired
  if cores_count_log and "Total Cores" in clean_df.columns:
    log_cores_col = [None] * len(clean_df.index)
    for row in clean_df.index:
      log_cores_col[row] = math.log(clean_df.at[row, "Total Cores"])
    clean_df["Log(Total Cores)"] = log_cores_col

    if "Accelerator/Co-Processor Cores" in clean_df.columns:
      frac_accel_cores_col = [None] * len(clean_df.index)
      for row in clean_df.index:
        frac_accel_cores_col[row] = clean_df.at[row, "Accelerator/Co-Processor Cores"] / clean_df.at[row, "Total Cores"]
        clean_df["Fraction of Cores that are Accelerators"] = frac_accel_cores_col
      clean_df = clean_df.drop(["Accelerator/Co-Processor Cores"], axis=1)

    #June 2019 has an entry where Accel Cores = 27740 but Total Cores = 12592. Is this a mistake?
    clean_df = clean_df.drop(["Total Cores"], axis=1)
    #dependent variable must still be last
    dep_var_col = clean_df[dependent_variable[0]].copy()
    clean_df = clean_df.drop(columns=[dependent_variable[0]])
    clean_df[dependent_variable[0]] = dep_var_col

  return clean_df

def _clean_and_manage_dupes(df_to_clean, training_df, numerical_features, categorical_features, dependent_variable, exclude_dupes, is_training_set, oh_encode):
  clean_df = _clean_data(df_to_clean, training_df, numerical_features, categorical_features, dependent_variable, exclude_dupes, is_training_set, oh_encode=oh_encode)
  if exclude_dupes and not is_training_set:
    #both train and test set must be cleaned for the no_dupes() function to work
    clean_train = _clean_data(training_df, training_df, numerical_features, categorical_features, dependent_variable, exclude_dupes, is_training_set, oh_encode=oh_encode)
    return no_dupes(clean_train, clean_df)
  else:
    return clean_df

"""
Performs preprocessing steps such as removing missing values,
excluding columns known to be irrelevant, removing units,
and one-hot encoding categorical data. Should
be executed separately on the train and test dataframes. Set up for
TOP500 datasets, and treats rmax as the dependent variable
@param df_to_clean: raw data to be preprocessed
@param training_df: training dataset (used for one hot encoding)
@param is_training_set: true if df_to_clean is the training set; false otherwise
@param exclude_dupes: if true, exclude test set observations that also appear
in the training set
@param numerical_features: features that do not require one hot encoding
@param categorical_features: features that do require one hot encoding
@param oh_encode: if True, encode features using one hot encoding. Otherwise,
use label encoding
@return: clean data suitable for machine learning
"""
def clean_data_dep_var_rmax(df_to_clean, training_df, is_training_set, exclude_dupes=False, \
                                          numerical_features = ["Total Cores", "Accelerator/Co-Processor Cores", "Clockspeed", "Year"], \
                                          categorical_features = ["Microarchitecture", "Architecture"], oh_encode=True):
  #features may differ between this and the dep_var_efficiency function
  #put clockspeed (if it is included) in the numerical_features section
  dependent_variable = ["Log(Rmax)"]
  return _clean_and_manage_dupes(df_to_clean, training_df, numerical_features, categorical_features, \
                                 dependent_variable, exclude_dupes, is_training_set, oh_encode=oh_encode)

"""
Performs preprocessing steps such as removing missing values,
excluding columns known to be irrelevant, removing units,
and one-hot encoding categorical data. Should
be executed separately on the train and test dataframes. Set up for
TOP500 datasets, and treats power efficiency as the dependent variable
@param df_to_clean: raw data to be preprocessed
@param training_df: training dataset (used for one hot encoding)
@param is_training_set: true if df_to_clean is the training set; false otherwise
@param exclude_dupes: if true, exlude test set observations that also appear
in the training set
@param numerical_features: features that do not require one hot encoding
@param categorical_features: features that do require one hot encoding
@param oh_encode: if True, encode features using one hot encoding. Otherwise,
use label encoding
@return: clean data suitable for machine learning
"""
def clean_data_dep_var_efficiency(df_to_clean, training_df, is_training_set, exclude_dupes=False, \
                                  numerical_features = ["Total Cores", "Accelerator/Co-Processor Cores", "Clockspeed", "Year"], \
                                  categorical_features = ["Microarchitecture", "Architecture"], oh_encode=True):
  dependent_variable = ["Log(Efficiency)"]
  return _clean_and_manage_dupes(df_to_clean, training_df, numerical_features, categorical_features, \
                                 dependent_variable, exclude_dupes, is_training_set, oh_encode=oh_encode)

"""
Normalizes data and splits it into x and y. Assumes last column of the
data contains the dependent variable.
@param train: the cleaned, unnormalized training DataFrame to use
@param test: the cleaned, unnormalized testing DataFrame to use
@param normalizer: the normalizer class to use. If None, do not normalize
@return: (train_x, train_y, test_x, test_y), where train_x is normalized
training x data, train_y is unnormalized training y data, test_x is normalized
testing x data, and test_y is unnormalized testing y data
"""
def normalize_and_split(train, test, normalizer=StandardScaler):
  train_x = train.values[:,:-1]
  train_y = train.values[:,-1]
  test_x = test.values[:,:-1]
  test_y = test.values[:,-1]
  #no need to normalize y data
  if normalizer is not None:
    norm = normalizer()
    norm.fit(train_x)
    train_x = norm.transform(train_x)
    test_x = norm.transform(test_x)
  return (train_x, train_y, test_x, test_y)

"""
Removes square brackets from an input string
@param in_str: input string
@return: string with all square bracket characters removed
"""
def _no_bracks(in_str):
  return in_str.replace("[", "").replace("]", "")

"""
Uses one of the paper's four best machine learning models to predict Log(Rmax)
or Log(Efficiency) TOP500 benchmark scores
@param model_evaluation_methodology: 'ToP' or 'ToA', specifies which of the
experimental case studies to look at when identifying the best model. For instance,
setting this parameter to 'ToP' means that this function will use one of the models that
did best in the ToP case study, not the ToA one.
@param dependent_variable: 'Log(Rmax)' or 'Log(Efficiency)', specifies which 
dependent variable should be predicted. Note that the model optimized for the correct
dependent variable will be used.
@param features: a dictionary specifying values for each of the six feature variables
in the observation that needs to be predicted
@return: a float specifying the predicted dependent variable value
"""
def predict_using_model(model_evaluation_methodology, dependent_variable, features):
  req_feats = ["Architecture", "Microarchitecture", "Year", "Clockspeed", 
               "Total Cores", "Fraction of Cores that are Accelerators"]
  legit_mem_vals = ["ToP", "ToA"]
  legit_dv_vals = ["Log(Rmax)", "Log(Efficiency)"]
  legit_arch_vals = ["Cluster", "MPP", "Constellations"]  
  legit_uarch_vals = ['AMD Zen (Naples)', 'AMD Zen-2 (Rome)', 'Bulldozer', 'CBEA',
       'IBM A2', 'Intel Broadwell', 'Intel Cascade lake', 'Intel Core',
       'Intel Haswell', 'Intel IvyBridge', 'Intel Nehalem',
       'Intel SandyBridge', 'Intel Skylake', 'Intel Westmere', 'K10',
       'K8', 'Knights Landing', 'Many Integrated Cores', 'Montecito',
       'POWER5', 'POWER6', 'POWER7', 'POWER9', 'Piledriver',
       'PowerPC 970', 'SBSA', 'SPARC64 IXfx', 'SPARC64 VII',
       'SPARC64 VIIIfx', 'SPARC64 XIfx', 'SW26010', 'Unknown', 'Vulcan', 'Zen']

  # input validation
  if not isinstance(model_evaluation_methodology, str):
    raise TypeError(f"The variable model_evaluation_methodology must be a string, but was a {type(model_evaluation_methodology).__name__}")
  elif model_evaluation_methodology not in legit_mem_vals:
    raise ValueError(_no_bracks(f"The variable model_evaluation_methodology must have one of these values: {legit_mem_vals}, but was {model_evaluation_methodology}"))
  elif not isinstance(dependent_variable, str):
    raise TypeError(f"The variable dependent_variable must be a string, but was a {type(dependent_variable).__name__}")
  elif dependent_variable not in legit_dv_vals:
    raise ValueError(_no_bracks(f"The variable dependent_variable must have one of these values: {legit_dv_vals}, but was {dependent_variable}"))

  missing_feats = [feat for feat in req_feats if feat not in features]
  if len(missing_feats) != 0:
    raise ValueError(_no_bracks(f"The following features are missing: {missing_feats}"))

  if not isinstance(features["Architecture"], str):
    raise TypeError(f"The feature 'Architecture' must be a string, but was a {type(features['Architecture']).__name__}")
  elif features["Architecture"] not in legit_arch_vals:
    raise ValueError(_no_bracks(f"The feature 'Architecture' must have one of these values: {legit_arch_vals}, but was {features['Architecture']}"))
  elif not isinstance(features["Microarchitecture"], str):
    raise(TypeError(f"The feature 'Microarchitecture' must be a string, but was a {type(features['Microarchitecture']).__name__}"))
  elif features["Microarchitecture"] not in legit_uarch_vals:
    raise ValueError(_no_bracks(f"The feature 'Microarchitecture must have one of these values: {legit_uarch_vals}, but was {features['Microarchitecture']}"))
  elif not isinstance(features["Clockspeed"], int) and not isinstance(features['Clockspeed'], float):
    raise TypeError(f"The feature 'Clockspeed' must be either an int or a float, but was a {type(features['Clockspeed']).__name__}")
  elif features["Clockspeed"] <= 0:
    raise ValueError(f"The feature 'Clockspeed' must be positive, but was {features['Clockspeed']}")
  elif not isinstance(features["Year"], int) and not isinstance(features["Year"], float):
    raise TypeError(f"The feature 'Year' must be either an int or a float, but was a {type(features['Year']).__name__}")
  elif not isinstance(features["Total Cores"], int):
    raise TypeError(f"The feature 'Total Cores' must be an int, but was a {type(features['Total Cores']).__name__}")
  elif features["Total Cores"] <= 0:
    raise ValueError(f"The feature 'Total Cores' must be positive, but was {features['Total Cores']}")
  elif not isinstance(features["Fraction of Cores that are Accelerators"], int) and not isinstance(features["Fraction of Cores that are Accelerators"], float):
    raise TypeError(f"The feature 'Fraction of Cores that are Accelerators' must be either an int or a float, but was a " +
    f"{type(features['Fraction of Cores that are Accelerators']).__name__}")
  elif features["Fraction of Cores that are Accelerators"] < 0 or features["Fraction of Cores that are Accelerators"] > 1:
    raise ValueError(f"The feature 'Fraction of Cores that are Accelerators' must occupy the interval [0, 1], but was " + 
                     f"{features['Fraction of Cores that are Accelerators']}")
  
  # making a prediction
  train_raw = None
  obs_raw = None
  model = None
  scaler = None

  if model_evaluation_methodology == "ToP" and dependent_variable == "Log(Rmax)":
    #model is gbt with default hyperparameters, RobustScaler, and two_prev
    list_17 = standardize_dataset(all_datasets[16])
    list_18 = standardize_dataset(all_datasets[17])
    train_raw = list_17.append(list_18, ignore_index=True) # two prev means use two previous datasets
    model = GradientBoostingRegressor()
    scaler = RobustScaler
  elif model_evaluation_methodology == "ToP" and dependent_variable == "Log(Efficiency)":
    train_raw = standardize_dataset(all_datasets[17])
    model = GradientBoostingRegressor(max_depth=2)
    scaler = RobustScaler
  elif model_evaluation_methodology == "ToA" and dependent_variable == "Log(Rmax)":
    train_raw = pd.DataFrame()
    for dataset in all_datasets:
      train_raw = train_raw.append(standardize_dataset(dataset), ignore_index=True)
    model = RandomForestRegressor(n_estimators=1000)
    scaler = RobustScaler
  elif model_evaluation_methodology == "ToA" and dependent_variable == "Log(Efficiency)":
    train_raw = pd.DataFrame()
    for dataset in all_datasets:
      train_raw = train_raw.append(standardize_dataset(dataset), ignore_index=True)
    model = LGBMRegressor()
    scaler = MinMaxScaler
  else:
    raise ValueError("Invalid model evaluation methodology or dependent variable")

  accel_cores = features["Fraction of Cores that are Accelerators"] * features["Total Cores"]
  obs_dict = {
      "Architecture": features["Architecture"],
      "Processor Technology": features["Microarchitecture"], # cleaning code expects microarchitecture to be in the processor technology field
      "Year": features["Year"],
      "Total Cores": features["Total Cores"],
      "Processor": f"XXXXXX 10C {features['Clockspeed']}GHz", # cleaning code expects the clockspeed to be embedded in the processor field
      "Accelerator/Co-Processor Cores": accel_cores,
      "Rmax [TFlop/s]": 1, # placeholder for observation dependent variable (which isn't known)
      "Power Effeciency [GFlops/Watts]": 1
  }
  obs_raw = pd.DataFrame(obs_dict, index=[0])
  obs_raw = standardize_dataset(obs_raw)
  
  if dependent_variable == "Log(Rmax)":
    train_clean = clean_data_dep_var_rmax(train_raw, training_df=train_raw, is_training_set=True, exclude_dupes=False)
    obs_clean = clean_data_dep_var_rmax(obs_raw, training_df=train_raw, is_training_set=False, exclude_dupes=False)
  else:
    train_clean = clean_data_dep_var_efficiency(train_raw, training_df=train_raw, is_training_set=True, exclude_dupes=False)
    obs_clean = clean_data_dep_var_efficiency(obs_raw, training_df=train_raw, is_training_set=False, exclude_dupes=False)

  train_x, train_y, test_x, _ = normalize_and_split(train_clean, obs_clean, normalizer=scaler)
  model.fit(train_x, train_y)
  prediction = model.predict(test_x)[0]
  return prediction

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

prediction = predict_using_model(model_evaluation_methodology, dependent_variable, features)
print(f'{dependent_variable} prediction: {round(prediction, 3)}')
