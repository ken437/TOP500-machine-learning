"""
training set size selection strategy that always
returns a size of 1
@param test_pos: position of the test set
@return: recommended train set size (in files)
"""
def one_prev(test_pos):
  return 1

"""
training set size selection strategy that always
returns a size of 2, unless the test set is 
dataset #2, in which case it returns 1
@param test_pos: position of the test set
@return: recommended train set size (in files)
"""
def two_prev(test_pos):
  if test_pos == 2:
    return 1
  else:
    return 2

"""
training set size selection strategy that always
returns a size of 3, unless the test set is before 
dataset #4, in which case it uses all previous datasets
@param test_pos: position of the test set
@return: recommended train set size (in files)
"""
def three_prev(test_pos):
  if test_pos >= 4:
    return 3
  else:
    return test_pos - 1

"""
training set size selection strategy that always
returns a size of 4, unless the test set is before 
dataset #5, in which case it uses all previous datasets
@param test_pos: position of the test set
@return: recommended train set size (in files)
"""
def four_prev(test_pos):
  if test_pos >= 5:
    return 4
  else:
    return test_pos - 1

"""
training set size selection strategy that 
always includes all previous datasets
@param test_pos: position of the test set
@return: recommended train set size (in files)
"""
def all_prev(test_pos):
  return test_pos - 1

"""
training set size selection strategy that 
includes the most recent half of the previous
datasets
@param test_pos: position of the test set
@return: recommended train set size (in files)
"""
def half_prev(test_pos):
  return test_pos // 2

"""
training set size selection strategy that 
includes the most recent third of the previous
datasets
@param test_pos: position of the test set
@return: recommended train set size (in files)
"""
def third_prev(test_pos):
  if test_pos == 2:
    return 1
  else:
    return test_pos // 3

ALL_TRAIN_SET_SELECT_STRATS = [one_prev, two_prev, three_prev, four_prev, all_prev, half_prev, third_prev]
