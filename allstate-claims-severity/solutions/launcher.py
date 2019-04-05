
import pandas
from solutions import preprocessing
from solutions.models import linear

import warnings
warnings.filterwarnings('ignore')

dataset = pandas.read_csv("../data/train.csv")
dataset_test = pandas.read_csv("../data/test.csv")

# save the id's for submission file
ID = dataset_test['id']

continuous_features_dataset = preprocessing.get_matrix_with_all_continuous_features(dataset, dataset_test)

seed = 0  # use a common seed in all experiments so that same chunk is used for validation
test_set_fraction = 0.1

X_train, X_val, Y_train, Y_val, X_all = preprocessing.get_training_and_validation_sets(continuous_features_dataset,
                                                                                       random_seed=seed,
                                                                                       test_size=test_set_fraction)
# list of combinations
combinations = []

# dictionary to store the mean absolute errors for all algorithms
maes = []

linear.evaluate_linear_model(X_train, X_val, Y_train, Y_val, X_all)



