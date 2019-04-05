
import pandas, numpy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection


def get_matrix_with_all_continuous_features(dataset, dataset_test):
    """ This method takes training and test dataset, converts all the categorical features to continuous ones,
        using OneHotEncoder, and returns encoded dataset. """

    # drop unnecessary columns
    dataset_test.drop('id', axis=1, inplace=True)

    # # to print all rows and columns
    # pandas.set_option('display.max_rows', None)
    # pandas.set_option('display.max_columns', None)

    # drop the first column 'id' since it just has serial numbers
    dataset = dataset.iloc[:, 1:]
    cols = dataset.columns  # names of all the columns

    # variable to hold the list of variables for an attribute in the train and test data
    labels = []

    for i in range(0, 116):
        train = dataset[cols[i]].unique()
        test = dataset_test[cols[i]].unique()

        list_from_sets = list(set(train) | set(test))

        labels.append(list_from_sets)

    del dataset_test

    # one hot encode all categorical attributes
    cats = []
    for i in range(0, 116):
        # label encode
        label_encoder = LabelEncoder()
        label_encoder.fit(labels[i])
        feature = label_encoder.transform(dataset.iloc[:, i])
        feature = feature.reshape(dataset.shape[0], 1)

        # one hot encode
        onehot_encoder = OneHotEncoder(sparse=False, n_values=len(labels[i]))
        feature = onehot_encoder.fit_transform(feature)
        cats.append(feature)

    # make a 2D array from a list of 1D arrays
    encoded_cats = numpy.column_stack(cats)

    # # print the shape of the encoded data
    # print(encoded_cats.shape)

    # concatenate encoded attributes with continuous attributes
    dataset_encoded = numpy.concatenate((encoded_cats, dataset.iloc[:, 116:].values), axis=1)

    del cats
    del feature
    del dataset
    del encoded_cats

    # print(dataset_encoded.shape)
    return dataset_encoded


def get_training_and_validation_sets(encoded_dataset, random_seed, test_size):
    """ This method splits encoded dataset into chunks of training and validation sets randomly. """

    # get the number of rows and columns
    n_rows, n_cols = encoded_dataset.shape

    # create an array which has indexes of columns
    i_cols = []
    for i in range(0, n_cols-1):
        i_cols.append(i)

    # Y is the target column, X has the rest
    X = encoded_dataset[:, 0:(n_cols-1)]
    Y = encoded_dataset[:, (n_cols-1)]
    del encoded_dataset

    # split the data into chunks
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X,Y, test_size=test_size,random_state=random_seed)
    del X
    del Y

    X_all = []

    # add this version of X to the list
    n = "All"

    # X_all.append([n, X_train,X_val,i_cols])
    X_all.append([n, i_cols])

    return X_train, X_val, Y_train, Y_val, X_all
