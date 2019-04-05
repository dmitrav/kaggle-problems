
# evaluation of various combinations of LinearRegression

# import the library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import numpy


def evaluate_linear_model(X_train, X_val, Y_train, Y_val, X_all):

    # set the base model
    model = LinearRegression(n_jobs=-1)

    # accuracy of the model using all features
    for name, i_cols_list in X_all:
        model.fit(X_train[:, i_cols_list], Y_train)

        # result = mean_absolute_error(numpy.nan_to_num(numpy.expm1(Y_val)),
        #                              numpy.nan_to_num(numpy.expm1(model.predict(X_val[:, i_cols_list]))))
        # TODO: figure out why to use expm1 and why it's not working here
        # TODO: check if expm1 works for other models
        result = mean_absolute_error(Y_val, model.predict(X_val[:, i_cols_list]))

        print(name + " %s" % result)

    return result
