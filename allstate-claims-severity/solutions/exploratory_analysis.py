
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

# Read raw data from the file

import pandas, numpy
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pandas.read_csv("../data/train.csv")
dataset_test = pandas.read_csv("../data/test.csv")

# save the id's for submission file
ID = dataset_test['id']

# drop unnecessary columns
dataset_test.drop('id', axis=1, inplace=True)

# print all rows and columns
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

# display the first five rows to get a feel of the data
if True:
    print(dataset.head(5))

# size of the data frame
if False:
    print(dataset.shape)

# drop the first column 'id' since it just has serial numbers
dataset = dataset.iloc[:,1:]

# statistical description
if False:
    print(dataset.describe())

# skewness of the distribution (values close to 0 show less skew)
if False:
    print(dataset.skew())

data = dataset.iloc[:, 116:]  # create a data frame with only continuous features
size = 15  # number of features considered
cols = data.columns  # get names

# violin plots for continuous features
if False:

    # plot violin for all attributes in a 7x2 grid
    n_cols = 2
    n_rows = 7

    for i in range(n_rows):
        fg, ax = plt.subplots(nrows=1, ncols=n_cols, figsize=(12, 8))
        for j in range(n_cols):
            sns.violinplot(y=cols[i*n_cols+j], data=dataset, ax=ax[j])

    plt.show()

# violin plot for log-transformed loss
if False:
    # log1p function applies log(1+x) to all elements of the column
    dataset["loss"] = numpy.log1p(dataset["loss"])

    # visualize the transformed column
    sns.violinplot(data=dataset,y="loss")

    plt.show()


# inspect correlations among continuous variables
if False:
    # calculate pearson co-efficient for all combinations
    data_corr = data.corr()

    # set the threshold to select only highly correlated attributes
    threshold = 0.5

    # list of pairs along with correlation above threshold
    corr_list = []

    # search for the highly correlated pairs
    for i in range(0, size):  # for 'size' features
        for j in range(i+1,size):  # avoid repetition
            if (threshold <= data_corr.iloc[i, j] < 1) or (-1 < data_corr.iloc[i, j] <= -threshold):

                corr_list.append([data_corr.iloc[i, j], i, j])  # store correlation and columns index

    # sort to show higher ones first
    s_corr_list = sorted(corr_list, key=lambda x: abs(x[0]), reverse=True)

    # print correlated features
    for v, i, j in s_corr_list:
        print("%s and %s = %.2f" % (cols[i], cols[j], v))

    if False:

        # scatter plot of only the highly correlated pairs
        for v, i, j in s_corr_list:
            sns.pairplot(dataset, size=6, x_vars=cols[i], y_vars=cols[j])

        # visual inspection helps to figure out which (highly correlated) features could be safely removed
        plt.show()
