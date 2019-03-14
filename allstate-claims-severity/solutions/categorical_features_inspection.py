# count of each label in each category

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

# drop the first column 'id' since it just has serial numbers
dataset = dataset.iloc[:,1:]
cols = dataset.columns  # names of all the columns

# plot count plot for all attributes in a 29x4 grid
n_cols = 4
n_rows = 29
for i in range(n_rows):
    fg, ax = plt.subplots(nrows=1, ncols=n_cols, sharey=True, figsize=(12, 8))
    for j in range(n_cols):
        sns.countplot(x=cols[i*n_cols+j], data=dataset, ax=ax[j])

# helps to see how many values each of the categorical feature has
plt.show()