import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import model as md
columns = ["vendor_name", "Model_Name",
           "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
df = pd.read_csv('machine.data', names=columns)
print("order")
print(df.head())

# Data clean up
my_df = df.copy()

# Shuffle the data 
print("shuffled")
my_df = my_df.sample(frac=1).reset_index(drop=True)
print(my_df.head())

#One-hot divide 
# dummies = pd.get_dummies(my_df.vendor_name,prefix="maker")
# my_df = pd.concat([my_df,dummies],axis=1)

## clean useless attributes
del my_df["Model_Name"]
del my_df["vendor_name"]

# normalalize data
# my_df = md.normalize(my_df)

# Check if data is correct
print("Afteer One-hot")
print(my_df.head())

# Scatter plot the data
# pd.plotting.scatter_matrix(
#     my_df[["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]])
# Divide the data in x and y
# my_df_x = my_df[["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "ERP"]]
my_df_y = my_df["PRP"]
del my_df["PRP"]
my_df_x = my_df.copy()
print(my_df_y.head())
print(my_df_x.head())

# Divide the data in into training and test set
my_df_train_x=my_df_x[0:180]
my_df_test_x=my_df_x[180:]

# Divide the target in into training and test set
my_df_train_y=my_df_y[0:180]
my_df_test_y=my_df_y[180:]

# Define params for each x in my_df_x
my_params = [0] * len(my_df_x.columns)
# define epochs, and learning rate
epochs = 1000
rate = 0.05
print(my_params)

my_df_train_x.reset_index(drop=True)
my_df_train_y.reset_index(drop=True)
pp = md.train(my_params,my_df_train_x,my_df_train_y,epochs,rate)
# predictions = md.predict(my_params,my_df_test_x)


# To do 
# d implement Gradient descent
# d implement hypothesis
# d implement normaization (no big numbers it is now small nums ex 522 = .52)
# d Run by epochs
# - Test
# - Predict user data




