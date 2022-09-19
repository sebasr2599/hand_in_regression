import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
my_df.head()

# One-hot divide 
del my_df["Model_Name"]
# del my_df["vendor_name"]
# # df.drop(columns=["Model_Name", 'CACH'])
# Check if data is correct
print(my_df.head())

# Scatter plot the data
pd.plotting.scatter_matrix(
    my_df[["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]])
# Divide the data in x and y
my_df_x = my_df[["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "ERP"]]
my_df_y = my_df["PRP"]
print(my_df_y.head())
print(my_df_x.head())

# Divide the data in x and y into training and test set
# my_df_train=my_df[0:180]
# my_df_=my_df[0:180]
