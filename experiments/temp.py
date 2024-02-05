# create 2 csv of just 3 stocks; stocks are: AAPL, IBM, MCD
# one csv contains the stock prices, the other contains the stock changes

import pandas as pd

# Assume the file paths are provided or known
file_paths = ['experiments\AAPL.csv', 'experiments\IBM.csv', 'experiments\MCD.csv']
stock_names = ['AAPL', 'IBM', 'MCD']

# Initialize an empty list to store the DataFrames with percentage change
percentage_change_dfs = []

# Load the full dataset
full_data = pd.read_csv("experiments/data/stocksPrices.csv")

# Select the first 249 trading days as training data
training_data = full_data.iloc[:249]

# now select all the other rows(exclude the first 249 rows) as testing data
testing_data = full_data.iloc[249:]

# Write the testing data to a new CSV file
testing_data.to_csv("experiments/data/testingDatasetStocksPrices.csv", index=False)
