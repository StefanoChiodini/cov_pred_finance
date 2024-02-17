# in this file i will unify all the csv files into one; so there will be the file with prices and the file with the percentage change

import pandas as pd

file_paths = ['experiments/data/stokcsData/AAPL.csv', 'experiments/data/stokcsData/IBM.csv', 'experiments/data/stokcsData/MCD.CSV', 
             'experiments/data/stokcsData/KO.csv', 'experiments/data/stokcsData/PEP.csv', 'experiments/data/stokcsData/JNJ.csv',]

# Replace these file paths with the actual paths of your CSV files
stock_names = ['AAPL', 'IBM', 'MCD', 'KO', 'PEP', 'JNJ']

# Initialize an empty list to store the DataFrames with percentage change
percentage_change_dfs = []

# Loop through each file path and stock name
for file_path, stock_name in zip(file_paths, stock_names):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Calculate the percentage change according to the provided formula
    df['Pct_Change'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) 
    
    # Truncate the percentage change to 6 digits
    df['Pct_Change'] = df['Pct_Change'].round(6)
    
    # Extract the Date and percentage change columns, and rename the Pct_Change column
    df = df[['Date', 'Pct_Change']].rename(columns={'Pct_Change': stock_name})
    
    # Append the DataFrame to the list
    percentage_change_dfs.append(df)

# Merge all dataframes on the 'Date' column
merged_pct_change_df = percentage_change_dfs[0]
for df in percentage_change_dfs[1:]:
    merged_pct_change_df = merged_pct_change_df.merge(df, on='Date', how='outer')

# Sort the data by date to ensure the order is correct
merged_pct_change_df.sort_values(by='Date', inplace=True)

# Save the merged dataframe to a new CSV file
merged_pct_change_df.to_csv('experiments/data/sixStocksPortfolioPercentageChange.csv', index=False)

print('Merged percentage change CSV with 6 digit truncation created successfully.')



file_paths = ['experiments/data/stokcsData/AAPL.csv', 'experiments/data/stokcsData/IBM.csv', 'experiments/data/stokcsData/MCD.CSV', 
             'experiments/data/stokcsData/KO.csv', 'experiments/data/stokcsData/PEP.csv', 'experiments/data/stokcsData/JNJ.csv',
             'experiments/data/stokcsData/ORCL.csv', 'experiments/data/stokcsData/PFE.csv', 'experiments/data/stokcsData/WMT.csv']

# Replace these file paths with the actual paths of your CSV files
stock_names = ['AAPL', 'IBM', 'MCD', 'KO', 'PEP', 'JNJ', 'ORCL', 'PFE', 'WMT']

# Initialize an empty list to store the DataFrames with percentage change
percentage_change_dfs = []

# Loop through each file path and stock name
for file_path, stock_name in zip(file_paths, stock_names):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Calculate the percentage change according to the provided formula
    df['Pct_Change'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) 
    
    # Truncate the percentage change to 6 digits
    df['Pct_Change'] = df['Pct_Change'].round(6)
    
    # Extract the Date and percentage change columns, and rename the Pct_Change column
    df = df[['Date', 'Pct_Change']].rename(columns={'Pct_Change': stock_name})
    
    # Append the DataFrame to the list
    percentage_change_dfs.append(df)

# Merge all dataframes on the 'Date' column
merged_pct_change_df = percentage_change_dfs[0]
for df in percentage_change_dfs[1:]:
    merged_pct_change_df = merged_pct_change_df.merge(df, on='Date', how='outer')

# Sort the data by date to ensure the order is correct
merged_pct_change_df.sort_values(by='Date', inplace=True)

# Save the merged dataframe to a new CSV file
merged_pct_change_df.to_csv('experiments/data/nineStocksPortfolioPercentageChange.csv', index=False)

print('Merged percentage change CSV with 6 digit truncation created successfully.')