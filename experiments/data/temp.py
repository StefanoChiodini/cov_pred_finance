import pandas as pd
path = "C:\\Users\\chiod\\Desktop\\MyData\\universita\\tesi\\openSourceImplementations\\cov_pred_finance\\experiments\\data\\stockDataForResiduals\\"
# List of your CSV file paths
file_paths = [path + "AAPL (1).csv", path + "IBM (1).csv", path + "MCD (1).csv", path + "KO (1).csv", path + "PEP (1).csv", path + "JNJ (1).csv", path + "ORCL (1).csv", path + "PFE (1).csv", path + "WMT (1).csv"]
stock_names = ['AAPL', 'IBM', 'MCD', 'KO', 'PEP', 'JNJ', 'ORCL', 'PFE', 'WMT']

# Load the first CSV to start the merging process
df_merged = pd.read_csv(file_paths[0])[['Date', 'Close']].rename(columns={'Close': f'{stock_names[0]}'})

# Loop through the remaining CSV files and merge
for file_path, stock_name in zip(file_paths[1:], stock_names[1:]):
    df_temp = pd.read_csv(file_path)[['Date', 'Close']].rename(columns={'Close': f'{stock_name}'})
    df_merged = pd.merge(df_merged, df_temp, on='Date', how='outer')

# Save the merged DataFrame to a new CSV file
df_merged.to_csv('consolidated_close_prices.csv', index=False)

csv_file_path = 'consolidated_close_prices.csv'

# Load your existing CSV file
df = pd.read_csv(csv_file_path)

# Calculate the daily percentage change for each stock
df_pct_change = df.set_index('Date').pct_change()

# truncate the percentage changes to 6 decimal places
df_pct_change = df_pct_change.round(6)

# Reset the index so 'Date' becomes a column again
df_pct_change.reset_index(inplace=True)

# Save the resulting DataFrame with percentage changes to a new CSV file
df_pct_change.to_csv('daily_percentage_changes.csv', index=False)