import pandas as pd
from pandas.tseries.offsets import BDay

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import matplotlib.dates as md
from tqdm import trange
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from utils.portfolio_backtests import *
from utils.trading_model import *
from utils.experiment_utils import *
from utils.portfolio_backtests import MeanVariance

from predictorsImplementation import * # this file contains the implementation of the predictors ( one function implementation for each predictor)

import datetime
import random

import json
import subprocess
import os
from dotenv import load_dotenv
load_dotenv()

# code for plotting and compare inside a unique chart prices and volatilities for mgarch predictions

def plot_prices_volatilities_for_predictor(stock_prices, real_volatility, real_volatility_startDate, real_volatility_endDate, predictorVolatility, asset_name, predictor_name):
    '''
    Function to plot prices and volatilities for a specific predictor
    '''
    # filter the real volatility between the start and end date
    real_volatility_startDate = pd.to_datetime(real_volatility_startDate)
    real_volatility_endDate = pd.to_datetime(real_volatility_endDate)

    # Correct way to filter using & operator and parentheses
    real_volatility_filtered = real_volatility[(real_volatility.index >= real_volatility_startDate) & (real_volatility.index <= real_volatility_endDate)]
    predictorVolatility = predictorVolatility[(predictorVolatility.index >= real_volatility_startDate) & (predictorVolatility.index <= real_volatility_endDate)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), sharex=True)

    # Plot stock prices
    ax1.plot(stock_prices[asset_name], label=f'{asset_name} Price', color='green')
    ax1.set_title(f'{asset_name} Stock Prices')
    ax1.set_ylabel('Price(dollars)')
    ax1.legend(loc='upper left')
    
    # Plot real and rolling window volatilities
    ax2.plot(real_volatility_filtered, label=f'Real {asset_name} Volatility', color='blue')
    ax2.plot(predictorVolatility, label=f'{predictor_name} {asset_name} Volatility', color='orange', linestyle='--')
    ax2.set_title(f'{asset_name} Volatility: Real vs {predictor_name}')
    ax2.set_xlabel('Time(days)')
    ax2.set_ylabel('Volatility(%)')
    ax2.legend(loc='upper left')

    # Set x-axis limits to match the start and end dates
    ax1.set_xlim(left=real_volatility_startDate, right=real_volatility_endDate)
    ax2.set_xlim(left=real_volatility_startDate, right=real_volatility_endDate)

    # Adding vertical lines for specific events
    ax1.axvline(pd.Timestamp('2020-02-24'), color='gray', linestyle='--', lw=2)  # COVID start
    ax1.axvline(pd.Timestamp('2022-02-24'), color='red', linestyle='--', lw=2)  # Ukraine War start
    
    # Adding vertical lines for specific events
    ax2.axvline(pd.Timestamp('2020-02-24'), color='gray', linestyle='--', lw=2)  # COVID start
    ax2.axvline(pd.Timestamp('2022-02-24'), color='red', linestyle='--', lw=2)  # Ukraine War start
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_volatility(predictorDict, start_date, end_date, predictor_name):
    '''
    this function is for plotting assets volatilities
    '''

    predictor_volatilities = {} # this dictionary will contain the volatilities of the 3 assets for every day with the same key of the predictorDict dictionary(the timestamp)
    for date, cov_matrix in predictorDict.items():
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        predictor_volatilities[date] = pd.DataFrame(data = volatilities, index = cov_matrix.index, columns = ["volatility"])

    # now predictor_volatilities is a dictionary that contains the real volatilities of the 3 assets for every day with the same key of the prescientDict dictionary(the timestamp)

    # filter the dictionary between the start and end date
    predictor_volatilities = {k: v for k, v in predictor_volatilities.items() if k >= start_date and k <= end_date}

    # now separate the real volatilities of the 3 assets in 3 different dataframes
    volatility_dict_aapl, volatility_dict_ibm, volatility_dict_mcd = {}, {}, {}

    for date, volatilities in predictor_volatilities.items():
        volatility_dict_aapl[date] = volatilities.loc[7]["volatility"] # 7 is the PERMCO code of AAPL
        volatility_dict_ibm[date] = volatilities.loc[20990]["volatility"] # 20990 is the PERMCO code of IBM
        volatility_dict_mcd[date] = volatilities.loc[21177]["volatility"] # 21177 is the PERMCO code of MCD

    # check if dictionaries are empty or not
    if not volatility_dict_aapl or not volatility_dict_ibm or not volatility_dict_mcd:
        raise ValueError("No volatilities found for the specified dates.")
    
    # Convert the dictionaries to DataFrames for easier manipulation and plotting
    df_volatility_aapl = pd.DataFrame(list(volatility_dict_aapl.items()), columns=['Date', 'AAPL Volatility'])
    df_volatility_ibm = pd.DataFrame(list(volatility_dict_ibm.items()), columns=['Date', 'IBM Volatility'])
    df_volatility_mcd = pd.DataFrame(list(volatility_dict_mcd.items()), columns=['Date', 'MCD Volatility'])

    # Set the 'Date' column as the index
    df_volatility_aapl.set_index('Date', inplace=True)
    df_volatility_ibm.set_index('Date', inplace=True)
    df_volatility_mcd.set_index('Date', inplace=True)

    # Plot the real volatilities of the 3 assets
    plt.figure(figsize=(18, 11))
    plt.plot(df_volatility_aapl, label='AAPL Volatility')
    plt.plot(df_volatility_ibm, label='IBM Volatility')
    plt.plot(df_volatility_mcd, label='MCD Volatility')
    plt.legend()
    plt.title(f"Volatilities of the 3 assets with {predictor_name} Predictor")
    plt.xlabel("Time(days)")
    plt.ylabel("Volatility(%)")

    # check if the 2 dates(2020-02-24 and 2022-02-24) are in the range of the dataframe, if yes add vertical lines and annotations
    if pd.Timestamp('2020-02-24') in df_volatility_aapl.index:
        # Adding vertical lines for specific events
        plt.axvline(pd.Timestamp('2020-02-24'), color='gray', linestyle='--', lw=2)  # COVID start
        # Annotations for the events
        plt.text(pd.Timestamp('2020-02-24'), plt.ylim()[1], 'COVID', horizontalalignment='center', color='gray')

    if  pd.Timestamp('2022-02-24') in df_volatility_aapl.index:
        plt.axvline(pd.Timestamp('2022-02-24'), color='orange', linestyle='--', lw=2)  # Ukraine War start
        plt.text(pd.Timestamp('2022-02-24'), plt.ylim()[1], 'Ukraine War', horizontalalignment='center', color='orange')

    # Set x-axis between the first and last day of a dataframe for example aapl dataframe
    plt.xlim(left=df_volatility_aapl.index[0], right=df_volatility_aapl.index[-1])

    return df_volatility_aapl, df_volatility_ibm, df_volatility_mcd, volatility_dict_aapl, volatility_dict_ibm, volatility_dict_mcd


sns.set()
sns.set(font_scale=1.5)
prescientDict = {}

numberOfAssets = int(os.getenv("NUMBER_OF_ASSETS"))

file_path_first_part = os.getenv("FILE_PATH")
path_to_json_file_configurations = file_path_first_part + 'experiments/configurations.json'

# Load the configurations from the JSON file
with open(path_to_json_file_configurations, 'r') as config_file:
    allConfigurations = json.load(config_file)

# Convert numberOfAssets to a string since JSON keys are stored as strings
key = str(numberOfAssets)

# Select the correct configuration based on the number of assets
if key in allConfigurations:
    predictorsConfiguration = allConfigurations[key]
else:
    raise ValueError(f"No configuration found for {numberOfAssets} assets.")


stocksPrices = pd.read_csv('data/' + str(numberOfAssets) + 'StocksPortfolios.csv', index_col=0, parse_dates=True)
stocksPercentageChangeReturn = pd.read_csv('data/' + str(numberOfAssets) + 'StocksPortfolioPercentageChange.csv', index_col=0, parse_dates=True)
uniformlyDistributedReturns = stocksPercentageChangeReturn.copy() # this is a copy of the original dataset returns; i will use this to make it non-uniformly distributed

# Risk-free rate
FF = pd.read_csv('data/ff5.csv', index_col=0, parse_dates=True)
rf_rate = pd.DataFrame(FF.loc[:,"RF"])
rf_rate.index = pd.to_datetime(rf_rate.index, format='%Y%m%d')

# i have 13 years of trading data; 3273 days; now i will split the dataset into 70% training, 20% validation and 10% test

total_days = len(stocksPrices)
date_70_percent = stocksPrices.index[int(total_days * 0.7)]
date_90_percent = stocksPrices.index[int(total_days * 0.9)]

# 70% training
trainingDataWithPrices = stocksPrices.loc[:date_70_percent]
trainingDataWithPercentageChange = stocksPercentageChangeReturn.loc[:date_70_percent]

# Adjust the start date for the validation set to exclude the last date of the training set
validation_start_date = date_70_percent + BDay(1)

# 20% validation
validationDataWithPrices = stocksPrices.loc[validation_start_date:date_90_percent]
validationDataWithPercentageChange = stocksPercentageChangeReturn.loc[validation_start_date:date_90_percent]

validation_end_date = validationDataWithPrices.index[-1]

# Adjust the start date for the test set to exclude the last date of the validation set
test_start_date = date_90_percent + BDay(1)

# 10% test
testDataWithPrices = stocksPrices.loc[test_start_date:]
testDataWithPercentageChange = stocksPercentageChangeReturn.loc[test_start_date:]

# print the first date and the last date of each dataset
print("First date of training dataset: ", trainingDataWithPrices.index[0])
print("Last date of training dataset: ", trainingDataWithPrices.index[-1])
print("len of training dataset: ", len(trainingDataWithPrices))

print("\nFirst date of validation dataset: ", validationDataWithPrices.index[0])
print("Last date of validation dataset: ", validationDataWithPrices.index[-1])
print("len of validation dataset: ", len(validationDataWithPrices))

print("\nFirst date of test dataset: ", testDataWithPrices.index[0])
print("Last date of test dataset: ", testDataWithPrices.index[-1])
print("len of test dataset: ", len(testDataWithPrices))

print("\ntotal dataset lenght: ", len(stocksPrices))
print("sum of the three datasets: ", len(trainingDataWithPrices) + len(validationDataWithPrices) + len(testDataWithPrices))

print("original returns dataframe dimension: " + str(stocksPercentageChangeReturn.shape))

# Import pickle
with open('data/permco_to_ticker.pkl', 'rb') as f:
    permco_to_ticker = pickle.load(f)

# Create a reverse mapping if necessary
ticker_to_permco = {v: k for k, v in permco_to_ticker.items()}

# Replace ticker symbols with PERMCO codes if the ticker symbol is found in the reverse mapping
columns = [ticker_to_permco.get(col, col) for col in stocksPercentageChangeReturn.columns]
stocksPercentageChangeReturn.columns = columns 

# Repeat for uniformlyDistributedReturns

uniformlyDistributedReturns.columns = columns

# repeat for validation and test dataset
trainingDataWithPercentageChange.columns = columns
validationDataWithPercentageChange.columns = columns
testDataWithPercentageChange.columns = columns

# Plot the returns of the stocks with highlights and annotations
plt.figure(figsize=(14, 7))
plt.plot(stocksPrices)
plt.legend(["AAPL", "IBM", "MCD"])
plt.title("Returns of the stocks")
plt.xlabel("Time(days)")
plt.ylabel("Returns(dollars)")

# Adding vertical lines for specific events
plt.axvline(pd.Timestamp('2020-02-24'), color='gray', linestyle='--', lw=2)  # COVID start
plt.axvline(pd.Timestamp('2022-02-24'), color='orange', linestyle='--', lw=2)  # Ukraine War start

# Annotations for the events
plt.text(pd.Timestamp('2020-02-24'), plt.ylim()[1], 'COVID', horizontalalignment='center', color='gray')
plt.text(pd.Timestamp('2022-02-24'), plt.ylim()[1], 'Ukraine War', horizontalalignment='center', color='orange')

plt.xlim(left=stocksPrices.index[0], right=stocksPrices.index[-1])
plt.show()

# Plot also the percentage change of the stocks with highlights and annotations
plt.figure(figsize=(14, 7))
plt.plot(stocksPercentageChangeReturn)
plt.legend(["AAPL", "IBM", "MCD"])
plt.title("Percentage change of the stocks")
plt.xlabel("Time(days)")
plt.ylabel("daily returns(%)")

# Adding vertical lines for specific events
plt.axvline(pd.Timestamp('2020-02-24'), color='gray', linestyle='--', lw=2)  # COVID start
plt.axvline(pd.Timestamp('2022-02-24'), color='orange', linestyle='--', lw=2)  # Ukraine War start

# Annotations for the events
plt.text(pd.Timestamp('2020-02-24'), plt.ylim()[1], 'COVID', horizontalalignment='center', color='gray')
plt.text(pd.Timestamp('2022-02-24'), plt.ylim()[1], 'Ukraine War', horizontalalignment='center', color='orange')

plt.xlim(left=stocksPercentageChangeReturn.index[0], right=stocksPercentageChangeReturn.index[-1])
plt.show()


# COVARIANCE PREDICTORS

# PRESCIENT(GROUND TRUTH)

# THIS CODE IS CALCULATING THE REAL VOLATILITY

# prescient is a dictionary that contains the covariance matrix calculated using the ewma formula written inside the paper
# the key of the dictionary is the timestamp and the value is the covariance matrix calculated for that day

# The prescient predictor will always use the original dataset, so it will be uniformly distributed; this is because the prescient predictor is used to compare the other predictors
# and we need to have a measure of the real covariance matrix; so this can't be used with the non-uniformly distributed dataset

prescientDict = originalPrescientPredictor(uniformlyDistributedReturns)

    
# print the first 5 elements of the dictionary
for key in list(prescientDict.keys())[:5]:
    print(key, prescientDict[key])


print("dimension of the prescient dictionary: " + str(len(prescientDict)))

# print just the first key and value of the dictionary
print(list(prescientDict.keys())[0])
print(prescientDict[list(prescientDict.keys())[0]])

# print the 60 and value of the dictionary
print(list(prescientDict.keys())[30])
print(prescientDict[list(prescientDict.keys())[30]])

# print the 60 and value of the dictionary
print(list(prescientDict.keys())[60])
print(prescientDict[list(prescientDict.keys())[60]])

# print the 60 and value of the dictionary
print(list(prescientDict.keys())[90])
print(prescientDict[list(prescientDict.keys())[90]])

# print the 60 and value of the dictionary
print(list(prescientDict.keys())[120])
print(prescientDict[list(prescientDict.keys())[120]])

# print the 60 and value of the dictionary
print(list(prescientDict.keys())[150])
print(prescientDict[list(prescientDict.keys())[150]])


# REAL VOLATILITIES
# THIS IS THE VISUALIZATION OF THE REAL VOLAITILITIES OF THE 3 ASSETS

# now calculates/extract the real volatilities of the 3 assets
real_volatilities = {}

# now filter the rw volatilities between the start and end date
real_volatility_startDate = pd.to_datetime('2010-01-04')
real_volatility_endDate = pd.to_datetime('2023-01-03')

df_volatility_aapl, df_volatility_ibm, df_volatility_mcd, volatility_dict_aapl, volatility_dict_ibm, volatility_dict_mcd = plot_volatility(prescientDict, real_volatility_startDate, real_volatility_endDate, 'PRESCIENT')


# EXPANDING WINDOW DICT
# NOW I IMPLEMENT AN EXPANDING WINDOW MODEL FOR EVERY QUARTER

expandingWindowDict = expandingWindowPredictor(uniformlyDistributedReturns)

print("dimension of dataset: " + str(uniformlyDistributedReturns.shape))

print("len of the expanding window dictionary: " + str(len(expandingWindowDict)))

# print just the first key and value of the dictionary
print(list(expandingWindowDict.keys())[0])
print(expandingWindowDict[list(expandingWindowDict.keys())[0]])