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

import datetime
import random

from configurations import *

sns.set()
sns.set(font_scale=1.5)

boolUniformlyDistributedDataset = True
percentageOfRemovedDays = 10


# code to make the dataset not uniformly distributed
def removeRandomDays(dailyChangeReturnDataset, D):
    '''
    This function removes a percentage of days from the dataset and interpolates the returns of the removed days
    returnDataset: dataframe of returnDataset
    D: percentage of days to eliminate randomly from the dataset:10 = 10% of the days are eliminated
    '''

    # calculate the size of the dataset(so the lenght of the column)
    datasetSize = len(dailyChangeReturnDataset.index)
    number_of_days_to_eliminate = int(datasetSize * D / 100)

    # Define the range of indices that can be removed; avoid the first and last days
    valid_indices = list(range(2, datasetSize - 2)) # Randomly select a group of indices to remove

    #Randomly select a group of indices to remove
    indices_to_remove = sorted(random.sample(valid_indices, number_of_days_to_eliminate))
    print("len of indices to remove and interpolate: " + str(len(indices_to_remove)))
    
    # Create a copy of the DataFrame to perform interpolation
    interpolatedReturns = dailyChangeReturnDataset.copy()

    # Interpolate the returns using linear interpolation method
    interpolatedReturns.iloc[indices_to_remove] = np.nan
    interpolatedReturns = interpolatedReturns.interpolate(method='linear', axis=0, limit_area='inside')
    # limit the number of decimals to 6
    interpolatedReturns = interpolatedReturns.round(6)

    # save the interpolated dataset in a csv file
    interpolatedReturns.to_csv("interpolatedReturns.csv")
    
    return interpolatedReturns


# here i select the correct configuration for the test part of every predictor
if numberOfAssets == 3:
    predictorsConfiguration = predictorConfigurations3

elif numberOfAssets == 6:
    predictorsConfiguration = predictorConfigurations6

elif numberOfAssets == 9:
    predictorsConfiguration = predictorConfigurations9

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

if not boolUniformlyDistributedDataset:

    # if i consider the non-uniformly distributed dataset, i have to apply the linear interpolation to the data to fill the missing values;
    # so i will have not the original dataset with real values, but a dataset with interpolated values
    stocksPercentageChangeReturn = removeRandomDays(uniformlyDistributedReturns, percentageOfRemovedDays) 
    print("returns dataframe dimention after interpolating random days: " + str(stocksPercentageChangeReturn.shape))

    originalDatasetVolatility = uniformlyDistributedReturns.std()
    print("original dataset volatility: " + str(originalDatasetVolatility))

    interterpolatedDatasetVolatility = stocksPercentageChangeReturn.std()
    print("interpolated dataset volatility: " + str(interterpolatedDatasetVolatility))

    # now i have to modify also the training, validation and test datasets; not only the complete dataset because otherwise i will have the full dataset with interpolated values
    # and the training, validation and test datasets with real values

    # 70% training
    trainingDataWithPrices = stocksPrices.loc[:date_70_percent]
    trainingDataWithPercentageChange = stocksPercentageChangeReturn.loc[:date_70_percent]

    # Adjust the start date for the validation set to exclude the last date of the training set
    validation_start_date = date_70_percent + BDay(1)

    # 20% validation
    validationDataWithPrices = stocksPrices.loc[validation_start_date:date_90_percent]
    validationDataWithPercentageChange = stocksPercentageChangeReturn.loc[validation_start_date:date_90_percent]

    # Adjust the start date for the test set to exclude the last date of the validation set
    test_start_date = date_90_percent + BDay(1)

    # 10% test
    testDataWithPrices = stocksPrices.loc[test_start_date:]
    testDataWithPercentageChange = stocksPercentageChangeReturn.loc[test_start_date:]


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