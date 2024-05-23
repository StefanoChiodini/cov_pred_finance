import pandas as pd
from pandas.tseries.offsets import BDay

import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import matplotlib.dates as md
import matplotlib.dates as mdates

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


# Mean Loglikelihood
mean_loglikelihood = {
    'RW': [8.1, 17.8, 25.6],
    'EWMA': [8.2, 18.2, 25.6],
    'MGARCH': [8.3, 18.5, 26.3],

    'H_RW': [8.2, 18.2, 25.4],
    'H_EWMA': [8.1, 18.1, 24.9],
    'H_MGARCH': [8.2, 18.4, 26],

    'PRESCIENT': [8.5, 19, 27.7]
}

# Mean Regret
mean_regret = {
    'RW': [0.4, 1.2, 2.1],
    'EWMA': [0.3, 0.8, 2],
    'MGARCH': [0.2, 0.5, 1.4],

    'H_RW': [0.3, 0.8, 2.2],
    'H_EWMA': [0.4, 0.9, 2.7],
    'H_MGARCH': [0.3, 0.7, 1.6],

    'PRESCIENT': [0, 0, 0]
}

# Mean RMSE
mean_rmse = {
    'RW': [0.0135287164, 0.0162867959, 0.0232394294],
    'EWMA': [0.0138895248, 0.0157856632, 0.0221445214],
    'MGARCH': [0.0156442254, 0.0183768807, 0.0247100441],

    'H_RW': [0.0128988004, 0.0152217991, 0.0211461890],
    'H_EWMA': [0.0136928594, 0.0156146511, 0.0218686480],
    'H_MGARCH': [0.0141520314, 0.0162460398, 0.0212515323],

    'PRESCIENT': [0, 0, 0]
}

mean_rmse_aapl = {
    'RW': [0.0030237629, 0.0030237629, 0.0035069682],
    'EWMA': [0.0029049090, 0.0028877435, 0.0030722798],
    'MGARCH': [0.0036536362, 0.0036536362, 0.0039912297],

    'H_RW': [0.0026903362, 0.0027630288, 0.0027872536],
    'H_EWMA': [0.0027650293, 0.0027061155,0.0027270461],
    'H_MGARCH': [0.0030092138, 0.0028697469, 0.0030092138],

    'PRESCIENT': [0, 0, 0]
}

mean_rmse_ibm = {
    'RW': [0.0019626498, 0.0019626498, 0.0018088209],
    'EWMA': [0.0025055011, 0.0018447868, 0.0017988968],
    'MGARCH': [0.0034359721, 0.0034359721, 0.0034859631],

    'H_RW': [0.0017044258, 0.0016245932, 0.0016211082],
    'H_EWMA': [0.0024125094, 0.0019597321, 0.0022463687],
    'H_MGARCH': [0.0023783371, 0.0023390251, 0.0023783371],

    'PRESCIENT': [0, 0, 0]
}

mean_rmse_mcd = {
    'RW': [0.0015576934, 0.0015576934, 0.0018403446],
    'EWMA': [0.0018173996, 0.0015029123, 0.0015136579],
    'MGARCH': [0.0016055425, 0.0016122437, 0.0026588134],

    'H_RW': [0.0017227619, 0.0017025038, 0.0017089481],
    'H_EWMA': [0.0017420476, 0.0016234617, 0.0016817572],
    'H_MGARCH': [0.0019413723, 0.0015212221, 0.0019413723],

    'PRESCIENT': [0, 0, 0]
}


# this function is used to plot the performance of the RW predictor in terms of regret and log-likelihood

def plotLogLikelihoodPerformancePredictor(hyperParameterValues, hyperParameterMeanlogLikelihoodValues, prescientMeanlogLikelihoodValue, predictor_name, hyperParameterName):
    '''
    plot the chart of the mean log-likelihood value for the given 
    '''
    hyperParameterMinValuesLogLikelihood = [] # this list will contain the hyperParameter values that have the max log-likelihood value
    plt.figure()
    plt.plot(hyperParameterValues, hyperParameterMeanlogLikelihoodValues)
    plt.title(f"Mean log-likelihood values of the {hyperParameterName} predictor")
    plt.xlabel(f"{predictor_name} {hyperParameterName}")
    plt.ylabel("Mean log-likelihood")
    plt.title(f"Mean log-likelihood values of the {hyperParameterName} predictor")

    # show also the loglikelihood value of the prescient predictor
    plt.axhline(y=prescientMeanlogLikelihoodValue, color='r', linestyle='-')
    plt.legend([f"{predictor_name} {hyperParameterName}", "Prescient Predictor"])

    # find the rw values that have the max log-likelihood value
    for j in range(len(hyperParameterMeanlogLikelihoodValues)):
        if hyperParameterMeanlogLikelihoodValues[j] == max(hyperParameterMeanlogLikelihoodValues):
            hyperParameterMinValuesLogLikelihood.append(hyperParameterValues[j])

    # these points show the interval of rw values that have the maximum log-likelihood value
    highlightsPoint1 = hyperParameterMinValuesLogLikelihood[0]
    plt.scatter(highlightsPoint1, max(hyperParameterMeanlogLikelihoodValues), color='r')
    #plt.text(highlightsPoint1, max(rwMeanlogLikelihoodValues), f' M: {highlightsPoint1:.1f}\n Log-likelihood: {max(rwMeanlogLikelihoodValues):.4f}', fontsize=9, color='red', ha='center', va='bottom')

    highlightsPoint2 = hyperParameterMinValuesLogLikelihood[-1]
    plt.scatter(highlightsPoint2, max(hyperParameterMeanlogLikelihoodValues), color='r')
    #plt.text(highlightsPoint2, max(rwMeanlogLikelihoodValues), f' M: {highlightsPoint2:.1f}\n Log-likelihood: {max(rwMeanlogLikelihoodValues):.4f}', fontsize=9, color='red', ha='center', va='bottom')
    
    # find the y value coordinate corresponding to the highlightPoint1LogLikelihood and highlightPoint2LogLikelihood
    highlightsPoint1Index = hyperParameterValues.index(highlightsPoint1)
    highlightsPoint2Index = hyperParameterValues.index(highlightsPoint2)

    highlightsPoint1Y = hyperParameterMeanlogLikelihoodValues[highlightsPoint1Index]
    highlightsPoint2Y = hyperParameterMeanlogLikelihoodValues[highlightsPoint2Index]

    print(f"{hyperParameterName}: {highlightsPoint1:.4f}\n Log-likelihood: {highlightsPoint1Y:.4f}")
    print(f"{hyperParameterName}: {highlightsPoint2:.4f}\n Log-likelihood: {highlightsPoint2Y:.4f}")

    # set the x-axis limits
    plt.xlim(left = hyperParameterValues[0], right = hyperParameterValues[-1])

    plt.show()


def plotRegretPerformancePredictor(hyperParameterValues, hyperParameterMeanRegretValues, predictor_name, hyperParameterName):
    '''
    plot the chart of the mean regret values of a single predictor
    '''

    hyperParameterMinValuesRegret = [] # this list will contain the hyperParameter values that have the min regret value
    plt.plot(hyperParameterValues, hyperParameterMeanRegretValues)
    plt.title(f"Mean regret values of the {predictor_name} predictor")
    plt.xlabel(f"{predictor_name} {hyperParameterName}")
    plt.ylabel("Mean regret")
    plt.title(f"Mean regret values of the {predictor_name} predictor")

    # find the rw values that have the min regret value
    for i in range(len(hyperParameterMeanRegretValues)):
        if hyperParameterMeanRegretValues[i] == min(hyperParameterMeanRegretValues):
            hyperParameterMinValuesRegret.append(hyperParameterValues[i])

    # these points show the interval of rw values that have the minimum regret value
    highlightsPoint1 = hyperParameterMinValuesRegret[0]
    plt.scatter(highlightsPoint1, min(hyperParameterMeanRegretValues), color='r')
    #plt.text(highlightsPoint1, min(rwMeanRegretValues), f' rwMemory: {highlightsPoint1:.1f} \n Regret: {min(rwMeanRegretValues):.4f}', fontsize=9, color='red', ha='center', va='bottom')

    highlightsPoint2 = hyperParameterMinValuesRegret[-1]
    plt.scatter(highlightsPoint2, min(hyperParameterMeanRegretValues), color='r')
    #plt.text(highlightsPoint2, min(rwMeanRegretValues), f' rwMemory: {highlightsPoint2:.1f} \n Regret: {min(rwMeanRegretValues):.4f}', fontsize=9, color='red', ha='center', va='bottom')
    
    # find the y value coordinate corresponding to the highlightPoint1Regret and highlightPoint2Regret
    highlightsPoint1Index = hyperParameterValues.index(highlightsPoint1)
    highlightsPoint2Index = hyperParameterValues.index(highlightsPoint2)

    highlightsPoint1Y = hyperParameterMeanRegretValues[highlightsPoint1Index]
    highlightsPoint2Y = hyperParameterMeanRegretValues[highlightsPoint2Index]

    print(f"{hyperParameterName}: {highlightsPoint1:.4f}\n Regret: {highlightsPoint1Y:.4f}")
    print(f"{hyperParameterName}: {highlightsPoint2:.4f}\n Regret: {highlightsPoint2Y:.4f}")

    # set the x-axis limits
    plt.xlim(left = hyperParameterValues[0], right = hyperParameterValues[-1])

    plt.show()


def plotPerformancePredictor(predictorValues, predictorMeanlogLikelihoodValues, prescientMeanlogLikelihoodValue, predictorMeanRegretValues, predictor_name, hyperParameterName):
    '''
    plotting the results of the a predictor expressed in terms of loglikelihood and regret on a single chart
    '''

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the mean log-likelihood values for the predictor
    color = 'tab:blue'
    ax1.set_xlabel(f'{hyperParameterName} values')
    ax1.set_ylabel('Mean log-likelihood', color=color)
    ax1.plot(predictorValues, predictorMeanlogLikelihoodValues, color=color, label=f'Loglikelihood {predictor_name}')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot the loglikelihood value of the prescient predictor
    ax1.axhline(y=prescientMeanlogLikelihoodValue, color='tab:green', linestyle='-', label='Loglikelihood PRESCIENT')

    # Highlight the maximum log-likelihood points
    max_log_likelihood = max(predictorMeanlogLikelihoodValues)
    max_points = [beta for beta, value in zip(predictorValues, predictorMeanlogLikelihoodValues) if value == max_log_likelihood]

    # take just the first and last element of the list
    max_points = [max_points[0], max_points[-1]]

    # scatter the points by writing the memory value and the loglikelihood value on the chart
    for point in max_points:
        ax1.scatter(point, max_log_likelihood, color='green')
        plt.text(point, max_log_likelihood, f' x: {point:.0f}\n y: {max_log_likelihood:.1f}', fontsize=9, color='green', ha='center', va='bottom')

    # Add a second y-axis for the regret values
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Mean regret', color=color)  
    ax2.plot(predictorValues, predictorMeanRegretValues, color=color, label=f'Regret {predictor_name}')
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlight the minimum regret points
    min_regret = min(predictorMeanRegretValues)
    min_points = [beta for beta, value in zip(predictorValues, predictorMeanRegretValues) if value == min_regret]

    # take just the first and last element of the list
    min_points = [min_points[0], min_points[-1]]

    # scatter the points by writing the beta value and the regret value on the chart
    for point in min_points:
        ax2.scatter(point, min_regret, color='red')
        plt.text(point, min_regret, f' x: {point:.0f}\n y: {min_regret:.1f}', fontsize=9, color='red', ha='center', va='bottom')

    # print also the y value of the max_points and min_points
    max_pointsIndex = [predictorValues.index(point) for point in max_points]
    min_pointsIndex = [predictorValues.index(point) for point in min_points]

    max_pointsY = [predictorMeanlogLikelihoodValues[index] for index in max_pointsIndex]
    min_pointsY = [predictorMeanRegretValues[index] for index in min_pointsIndex]

    print("max_pointsX: " + str(max_points))
    print("max_pointsY: " + str(max_pointsY))
    print("min_pointsX: " + str(min_points))
    print("min_pointsY: " + str(min_pointsY))

    # Create the legend, which combines both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Add title
    plt.title(f"Performance of {predictor_name} for different {hyperParameterName} values")

    # Set the x-axis limits
    ax1.set_xlim(left=predictorValues[0], right=predictorValues[-1])

    fig.tight_layout()  # to ensure the right y-label is not slightly clipped
    plt.show()


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

    # AAPL -> 7, IBM -> 20990, MCD -> 21177, KO -> 20468, PEP -> 21384, JNJ -> 21018, ORCL -> 8045, PFE -> 21394, WMT -> 21880

    predictor_volatilities = {} # this dictionary will contain the volatilities of the every assets for every day with the same key of the predictorDict dictionary(the timestamp)
    for date, cov_matrix in predictorDict.items():
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        predictor_volatilities[date] = pd.DataFrame(data = volatilities, index = cov_matrix.index, columns = ["volatility"])

    # now predictor_volatilities is a dictionary that contains the real volatilities of the 3 assets for every day with the same key of the prescientDict dictionary(the timestamp)

    # filter the dictionary between the start and end date
    predictor_volatilities = {k: v for k, v in predictor_volatilities.items() if k >= start_date and k <= end_date}

    # now separate the real volatilities of every assets in single different dataframes
    volatility_dict_aapl, volatility_dict_ibm, volatility_dict_mcd, volatility_dict_ko, volatility_dict_pep, volatility_dict_jnj, volatility_dict_orcl, volatility_dict_pfe, volatility_dict_wmt = {}, {}, {}, {}, {}, {}, {}, {}, {}


    for date, volatilities in predictor_volatilities.items():
        volatility_dict_aapl[date] = volatilities.loc[7]["volatility"] # 7 is the PERMCO code of AAPL
        volatility_dict_ibm[date] = volatilities.loc[20990]["volatility"] # 20990 is the PERMCO code of IBM
        volatility_dict_mcd[date] = volatilities.loc[21177]["volatility"] # 21177 is the PERMCO code of MCD

        if numberOfAssets == 6:
            volatility_dict_ko[date] = volatilities.loc[20468]["volatility"]
            volatility_dict_pep[date] = volatilities.loc[21384]["volatility"]
            volatility_dict_jnj[date] = volatilities.loc[21018]["volatility"]
        
        if numberOfAssets == 9:
            volatility_dict_ko[date] = volatilities.loc[20468]["volatility"]
            volatility_dict_pep[date] = volatilities.loc[21384]["volatility"]
            volatility_dict_jnj[date] = volatilities.loc[21018]["volatility"]
            volatility_dict_orcl[date] = volatilities.loc[8045]["volatility"]
            volatility_dict_pfe[date] = volatilities.loc[21394]["volatility"]
            volatility_dict_wmt[date] = volatilities.loc[21880]["volatility"]


    # check if dictionaries are empty or not
    if not volatility_dict_aapl or not volatility_dict_ibm or not volatility_dict_mcd:
        raise ValueError("No volatilities found for the specified dates.")
    
    # initialize the dataframes here
    df_volatility_aapl, df_volatility_ibm, df_volatility_mcd, df_volatility_ko, df_volatility_pep, df_volatility_jnj, df_volatility_orcl, df_volatility_pfe, df_volatility_wmt = None, None, None, None, None, None, None, None, None
    
    # Convert the dictionaries to DataFrames for easier manipulation and plotting
    df_volatility_aapl = pd.DataFrame(list(volatility_dict_aapl.items()), columns=['Date', 'AAPL Volatility'])
    df_volatility_ibm = pd.DataFrame(list(volatility_dict_ibm.items()), columns=['Date', 'IBM Volatility'])
    df_volatility_mcd = pd.DataFrame(list(volatility_dict_mcd.items()), columns=['Date', 'MCD Volatility'])

    # Set the 'Date' column as the index
    df_volatility_aapl.set_index('Date', inplace=True)
    df_volatility_ibm.set_index('Date', inplace=True)
    df_volatility_mcd.set_index('Date', inplace=True)

    if numberOfAssets == 6:
        df_volatility_ko = pd.DataFrame(list(volatility_dict_ko.items()), columns=['Date', 'KO Volatility'])
        df_volatility_pep = pd.DataFrame(list(volatility_dict_pep.items()), columns=['Date', 'PEP Volatility'])
        df_volatility_jnj = pd.DataFrame(list(volatility_dict_jnj.items()), columns=['Date', 'JNJ Volatility'])

        df_volatility_ko.set_index('Date', inplace=True)
        df_volatility_pep.set_index('Date', inplace=True)
        df_volatility_jnj.set_index('Date', inplace=True)

    if numberOfAssets == 9:
        df_volatility_ko = pd.DataFrame(list(volatility_dict_ko.items()), columns=['Date', 'KO Volatility'])
        df_volatility_pep = pd.DataFrame(list(volatility_dict_pep.items()), columns=['Date', 'PEP Volatility'])
        df_volatility_jnj = pd.DataFrame(list(volatility_dict_jnj.items()), columns=['Date', 'JNJ Volatility'])
        df_volatility_orcl = pd.DataFrame(list(volatility_dict_orcl.items()), columns=['Date', 'ORCL Volatility'])
        df_volatility_pfe = pd.DataFrame(list(volatility_dict_pfe.items()), columns=['Date', 'PFE Volatility'])
        df_volatility_wmt = pd.DataFrame(list(volatility_dict_wmt.items()), columns=['Date', 'WMT Volatility'])

        df_volatility_ko.set_index('Date', inplace=True)
        df_volatility_pep.set_index('Date', inplace=True)
        df_volatility_jnj.set_index('Date', inplace=True)
        df_volatility_orcl.set_index('Date', inplace=True)
        df_volatility_pfe.set_index('Date', inplace=True)
        df_volatility_wmt.set_index('Date', inplace=True)

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

    return df_volatility_aapl, df_volatility_ibm, df_volatility_mcd, df_volatility_ko, df_volatility_pep, df_volatility_jnj, df_volatility_orcl, df_volatility_pfe, df_volatility_wmt, volatility_dict_aapl, volatility_dict_ibm, volatility_dict_mcd, volatility_dict_ko, volatility_dict_pep, volatility_dict_jnj, volatility_dict_orcl, volatility_dict_pfe, volatility_dict_wmt 


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

# Risk-free rate
FF = pd.read_csv('data/ff5.csv', index_col=0, parse_dates=True)
rf_rate = pd.DataFrame(FF.loc[:,"RF"])
rf_rate.index = pd.to_datetime(rf_rate.index, format='%Y%m%d')

# i have 13 years of trading data; 3273 days; now i will split the dataset into 70% training, 20% validation and 10% test
'''
total_days = len(stocksPrices)
date_70_percent = stocksPrices.index[int(total_days * 0.7)]
date_90_percent = stocksPrices.index[int(total_days * 0.9)]
'''

# Hardcoded split dates based on your output
date_70_percent = pd.Timestamp('2019-02-08 00:00:00')
date_90_percent = pd.Timestamp('2021-09-14 00:00:00')

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

# repeat for validation and test dataset
trainingDataWithPercentageChange.columns = columns
validationDataWithPercentageChange.columns = columns
testDataWithPercentageChange.columns = columns

# now define the start date and end date for every dataset
startingTrainingDate = trainingDataWithPercentageChange.index[0].strftime("%Y-%m-%d")
endingTrainingDate = trainingDataWithPercentageChange.index[-1].strftime("%Y-%m-%d")

startingValidationDate = validationDataWithPercentageChange.index[0].strftime("%Y-%m-%d")
endingValidationDate = validationDataWithPercentageChange.index[-1].strftime("%Y-%m-%d")

startingTestDate = testDataWithPercentageChange.index[0].strftime("%Y-%m-%d")
endingTestDate = testDataWithPercentageChange.index[-1].strftime("%Y-%m-%d")

# Plot the returns of the stocks with highlights and annotations
plt.figure(figsize=(14, 7))
# plot the stock prices of just the test dataset
plt.plot(stocksPrices.loc[startingTestDate:endingTestDate])
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

prescientDict = originalPrescientPredictor(stocksPercentageChangeReturn)

    
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

# now filter the real volatilities between the start and end date
real_volatility_startDate = pd.to_datetime('2010-01-04')
real_volatility_endDate = pd.to_datetime('2024-05-21')

# assets names are AAPL, IBM, MCD, KO, PEP, JNJ, ORCL, PFE, WMT
df_volatility_aapl, df_volatility_ibm, df_volatility_mcd, df_volatility_ko, df_volatility_pep, df_volatility_jnj, df_volatility_orcl, df_volatility_pfe, df_volatility_wmt, volatility_dict_aapl, volatility_dict_ibm, volatility_dict_mcd, volatility_dict_ko, volatility_dict_pep, volatility_dict_jnj, volatility_dict_orcl, volatility_dict_pfe, volatility_dict_wmt = plot_volatility(prescientDict, real_volatility_startDate, real_volatility_endDate, 'PRESCIENT')

# EXPANDING WINDOW DICT
# NOW I IMPLEMENT AN EXPANDING WINDOW MODEL FOR EVERY QUARTER

expandingWindowDict = expandingWindowPredictor(stocksPercentageChangeReturn)

print("dimension of dataset: " + str(stocksPercentageChangeReturn.shape))

print("len of the expanding window dictionary: " + str(len(expandingWindowDict)))

# print just the first key and value of the dictionary
print(list(expandingWindowDict.keys())[0])
print(expandingWindowDict[list(expandingWindowDict.keys())[0]])