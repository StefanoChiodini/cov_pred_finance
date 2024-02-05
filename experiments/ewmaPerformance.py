import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

from cvx.covariance.ewma import iterated_ewma, _ewma_cov, _ewma_mean
from cvx.covariance.combination import from_sigmas

import datetime
import random

sns.set()
sns.set(font_scale=1.5)

boolUniformlyDistributedDataset = True
percentageOfRemovedDays = 40


# 
# IMPORT DATA 
#

stocksPrices = pd.read_csv('experiments/data/stocksPrices.csv', index_col=0, parse_dates=True)
stocksPercentageChangeReturn = pd.read_csv('experiments/data/stocksPercentageChange.csv', index_col=0, parse_dates=True)
uniformlyDistributedReturns = stocksPercentageChangeReturn.copy() # this is a copy of the original dataset returns; i will use this to make it non-uniformly distributed

# Risk-free rate
FF = pd.read_csv('experiments/data/ff5.csv', index_col=0, parse_dates=True)
rf_rate = pd.DataFrame(FF.loc[:,"RF"])
rf_rate.index = pd.to_datetime(rf_rate.index, format='%Y%m%d')

# i have 13 years of trading data; 3273 days; now i will split the dataset into 70% training, 20% validation and 10% test

total_days = len(stocksPrices)
date_70_percent = stocksPrices.index[int(total_days * 0.7)]
date_90_percent = stocksPrices.index[int(total_days * 0.9)]

# 70% training
trainingData = stocksPrices.loc[:date_70_percent]

# 20% validation
validationData = stocksPrices.loc[date_70_percent:date_90_percent]

# 10% test
testData = stocksPrices.loc[date_90_percent:]

# Import pickle
with open('experiments/data/permco_to_ticker.pkl', 'rb') as f:
    permco_to_ticker = pickle.load(f)

# Create a reverse mapping if necessary
ticker_to_permco = {v: k for k, v in permco_to_ticker.items()}

# Replace ticker symbols with PERMCO codes if the ticker symbol is found in the reverse mapping
columns = [ticker_to_permco.get(col, col) for col in stocksPercentageChangeReturn.columns]
stocksPercentageChangeReturn.columns = columns

# Repeat for uniformlyDistributedReturns
columns_udr = [ticker_to_permco.get(col, col) for col in uniformlyDistributedReturns.columns]
uniformlyDistributedReturns.columns = columns_udr

'''
# plot the returns of the stocks
plt.plot(stocksPrices)

# write a legend: the green line is aapl; the blue line is ibm; the red line is mcd(this is just an example)
plt.legend(["AAPL", "IBM", "MCD"])
plt.title("Returns of the stocks")
plt.xlabel("Time")
plt.ylabel("Returns")
plt.show()
'''

#
# DICTIONARY INITIALIZATION
#

# ewma_halflife = 100
# beta = 2 ** (-1 / ewma_halflife)
beta = 0.1
# Open a file to save the results
with open('ewmaPerformance.txt', 'w') as file:
    file.write('')  # Header for the text file

prescientDict = {}
log_likelihoods = {}
regrets = {}

# create a dictionary for every predictor that saves the log-likelihoods and the regrets
log_likelihood_rw = {}
log_likelihood_ewma = {}
log_likelihood_mgarch = {}
log_likelihood_prescient = {}

regret_rw = {}
regret_ewma = {}
regret_mgarch = {}
regret_prescient = {}

# collections for plotting charts about the performance of the EWMA predictor
betaValues = []
ewmaMeanRegretValues = []
ewmaMeanlogLikelihoodValues = []
prescientAlreadyPrinted = False

days_greater_than_one_values = []  # To store determinant values where > 1
days_less_than_one_values = []  # To store determinant values where < 1

#
# COVARIANCE PREDICTORS
#

#
# PRESCIENT 
#

# prescient is a dictionary that contains the covariance matrix calculated using the ewma formula written inside the paper
# the key of the dictionary is the timestamp and the value is the covariance matrix calculated for that day

# The prescient predictor will always use the original dataset, so it will be uniformly distributed; this is because the prescient predictor is used to compare the other predictors
# and we need to have a measure of the real covariance matrix; so this can't be used with the non-uniformly distributed dataset

for t in uniformlyDistributedReturns.index:
    # get sample covariance matrix for corresponding quarter
    quarter = (t.month-1)//3 + 1
    cov = np.cov(uniformlyDistributedReturns.loc[(uniformlyDistributedReturns.index.year == t.year) & (uniformlyDistributedReturns.index.quarter == quarter)].values, rowvar=False)
    mean = np.mean(uniformlyDistributedReturns.loc[(uniformlyDistributedReturns.index.year == t.year) & (uniformlyDistributedReturns.index.quarter == quarter)].values, axis=0)
    prescientDict[t] = pd.DataFrame(cov+np.outer(mean, mean), index=uniformlyDistributedReturns.columns, columns=uniformlyDistributedReturns.columns)


#
# VALIDATION PHASE
#
    
# Loop through beta values
while beta < 1:
    
    determinants_ewma = []  # To store determinant values
    days_less_than_one = 0  # Counter for days where determinant < 1
    days_greater_than_one = 0  # Counter for days where determinant > 1

    #
    # EWMA 
    #

    ewma_halflife = -np.log(2) / np.log(beta)
    ewmaDict = dict(_ewma_cov(stocksPercentageChangeReturn, halflife=ewma_halflife))

    # calculate the determinant of the covariance matrix for each timestamp
    for t in ewmaDict:
        determinants_ewma.append(np.linalg.det(ewmaDict[t]))

    # Count the number of days where the determinant is greater than 1 and where the determinant is less than 1
    for det in determinants_ewma:
        if det > 1:
            days_greater_than_one += 1
        else:
            days_less_than_one += 1
    
    # Append the number of days where the determinant is greater than 1 and where the determinant is less than 1
    days_greater_than_one_values.append(days_greater_than_one)
    days_less_than_one_values.append(days_less_than_one)


    #
    # DEFINE END AND START DATES FOR BACKTESTS
    #
        
    # Define start and end of backtest; first two years used for training/burn-in
    start_date = pd.to_datetime("2011-12-28", format="%Y-%m-%d")
    end_date = pd.to_datetime("2022-12-30", format="%Y-%m-%d")

    names = ["EWMA", "PRESCIENT"]

    #these predictors are all dictionaries where each entry contains a Pandas DataFrame representing a covariance matrix of returns at each timestamp.  
    predictors_temp = [ewmaDict, prescientDict]
    predictors = [] # so this is a list of dictionaries

    for predictor in predictors_temp:
        predictors.append({t: predictor[t] for t in predictor.keys() if t >= start_date and t <= end_date})


    #
    # MSEs
    #
        
    for i, predictorDict in enumerate(predictors):
        if names[i] != "PRESCIENT":
            MSE_temp = MSE(stocksPercentageChangeReturn, predictorDict).resample("Q").mean()

            # write the results in the file
            with open('ewmaPerformance.txt', 'a') as file:
                file.write(f'BETA:\t\t\t {beta}\newma_halflife: {ewma_halflife}\nMSEmean: {MSE_temp.mean():.6f}\nMSEstd: {MSE_temp.std():.6f}\nMSEmax: {MSE_temp.max():.6f}\n\n')


    #
    # LOG-LIKELIHOODS
    #

    '''
        this dictionary has a shape like this:
        {
            RW: pd.Series(log_likelihood(returns_temp, Sigmas_temp), index=times),
            EWMA: pd.Series(log_likelihood(returns_temp, Sigmas_temp), index=times),
            MGARCH: pd.Series(log_likelihood(returns_temp, Sigmas_temp), index=times),
            PRESCIENT: pd.Series(log_likelihood(returns_temp, Sigmas_temp), index=times),
        }

        where each pd.series is a series of log-likelihoods for each timestamp: so there is the log-likelihood value for each timestamp
    '''

    for i, predictorDict in enumerate(predictors):
        #print("Computing " + names[i] + " log-likelihood...")

        # if the predictor is the prescient predictor, i have to use the uniformly distributed dataset
        if names[i] == "PRESCIENT":
            returns_temp = uniformlyDistributedReturns.loc[pd.Series(predictorDict).index].values[1:]
        
        else:
            returns_temp = stocksPercentageChangeReturn.loc[pd.Series(predictorDict).index].values[1:]

        times = pd.Series(predictorDict).index[1:]
        Sigmas_temp = np.stack([predictorDict[t].values for t in predictorDict.keys()])[:-1]       
        log_likelihoods[names[i]] = pd.Series(log_likelihood(returns_temp, Sigmas_temp), index=times)

        if names[i] != "PRESCIENT":
           print("log_likelihoods[" + names[i] + "]: " + str(log_likelihoods[names[i]]))

    #
    # REGRETS
    #

    for name in log_likelihoods:
        regrets[name] =  log_likelihoods["PRESCIENT"] - log_likelihoods[name]
        
    for name in regrets:
        if name != "PRESCIENT":

            #Each data point in the regret series now represents the average regret for a respective quarter. If the original series spans multiple years, then the number of data points in regret will be the number of quarters in that time frame.
            regret = regrets[name].resample("Q").mean() #it resamples the regret Series to a quarterly frequency, This gives the average regret for each quarter rather than daily regret values  
            # so the regret variable is a series of average regret for each quarter
            
            regretMetrics = (np.mean(regret).round(1), np.std(regret).round(1), np.max(regret).round(1))
            # the round(1) function to each of these metrics, which rounds the result to one decimal place,

            # save the regret mean values to plot a chart
            ewmaMeanRegretValues.append(regretMetrics[0])

            # write the results in the file
            with open('ewmaPerformance.txt', 'a') as file:
                file.write(f'meanRegret: {regretMetrics[0]:.2f}\nstdRegret: {regretMetrics[1]:.2f}\nmaxRegret: {regretMetrics[2]:.2f}\n\n')


    # copy the log-likelihoods dictionary
    log_likelihoods_copy = log_likelihoods.copy()

    # do the same thing for log-likelihoods dictionary
    for name in log_likelihoods_copy:
        logLikelihood = log_likelihoods_copy[name].resample("Q").mean()
        logLikelihoodMetrics = (np.mean(logLikelihood).round(1), np.std(logLikelihood).round(1), np.max(logLikelihood).round(1))

        if name != "PRESCIENT":

            # save the loglikelihood mean values to plot a chart
            ewmaMeanlogLikelihoodValues.append(logLikelihoodMetrics[0])
            # write the results in the file
            with open('ewmaPerformance.txt', 'a') as file:
                file.write(f'meanLoglikelihood{name}: {logLikelihoodMetrics[0]:.2f}\nstdLoglikelihood{name}: {logLikelihoodMetrics[1]:.2f}\nmaxLoglikelihood{name}: {logLikelihoodMetrics[2]:.2f}\n\n')
        
        if name == "PRESCIENT" and prescientAlreadyPrinted == False:
            # write the results in the file so i'm writing the prescient predictor only once, because it's the same for every beta
            with open('ewmaPerformance.txt', 'a') as file:
                file.write(f'meanLoglikelihood{name}: {logLikelihoodMetrics[0]:.2f}\nstdLoglikelihood{name}: {logLikelihoodMetrics[1]:.2f}\nmaxLoglikelihood{name}: {logLikelihoodMetrics[2]:.2f}\n\n')
            prescientAlreadyPrinted = True

            # save the loglikelihood mean value to plot a chart
            prescientMeanlogLikelihoodValue = logLikelihoodMetrics[0]

    # save every fundamental value to plot a chart 
    betaValues.append(beta)

    # Increment beta
    beta += 0.01


print("days_greater_than_one_values: " + str(days_greater_than_one_values))
print("days_less_than_one_values: " + str(days_less_than_one_values))


# plot the chart of how many days the determinant is greater than 1 and how many days the determinant is less than 1 
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot for days where determinant is greater than 1
axs[0].plot(betaValues, days_greater_than_one_values, color='blue', marker='o', linestyle='-')
axs[0].set_title('Days with Determinant > 1 by beta decay factor')
axs[0].set_ylabel('Number of Days')

# Plot for days where determinant is less than 1
axs[1].plot(betaValues, days_less_than_one_values, color='red', marker='o', linestyle='-')
axs[1].set_title('Days with Determinant < 1 by by beta decay factor')
axs[1].set_xlabel('beta decay factor')
axs[1].set_ylabel('Number of Days')

plt.tight_layout()
plt.show()

'''
betaMinValuesRegret = [] # this list will contain the beta values that have the minimum regret value
betaMaxValuesLogLikelihood = [] # this list will contain the beta values that have the max loglikelihood value

# plot the chart of the mean regret values
plt.figure()
plt.plot(betaValues, ewmaMeanRegretValues)
plt.xlabel("Beta values")
plt.ylabel("Mean regret values")
plt.title("Mean regret values for different beta values")

# find the minimum value of the mean regret values
for i in range(len(ewmaMeanRegretValues)):
    if ewmaMeanRegretValues[i] == min(ewmaMeanRegretValues):
        betaMinValuesRegret.append(betaValues[i])

# these points show the interval of beta values that have the minimum regret value
highlightPoint1 = betaMinValuesRegret[0]
plt.scatter(highlightPoint1, min(ewmaMeanRegretValues), color='r')
plt.text(highlightPoint1, min(ewmaMeanRegretValues), f' Beta: {highlightPoint1:.4f}\n Regret: {min(ewmaMeanRegretValues):.4f}',
         fontsize=9, color='red', ha='center', va='bottom')

highlightPoint2 = betaMinValuesRegret[-1]
plt.scatter(highlightPoint2, min(ewmaMeanRegretValues), color='r')
plt.text(highlightPoint2, min(ewmaMeanRegretValues), f' Beta: {highlightPoint2:.4f}\n Regret: {min(ewmaMeanRegretValues):.4f}',
         fontsize=9, color='red', ha='center', va='bottom')

print("highlightPoint1: " + str(highlightPoint1))
print("highlightPoint2: " + str(highlightPoint2))

plt.show()


# plot the chart of the mean loglikelihood values
plt.figure()
plt.plot(betaValues, ewmaMeanlogLikelihoodValues)
plt.xlabel("Beta values")
plt.ylabel("Mean loglikelihood values")
plt.title("Mean loglikelihood values for different beta values")

# show the prescient predictor loglikelihood value
plt.axhline(y=prescientMeanlogLikelihoodValue, color='r', linestyle='-')
plt.legend(["EWMA", "PRESCIENT"])

for j in range(len(ewmaMeanlogLikelihoodValues)):
    if ewmaMeanlogLikelihoodValues[j] == max(ewmaMeanlogLikelihoodValues):
        betaMaxValuesLogLikelihood.append(betaValues[j])

# these points show the interval of beta values that have the max loglikelihood value
highlightPoint1LogLikelihood = betaMaxValuesLogLikelihood[0]
plt.scatter(highlightPoint1LogLikelihood, max(ewmaMeanlogLikelihoodValues), color='r')
plt.text(highlightPoint1LogLikelihood, max(ewmaMeanlogLikelihoodValues), f' Beta: {highlightPoint1LogLikelihood:.4f}\n Loglikelihood: {max(ewmaMeanlogLikelihoodValues):.4f}',
         fontsize=9, color='red', ha='center', va='bottom')

highlightPoint2LogLikelihood = betaMaxValuesLogLikelihood[-1]
plt.scatter(highlightPoint2LogLikelihood, max(ewmaMeanlogLikelihoodValues), color='r')
plt.text(highlightPoint2LogLikelihood, max(ewmaMeanlogLikelihoodValues), f' Beta: {highlightPoint2LogLikelihood:.4f}\n Loglikelihood: {max(ewmaMeanlogLikelihoodValues):.4f}',
         fontsize=9, color='red', ha='center', va='bottom')

print("highlightPoint1LogLikelihood: " + str(highlightPoint1LogLikelihood))
print("highlightPoint2LogLikelihood: " + str(highlightPoint2LogLikelihood))
plt.show()
'''

