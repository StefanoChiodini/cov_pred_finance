# this file contains the function that measure the performance of the predictor by calculating loglikelihood and regret
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.experiment_utils import *

def plotRMSE(RMSEs):
    '''
    this function is used for plotting the rmse values found
    '''
        # Convert Timestamps to strings for plotting
    timestamps = [ts.strftime('%Y-%m-%d') for ts in RMSEs.keys()]
    rmse_values = list(RMSEs.values())

    # Plot the RMSEs with improved formatting
    plt.figure(figsize=(14, 7))  # Increase the figure size for better readability
    plt.plot(timestamps, rmse_values, marker='o', linestyle='-', label='RW', color='b')

    # Set the x-axis to only include the dates from the dictionary
    plt.xticks(timestamps, rotation=45)

    # Remove the left margin
    plt.margins(x=0)

    # Add labels and title
    plt.xlabel('Time (Quarter)')
    plt.ylabel('RMSE')
    plt.title('RMSEs')
    plt.legend()

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def RMSEcovarianceMatrix(testDataWithPercentageChange, predictors, prescientDict, start_date, names):
    '''
    this function calculates the RMSE for the covariance matrix. the RMSE is calculated as the square root of the mean of the squared differences between the predicted covariance matrix and the true covariance matrix
    '''
    for i, predictorDict in enumerate(predictors):
        if names[i] != "PRESCIENT":
            print("lenght of predictorDict: ", len(predictorDict))
            RMSEs = RMSE(testDataWithPercentageChange, predictorDict, prescientDict, start_date)
            print("\n" + names[i] + " RMSE")

            # Calculate mean, standard deviation, and max value of the RMSEs
            mean_rmse = np.mean(list(RMSEs.values()))
            std_rmse = np.std(list(RMSEs.values()))
            max_rmse = np.max(list(RMSEs.values()))

            print(f"mean: {mean_rmse:.10f}")
            print(f"std: {std_rmse:.10f}")
            print(f"max: {max_rmse:.10f}")


    print("lenght of rmses: ", len(RMSEs))
    print("values of rmses: ", RMSEs)

    plotRMSE(RMSEs)


def RMSEforSingleAssets(testDataWithPercentageChange, volatility_dict_aapl_filtered, volatility_dict_ibm_filtered, volatility_dict_mcd_filtered,volatility_dict_aapl_predictor_filtered, volatility_dict_ibm_predictor_filtered, volatility_dict_mcd_predictor_filtered, start_date):
    '''
    this function calculates the RMSE for single assets. the RMSE is calculated as the square root of the mean of the squared differences between the predicted volatility and the true volatility
    '''
    # get the rmse of single assets. i take just aapl, ibm and mcd
    RMSEs_aapl_dict = RMSEforSingleVolatility(testDataWithPercentageChange, volatility_dict_aapl_filtered, volatility_dict_aapl_predictor_filtered, start_date)
    RMSEs_ibm_dict = RMSEforSingleVolatility(testDataWithPercentageChange, volatility_dict_ibm_filtered, volatility_dict_ibm_predictor_filtered, start_date)
    RMSEs_mcd_dict = RMSEforSingleVolatility(testDataWithPercentageChange, volatility_dict_mcd_filtered, volatility_dict_mcd_predictor_filtered, start_date)

    print("lenght of RMSEs_aapl: ", len(RMSEs_aapl_dict))
    print("values of RMSEs_aapl: ", RMSEs_aapl_dict)

    print("\n")
    print("RMSEs for AAPL")

    # Calculate mean, standard deviation, and max value of the RMSEs. the RMSEs are dictionaries whose key is the timestamp and the value is the rmse value
    mean_rmse_aapl = np.mean(list(RMSEs_aapl_dict.values()))
    std_rmse_aapl = np.std(list(RMSEs_aapl_dict.values()))
    max_rmse_aapl = np.max(list(RMSEs_aapl_dict.values()))

    print(f"mean: {mean_rmse_aapl:.10f}")
    print(f"std: {std_rmse_aapl:.10f}")
    print(f"max: {max_rmse_aapl:.10f}")

    print("\n")
    print("RMSEs for IBM")

    # Calculate mean, standard deviation, and max value of the RMSEs. the RMSEs are dictionaries whose key is the timestamp and the value is the rmse value
    mean_rmse_ibm = np.mean(list(RMSEs_ibm_dict.values()))
    std_rmse_ibm = np.std(list(RMSEs_ibm_dict.values()))
    max_rmse_ibm = np.max(list(RMSEs_ibm_dict.values()))

    print(f"mean: {mean_rmse_ibm:.10f}")
    print(f"std: {std_rmse_ibm:.10f}")
    print(f"max: {max_rmse_ibm:.10f}")

    print("\n")
    print("RMSEs for MCD")

    # Calculate mean, standard deviation, and max value of the RMSEs. the RMSEs are dictionaries whose key is the timestamp and the value is the rmse value
    mean_rmse_mcd = np.mean(list(RMSEs_mcd_dict.values()))
    std_rmse_mcd = np.std(list(RMSEs_mcd_dict.values()))
    max_rmse_mcd = np.max(list(RMSEs_mcd_dict.values()))

    print(f"mean: {mean_rmse_mcd:.10f}")
    print(f"std: {std_rmse_mcd:.10f}")
    print(f"max: {max_rmse_mcd:.10f}")


def predictorLogLikelihood(returnsDataset, predictors, names):
    '''
    this function calculates the loglikelihood of a predictor
    '''
    daily_log_likelihoods = {}  # Create an empty dictionary to store the daily log-likelihoods for each predictor

    for i, predictorDict in enumerate(predictors):

        returns_temp = returnsDataset.loc[pd.Series(predictorDict).index].values[1:]
        times = pd.Series(predictorDict).index[1:]
        Sigmas_temp = np.stack([predictorDict[t].values for t in predictorDict.keys()])[:-1]       
        daily_log_likelihoods[names[i]] = pd.Series(log_likelihood(returns_temp, Sigmas_temp), index=times)

    # Iterate through each predictor in the log_likelihoods dictionary
    for name in daily_log_likelihoods.keys():
        if name == 'PRESCIENT':
            # Resample by quarter, take the mean, and plot with specific color and label
            daily_log_likelihoods[name].resample("Q").mean().plot(label=name, c="k")
        else:
            # Resample by quarter, take the mean, and plot with default settings
            daily_log_likelihoods[name].resample("Q").mean().plot(label=name)

    plt.xlabel('Time(quarter)')  # Set the x-axis label
    plt.ylabel('Log Likelihood')  # Set the y-axis label
    plt.title('Quarterly Mean Log Likelihood by Predictor')  # Set the title of the plot
    plt.legend()  # Show the legend to identify each predictor
    plt.show()  # Display the plot

    # copy the log-likelihoods dictionary
    daily_log_likelihoods_copy = daily_log_likelihoods.copy()

    # do the same thing for log-likelihoods dictionary
    for name in daily_log_likelihoods_copy:
        logLikelihood = daily_log_likelihoods_copy[name].resample("Q").mean()

        print("logLikelihood length: ", len(logLikelihood))
        print("logLikelihood shape: ", logLikelihood.shape)
        logLikelihoodMetrics = (np.mean(logLikelihood).round(1), np.std(logLikelihood).round(1), np.max(logLikelihood).round(1))

        print("\n")
        print(f"meanLoglikelihood{name}: {logLikelihoodMetrics[0]:.3f}")
        print(f"stdLoglikelihood{name}: {logLikelihoodMetrics[1]:.3f}")
        print(f"maxLoglikelihood{name}: {logLikelihoodMetrics[2]:.3f}")

    return daily_log_likelihoods


def predictorRegret(daily_log_likelihoods, names):
    '''
    this function calculates the regret of a predictor
    '''
    daily_regrets = {}
    predictorMeanRegretValues = []

    for name in daily_log_likelihoods:
        daily_regrets[name] =  daily_log_likelihoods["PRESCIENT"] - daily_log_likelihoods[name]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for name in names:
        if name == 'PRESCIENT':
            pass
        else:
            daily_regrets[name].resample("Q").mean().plot(label=name)
    plt.legend(bbox_to_anchor=(1, 1.1), loc='center right', ncols=4, labels=names[:-1], scatterpoints=1, markerscale=5);
    plt.xlabel('Time(quarter)')  # Set the x-axis label
    plt.ylabel('regret')
    plt.title("Regret")

    for name in daily_regrets:
        if name != "PRESCIENT":

            #Each data point in the regret series now represents the average regret for a respective quarter. If the original series spans multiple years, then the number of data points in regret will be the number of quarters in that time frame.
            quarterly_regret = daily_regrets[name].resample("Q").mean() #it resamples the regret Series to a quarterly frequency, This gives the average regret for each quarter rather than daily regret values  
            # so the regret variable is a series of average regret for each quarter
            
            regretMetrics = (np.mean(quarterly_regret).round(1), np.std(quarterly_regret).round(1), np.max(quarterly_regret).round(1))
            # the round(1) function to each of these metrics, which rounds the result to one decimal place,

            # save the regret mean values to plot a chart
            predictorMeanRegretValues.append(regretMetrics[0])

    print("\n")
    print(f"meanRegret: {regretMetrics[0]:.3f}")
    print(f"stdRegret: {regretMetrics[1]:.3f}")
    print(f"maxRegret: {regretMetrics[2]:.3f}")