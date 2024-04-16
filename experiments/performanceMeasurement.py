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


def predictorLogLikelihood():
    '''
    this function calculates the loglikelihood of a predictor
    '''
    pass