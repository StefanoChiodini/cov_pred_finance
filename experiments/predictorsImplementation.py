# this file contains the implementation of the following predictors:

#     - EXPANDING WINDOW
#     - PRESCIENT
#     - HYBRID PREDICTOR

import numpy as np
import pandas as pd


def calculate_lambda_increment(day_number, days_in_quarter, increment_type ='linear'):
    """
    Calculate the lambda increment based on the increment type.
    
    Args:
    - day_number: current day number in the quarter.
    - days_in_quarter: total number of days in the quarter.
    - increment_type: type of increment - 'linear', 'exponential', 'logarithmic', etc.

    Returns:
    - The calculated lambda increment.
    """

    # TODO: CONTROL THE CORRECTNESS OF THIS FUNCTION

    if increment_type == 'linear':
        return 1 / max((days_in_quarter - 1), 1)
    elif increment_type == 'exponential':
        # Example exponential increment (needs to be adjusted based on requirements)
        return (1 - np.exp(-day_number)) / (1 - np.exp(-days_in_quarter))
    elif increment_type == 'logarithmic':
        # Example logarithmic increment (needs to be adjusted based on requirements)
        return np.log1p(day_number) / np.log1p(days_in_quarter)
    # You can add more increment types with corresponding formulas here
    else:
        raise ValueError("Unknown increment type")


def empiricalCovarianceMatrix(quarterCovarianceMatrixList):
    '''
    Function to calculate the empirical covariance matrix. The covariance matrix is calculated using exactly the same formula as in the paper.
    '''
    # sum all the covariance matrices for the quarter(sum all the matrices contained in the list)
    quarterCovarianceMatricesSum = sum(quarterCovarianceMatrixList)

    # take the average of the covariance matrices; divide the sum of the matrices by the lenght of the quarter (so the lenght of the list)
    empirical_cov_matrix = quarterCovarianceMatricesSum / len(quarterCovarianceMatrixList)

    # truncate every element of the covariance matrix to 6 decimals
    empirical_cov_matrix = empirical_cov_matrix.round(6)

    return empirical_cov_matrix


# PRESCIENT PREDICTOR (ORIGINAL VERSION OF THE PAPER)
def originalPrescientPredictor(uniformlyDistributedReturns):
    '''
    This function implements the prescient predictor as written inside github repo.
    It takes as input the uniformly distributed returns and returns the prescient dictionary
    '''
    prescientDict = {}

    for t in uniformlyDistributedReturns.index:
        # get sample covariance matrix for corresponding quarter
        quarter = (t.month-1)//3 + 1  
        cov = np.cov(uniformlyDistributedReturns.loc[(uniformlyDistributedReturns.index.year == t.year) & (uniformlyDistributedReturns.index.quarter == quarter)].values, rowvar=False)
        mean = np.mean(uniformlyDistributedReturns.loc[(uniformlyDistributedReturns.index.year == t.year) & (uniformlyDistributedReturns.index.quarter == quarter)].values, axis=0)
        prescientDict[t] = pd.DataFrame(cov+np.outer(mean, mean), index=uniformlyDistributedReturns.columns, columns=uniformlyDistributedReturns.columns)

    return prescientDict


# PRESCIENT PREDICTOR (MODIFIED VERSION -> PERSONAL IMPLEMENTATION MADE BY IMPLEMENTING EXACTILY THE DEFINITION OF THE EMPIRICAL COVARIANCE MATRIX)
def prescientPredictor(uniformlyDistributedReturns):
    '''
    This function implements the prescient predictor by using the definition of the empirical covariance matrix.
    It takes as input the uniformly distributed returns and returns the prescient dictionary
    '''
    # empirical covariance matrix using paper formula
    prescientDictModified = {}

    empCovarianceMatrixList = []
    quarterCovarianceMatrixList = []

    initialMonth = uniformlyDistributedReturns.index[0].month # get the month of the first day of the test dataset
    initialQuarter = (initialMonth-1)//3 + 1 # get the quarter of the first day of the test dataset
    tempQuarter = initialQuarter # this is the initial quarter

    for t in uniformlyDistributedReturns.index:
        
        # get sample covariance matrix for corresponding quarter
        quarter = (t.month-1)//3 + 1

        # if the quarter changes, calculate the empirical covariance matrix for the quarter just passed
        if quarter != tempQuarter:
            # CALCULATE HERE THE EMPIRICAL COVARIANCE MATRIX FOR THE QUARTER JUST PASSED
            
            empirical_cov_matrix = empiricalCovarianceMatrix(quarterCovarianceMatrixList)

            # add the empirical covariance matrix to the list
            empCovarianceMatrixList.append(empirical_cov_matrix)

            # now attribute the empirical covariance matrix to every single day of the quarter
            for i in range(len(quarterCovarianceMatrixList)):
                # append the entry to the dictionary: the key is the date and the value is the empirical covariance matrix; add t to the key to avoid overwriting
                prescientDictModified[t + pd.DateOffset(days=i)] = pd.DataFrame(empirical_cov_matrix, index=uniformlyDistributedReturns.columns, columns=uniformlyDistributedReturns.columns)
            
            # variable reset
            tempQuarter = quarter
            quarterCovarianceMatrixList = []

        # get the percentage change of the stocks for the current day
        today_returns = uniformlyDistributedReturns.loc[t]

        # multiply the returns by the transpose of the returns
        covariance_matrix = np.outer(today_returns, today_returns.T) # covariance matrix at time t

        # add the covariance matrix to the list
        quarterCovarianceMatrixList.append(covariance_matrix) # this is the list of the covariance matrix for the current quarter


    # when the loop ends, calculate the empirical covariance matrix for the last quarter ( this because the last quarter has been excluded from the loop)
    empirical_cov_matrix = empiricalCovarianceMatrix(quarterCovarianceMatrixList)

    # add the empirical covariance matrix to the list
    empCovarianceMatrixList.append(empirical_cov_matrix)

    # now attribute the empirical covariance matrix to every single day of the quarter
    for i in range(len(quarterCovarianceMatrixList)):
        # append the entry to the dictionary: the key is the date and the value is the empirical covariance matrix; add t to the key to avoid overwriting
        prescientDictModified[t + pd.DateOffset(days=i)] = pd.DataFrame(empirical_cov_matrix, index=uniformlyDistributedReturns.columns, columns=uniformlyDistributedReturns.columns)

    return prescientDictModified


# EXPANDING WINDOW PREDICTOR
def expandingWindowPredictor(uniformlyDistributedReturns):
    '''
    This function implements the expanding window predictor.
    '''
    expandingWindowDict = {}

    quarterCovarianceMatrixList = []
    initialMonth = uniformlyDistributedReturns.index[0].month # get the month of the first day of the test dataset
    initialQuarter = (initialMonth-1)//3 + 1 # get the quarter of the first day of the test dataset
    tempQuarter = initialQuarter # this is the initial quarter


    for t in uniformlyDistributedReturns.index:

        # get sample covariance matrix for corresponding quarter
        quarter = (t.month-1)//3 + 1

        # if the quarter changes, i have to reset the list of the covariance matrices
        if quarter != tempQuarter:
            quarterCovarianceMatrixList = []
            tempQuarter = quarter

        # get the percentage change of the stocks for the current day
        today_returns = uniformlyDistributedReturns.loc[t]

        # multiply the returns by the transpose of the returns
        covariance_matrix = np.outer(today_returns, today_returns.T) # covariance matrix at time t

        # add the covariance matrix to the list
        quarterCovarianceMatrixList.append(covariance_matrix) # this is the list of the covariance matrix for the current quarter

        # sum all the covariance matrices for the quarter(sum all the matrices contained in the list)
        quarterCovarianceMatricesSum = sum(quarterCovarianceMatrixList)

        # take the average of the covariance matrices; divide the sum of the matrices by the lenght of the quarter (so the lenght of the list)
        ew_cov_matrix = quarterCovarianceMatricesSum / len(quarterCovarianceMatrixList)

        # truncate every element of the covariance matrix to 6 decimals
        ew_cov_matrix = ew_cov_matrix.round(6)

        # Convert the empirical covariance matrix to a DataFrame for this specific date
        expandingWindowDict[t] = pd.DataFrame(ew_cov_matrix, index=uniformlyDistributedReturns.columns, columns=uniformlyDistributedReturns.columns)

    return expandingWindowDict


def hybridPredictorNewVersion(uniformlyDistributedReturns, datasetWithPercentageChange, expandingWindowDict, predictorDict, start_date, increment_type='linear'):
    '''
    This function implements the hybrid predictor.
    '''
    hybridModelDict = {}

    lambdaParam = 0  # initial value of lambda for each quarter
    #extract the initial month from the start date
    initialMonth = datasetWithPercentageChange.index[0].month # get the month of the first day of the test dataset
    initialQuarter = (initialMonth-1)//3 + 1 # get the quarter of the first day of the test dataset
    tempQuarter = initialQuarter # this is the initial quarter

    numberOfDaysInQuarter = len(datasetWithPercentageChange.loc[(datasetWithPercentageChange.index.year == start_date.year) & (datasetWithPercentageChange.index.quarter == initialQuarter)])
    day_number = 0  # to keep track of the day number within the quarter

    for t in datasetWithPercentageChange.index:

        # check if the matrix exists in every dictionary
        if t not in expandingWindowDict or t not in predictorDict:
            continue

        ewMatrix = expandingWindowDict[t]
        predMatrix = predictorDict[t]

        quarter = (t.month-1)//3 + 1

        if quarter != tempQuarter:
            # enter here if the quarter changes
            lambdaParam = 0
            tempQuarter = quarter

            # i recalculate the increment of the lambda parameter
            numberOfDaysInQuarter = len(datasetWithPercentageChange.loc[(datasetWithPercentageChange.index.year == t.year) & (datasetWithPercentageChange.index.quarter == quarter)])
            day_number = 0

        # calculate the covariance matrix using the hybrid model
        hybrid_cov_matrix = (1 - lambdaParam) * predMatrix + lambdaParam * ewMatrix # covariance matrix at time t

        # truncate every element of the covariance matrix to 6 decimals
        hybrid_cov_matrix = hybrid_cov_matrix.round(6)

        # convert the hybrid covariance matrix to a DataFrame for this specific date
        hybridModelDict[t] = pd.DataFrame(hybrid_cov_matrix, index=uniformlyDistributedReturns.columns, columns=uniformlyDistributedReturns.columns)

        if day_number == numberOfDaysInQuarter - 1:
            lambdaParam = 1
        else:
            lambdaIncrement = calculate_lambda_increment(day_number, numberOfDaysInQuarter, increment_type)
            lambdaParam += lambdaIncrement
        day_number += 1

    return hybridModelDict