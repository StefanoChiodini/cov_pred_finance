# this file contains the implementation of the following predictors:

#     - EXPANDING WINDOW
#     - PRESCIENT
#     - HYBRID PREDICTOR

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
partialPath = "C:\\Users\\chiod\\Downloads\\"


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
    

def logistic_function(x, k = 10, x_0 = 0.5):
    """
    Logistic function to calculate the lambda parameter.
    
    Args:
    - x: The normalized day within the quarter.
    - k: The steepness of the curve.
    - x_0: The x-value of the sigmoid's midpoint.

    Returns:
    - The lambda parameter as a value between 0 and 1.
    """
    return 1 / (1 + np.exp(-k * (x - x_0)))


def linear_increment(day_number, days_in_quarter):
    """
    Linear increment function that maps a day number to its normalized position within the quarter.
    
    Args:
    - day_number: current day number in the quarter.
    - days_in_quarter: total number of days in the quarter.
    
    Returns:
    - The normalized position of the day within the quarter, which serves as the lambda value.
    """
    if days_in_quarter <= 1:
        return 0 if day_number == 0 else 1
    return day_number / (days_in_quarter - 1)


def exponential_increment(x, k):
    """
    Exponential increment function for lambda value that normalizes within the quarter.
    
    Args:
    - x: Normalized day within the quarter (range from 0 to 1).
    - k: Controls the growth rate of the function.
    
    Returns:
    - The lambda value based on the exponential growth function.
    """
    return np.exp(k * (x - 1)) 


def logarithmic_increment(x, k):
    """
    Logarithmic increment function for lambda value that normalizes within the quarter.
    
    Args:
    - x: Normalized day within the quarter (range from 0 to 1).
    - k: Controls the steepness of the logarithmic curve.
    
    Returns:
    - The lambda value based on the logarithmic growth function.
    """
    return np.log(k * x + 1) / np.log(k + 1)


def hybridPredictor(uniformlyDistributedReturns, datasetWithPercentageChange, expandingWindowDict, predictorDict, start_date, increment_type='linear', k = 10):
    '''
    This function implements the hybrid predictor.
    '''
    # Ensure datetime index
    datasetWithPercentageChange.index = pd.to_datetime(datasetWithPercentageChange.index)
    start_date = pd.to_datetime(start_date)

    # Align the dictionaries by ensuring they start on the same date
    start = max(min(expandingWindowDict.keys()), min(predictorDict.keys()), start_date)
    expandingWindowDict = {k: v for k, v in expandingWindowDict.items() if k >= start}
    predictorDict = {k: v for k, v in predictorDict.items() if k >= start}

    hybridModelDict = {}
    datasetWithPercentageChange = datasetWithPercentageChange[start:]
    initialMonth = start.month
    initialYear = start.year
    initialQuarter = (initialMonth - 1) // 3 + 1
    tempQuarter = initialQuarter

    lambdaValuesList = [] # for testing purposes, delete it

    # Get the number of days in the initial quarter
    numberOfDaysInQuarter = len(datasetWithPercentageChange.loc[(datasetWithPercentageChange.index.year == initialYear) & (datasetWithPercentageChange.index.quarter == initialQuarter)])

    lambdaParam = 0  # Reset lambda at the start
    day_number = 0

    for t in datasetWithPercentageChange.index:

        if t not in expandingWindowDict or t not in predictorDict:
            continue

        ewMatrix = expandingWindowDict[t]
        predMatrix = predictorDict[t]

        quarter = (t.month-1)//3 + 1

        if quarter != tempQuarter:
            # enter here if the quarter changes
            tempQuarter = quarter

            # i recalculate the increment of the lambda parameter
            numberOfDaysInQuarter = len(datasetWithPercentageChange.loc[(datasetWithPercentageChange.index.year == t.year) & (datasetWithPercentageChange.index.quarter == quarter)])           
            day_number = 0

        # Calculate the lambda parameter using the linear function
        x_value = linear_increment(day_number, numberOfDaysInQuarter)  # normalize day number to range [0, 1]
        lambdaParam = x_value  # apply the linear function

        # Calculate the lambda parameter using the logistic function
        #lambdaParam = logistic_function(day_number / numberOfDaysInQuarter, k)

        # Calculate the lambda parameter using the exponential function
        #lambdaParam = exponential_increment(day_number / numberOfDaysInQuarter, k)  # apply the exponential function

        lambdaValuesList.append(lambdaParam) # testing purposes, delete later

        # calculate the covariance matrix using the hybrid model
        hybrid_cov_matrix = (1 - lambdaParam) * predMatrix + lambdaParam * ewMatrix # covariance matrix at time t

        # truncate every element of the covariance matrix to 6 decimals
        hybrid_cov_matrix = hybrid_cov_matrix.round(6)

        # convert the hybrid covariance matrix to a DataFrame for this specific date
        hybridModelDict[t] = pd.DataFrame(hybrid_cov_matrix, index=uniformlyDistributedReturns.columns, columns=uniformlyDistributedReturns.columns)
        
        day_number += 1
        # assert that lambda value is always between 0 and 1, if it is not print the value of lambda and the day number
        assert 0 <= lambdaParam <= 1, f"Lambda value is not between 0 and 1: lambda = {lambdaParam}, day_number = {day_number}"

 
    #######################################################################################
    # Sample lambdaValuesList for demonstration
    #lambdaValuesList = [i/365 for i in range(366)] # Remove this line in your actual code

    # midOfTheQuarterList = [] # for testing purposes, delete it; this list contains the number of the days that are in the middle of the quarter
    # endOfTheQuarterList = [0, 57, 119, 181, 245, 308] # for testing purposes, delete it; this list contains the number of the days that are in the end of the quarter

    print("length of lambda values list: ", len(lambdaValuesList)) # testing purposes, delete later

    # for i in range(1, 6):
    #     midOfTheQuarterList.append((endOfTheQuarterList[i-1] + endOfTheQuarterList[i]) // 2)
    
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
    mpl.rcParams['font.size'] = 36
    mpl.rcParams['figure.figsize'] = [14, 8]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(lambdaValuesList)

    # Set face color of the figure and axes to white
    # fig.patch.set_facecolor('white')
    # ax.set_facecolor('white')

    # on the y axis write exactly the numbers 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0
    yAxisNumbers = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    plt.yticks(yAxisNumbers)

    plt.title('Lambda values for every quarter')
    plt.xlabel('Day number')
    # use the latex notation to write the lambda values
    plt.ylabel(r'$\lambda$ Values')
    plt.xlim(0, len(lambdaValuesList))
    plt.ylim(0.0, 1.0)

    # Disable default style
    plt.style.use('default')
    
    plt.show()
    
    # save the chart in an eps format
    plt.savefig(partialPath + 'lambda_values_chart.png', format='png', dpi=1000)

    print("file saved")

    #######################################################################################
    return hybridModelDict