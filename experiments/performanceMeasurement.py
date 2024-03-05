# this file contains the function that measure the performance of the predictor by calculating loglikelihood and regret
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predictorPerformance(predictors, names, uniformlyDistributedReturns, testDataWithPercentageChange, rwMeanRegretValues):
    '''
    This function measures the performance of the predictor by calculating loglikelihood and regret
    '''
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

    log_likelihoods = {}
    for i, predictorDict in enumerate(predictors):

        # if the predictor is the prescient predictor, i have to use the uniformly distributed dataset
        if names[i] == "PRESCIENT":
            returns_temp = uniformlyDistributedReturns.loc[pd.Series(predictorDict).index].values[1:]
        
        else:
            returns_temp = testDataWithPercentageChange.loc[pd.Series(predictorDict).index].values[1:]

        times = pd.Series(predictorDict).index[1:]
        Sigmas_temp = np.stack([predictorDict[t].values for t in predictorDict.keys()])[:-1]       
        log_likelihoods[names[i]] = pd.Series(log_likelihood(returns_temp, Sigmas_temp), index=times)

    # Iterate through each predictor in the log_likelihoods dictionary
    for name in log_likelihoods.keys():
        if name == 'PRESCIENT':
            # Resample by quarter, take the mean, and plot with specific color and label
            log_likelihoods[name].resample("Q").mean().plot(label=name, c="k")
        else:
            # Resample by quarter, take the mean, and plot with default settings
            log_likelihoods[name].resample("Q").mean().plot(label=name)

    plt.xlabel('Time(quarter)')  # Set the x-axis label
    plt.ylabel('Log Likelihood')  # Set the y-axis label
    plt.title('Quarterly Mean Log Likelihood by Predictor')  # Set the title of the plot
    plt.legend()  # Show the legend to identify each predictor
    plt.show()  # Display the plot

    '''
        this dictionary has a shape like this:
        {
            RW: pd.Series(...),
            EWMA: pd.Series(...),
            MGARCH: pd.Series(...),
            PRESCIENT: pd.Series(...),
        }

        where each pd.series is a series of regret for each timestamp: so there is the 
        regret value (the difference between the log-likelihood of the prescient model and the log-likelihood of the model) for each timestamp
    '''
    regrets = {}
    for name in log_likelihoods:
        regrets[name] =  log_likelihoods["PRESCIENT"] - log_likelihoods[name]


    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for name in names:
        if name == 'PRESCIENT':
            pass
        else:
            regrets[name].resample("Q").mean().plot(label=name)
    plt.legend(bbox_to_anchor=(1, 1.1), loc='center right', ncols=4, labels=names[:-1], scatterpoints=1, markerscale=5);
    plt.xlabel('Time(quarter)')  # Set the x-axis label
    plt.ylabel('Regret')  # Set the y-axis label
    plt.title("Regret")

    for name in regrets:
        if name != "PRESCIENT":

            #Each data point in the regret series now represents the average regret for a respective quarter. If the original series spans multiple years, then the number of data points in regret will be the number of quarters in that time frame.
            regret = regrets[name].resample("Q").mean() #it resamples the regret Series to a quarterly frequency, This gives the average regret for each quarter rather than daily regret values  
            # so the regret variable is a series of average regret for each quarter
            
            regretMetrics = (np.mean(regret).round(1), np.std(regret).round(1), np.max(regret).round(1))
            # the round(1) function to each of these metrics, which rounds the result to one decimal place,

            # save the regret mean values to plot a chart
            rwMeanRegretValues.append(regretMetrics[0])

    print("\n")
    print(f"meanRegret: {regretMetrics[0]:.3f}")
    print(f"stdRegret: {regretMetrics[1]:.3f}")
    print(f"maxRegret: {regretMetrics[2]:.3f}")

    # copy the log-likelihoods dictionary
    log_likelihoods_copy = log_likelihoods.copy()

    # do the same thing for log-likelihoods dictionary
    for name in log_likelihoods_copy:
        logLikelihood = log_likelihoods_copy[name].resample("Q").mean()
        logLikelihoodMetrics = (np.mean(logLikelihood).round(1), np.std(logLikelihood).round(1), np.max(logLikelihood).round(1))

        print("\n")
        print(f"meanLoglikelihood{name}: {logLikelihoodMetrics[0]:.3f}")
        print(f"stdLoglikelihood{name}: {logLikelihoodMetrics[1]:.3f}")
        print(f"maxLoglikelihood{name}: {logLikelihoodMetrics[2]:.3f}")


    #
    # MSEs
    #

    for i, predictorDict in enumerate(predictors):
        if names[i] != "PRESCIENT":
            MSE_temp = MSE(testDataWithPercentageChange, predictorDict).resample("Q").mean()

            print("\n" + names[i] + " MSE")
            print(f"mean: {MSE_temp.mean():.10f}")
            print(f"std: {MSE_temp.std():.10f}")
            print(f"max: {MSE_temp.max():.10f}")
