from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from cvx.covariance.regularization import em_regularize_covariance
from cvx.covariance.regularization import regularize_covariance
from .iterated_ewma_vec import ewma


def _realized_covariance(returns):
    for time in returns.index.unique():
        returns_temp = returns.loc[time]
        T_temp = len(returns_temp)

        yield time, returns_temp.cov(ddof=0) * T_temp


def realized_ewma(returns, halflife, clip_at=None, min_periods=None):
    realized_covariances = dict(_realized_covariance(returns))

    return ewma(
        realized_covariances,
        halflife,
        clip_at=clip_at,
        min_periods=min_periods,
    )


def realized_volas(returns):
    for time in returns.index.unique():
        yield time, np.sqrt((returns.loc[time] ** 2).sum(axis=0))


def MSE(returns, covariances):
    '''
    this function calculates the Mean Squared Error between predicted and realized covariance matrices for financial returns.
    '''
    returns_shifted = returns.shift(-1)

    MSEs = []
    for time, cov in covariances.items():
        realized_cov = returns_shifted.loc[time].values.reshape(-1, 1) @ returns_shifted.loc[time].values.reshape(1, -1) # here there is a reshape to a column vector (reshape(-1, 1)) and a row vector (reshape(1, -1))
        MSEs.append(np.linalg.norm(cov - realized_cov) ** 2) # this is a frobenius norm

    return pd.Series(MSEs, index=covariances.keys())


def RMSE(datasetWithPercentageChange, predictedCovariancesDict, realCovariancesDict, startDate):
    '''
    This function calculates the Root Mean Squared Error quarter by quarter between predicted and realized covariance matrices for financial returns.
    So this function returns a vector of RMSEs, one for each quarter.
    '''
    # define a list of residuals (difference between predicted and realized covariance matrices)
    residuals = []
    RMSEs = {} # this is the dictionary of RMSEs(one value for each quarter)

    # take the inital month from the start date
    initialMonth = datasetWithPercentageChange.index[0].month
    initialQuarter = (initialMonth - 1) // 3 + 1
    tempQuarter = initialQuarter

    N = len(datasetWithPercentageChange.loc[(datasetWithPercentageChange.index.year == startDate.year) & (datasetWithPercentageChange.index.quarter == initialQuarter)]) # get the number of days in the first quarter

    for t in datasetWithPercentageChange.index:

        if t not in predictedCovariancesDict or t not in realCovariancesDict:
            continue

        # get the quarter of the current date
        quarter = (t.month - 1) // 3 + 1

        # if the quarter has changed, calculate the RMSE
        if quarter != tempQuarter:

            # get the timestamp of"yesterday"
            yesterday = t - pd.Timedelta(days=1)

            # calculate the RMSE
            residualsSum = sum(residuals) # sum of the residuals
            RMSE = np.sqrt(residualsSum / N) # calculate the RMSE
            RMSEs[yesterday] = RMSE # append the RMSE to the list of RMSEs
            # reset the variables
            tempQuarter = quarter
            N = len(datasetWithPercentageChange.loc[(datasetWithPercentageChange.index.year == t.year) & (datasetWithPercentageChange.index.quarter == quarter)]) # get the number of days in the quarter
            residuals = [] # reset the residuals list

        predictedCovarianceMatrix = predictedCovariancesDict[t] # get the predicted covariance matrix at time t
        realCovarianceMatrix = realCovariancesDict[t] # get the realized covariance matrix at time t

        # calculate the residual
        residual = (np.linalg.norm(realCovarianceMatrix - predictedCovarianceMatrix)) ** 2 # this is a frobenius norm

        # append the residual to the list of residuals
        residuals.append(residual)
    
    # calculate the RMSE for the last quarter
    residualsSum = sum(residuals) # sum of the residuals
    RMSE = np.sqrt(residualsSum / N) # calculate the RMSE

    # get the timestamp of"yesterday"
    yesterday = t - pd.Timedelta(days=1)
    RMSEs[yesterday] = RMSE # append the RMSE to the list of RMSEs

    # now check if inside the RMSEs dictionary there are 0 values, if so, remove them
    for key in list(RMSEs.keys()):
        if RMSEs[key] == 0:
            del RMSEs[key]

    return RMSEs


def RMSEforSingleVolatility(datasetWithPercentageChange, predictedVolatilityDict, realVolatilityDict, startDate):
    '''
    This function calculates the Root Mean Squared Error quarter by quarter between predicted and realized volatility of the specified asset(like aapl, ibm, mcd...).
    So this function returns a vector of RMSEs, one for each quarter.
    '''
    # define a list of residuals (difference between predicted and realized covariance matrices)
    volatilityResiduals = []
    volatilityRMSEsDict = {} # this is the dict of RMSEs(one value for each quarter)

    # take the inital month from the start date
    initialMonth = startDate.month
    initialQuarter = (initialMonth - 1) // 3 + 1
    tempQuarter = initialQuarter

    N = len(datasetWithPercentageChange.loc[(datasetWithPercentageChange.index.year == startDate.year) & (datasetWithPercentageChange.index.quarter == initialQuarter)]) # get the number of days in the first quarter

    for t in datasetWithPercentageChange.index:

        if t not in predictedVolatilityDict or t not in realVolatilityDict:
            continue

        # get the quarter of the current date
        quarter = (t.month - 1) // 3 + 1

        # if the quarter has changed, calculate the RMSE
        if quarter != tempQuarter:

            # get the timestamp of"yesterday"
            yesterday = t - pd.Timedelta(days=1)

            # calculate the RMSE
            residualsSum = sum(volatilityResiduals) # sum of the residuals
            RMSE = np.sqrt(residualsSum / N) # calculate the RMSE
            volatilityRMSEsDict[yesterday] = RMSE # append the RMSE to the list of RMSEs

            # reset the variables
            tempQuarter = quarter
            N = len(datasetWithPercentageChange.loc[(datasetWithPercentageChange.index.year == t.year) & (datasetWithPercentageChange.index.quarter == quarter)]) # get the number of days in the quarter
            volatilityResiduals = [] # reset the residuals list

        predictedAssetVolatility = predictedVolatilityDict[t] # get the predicted volatility at time t

        realAssetVolatility = realVolatilityDict[t] # get the realized volatility at time t

        # calculate the residual
        residual = (realAssetVolatility - predictedAssetVolatility) ** 2

        # append the residual to the list of residuals
        volatilityResiduals.append(residual)
    
    # calculate the RMSE for the last quarter
    residualsSum = sum(volatilityResiduals) # sum of the residuals
    RMSE = np.sqrt(residualsSum / N) # calculate the RMSE

    # get the timestamp of"yesterday"
    yesterday = t - pd.Timedelta(days=1)
    volatilityRMSEsDict[yesterday] = RMSE # append the RMSE to the list of RMSEs

    # remove 0 values from the dict
    for key in list(volatilityRMSEsDict.keys()):
        if volatilityRMSEsDict[key] == 0:
            del volatilityRMSEsDict[key]

    return volatilityRMSEsDict


def yearly_SR(trader, plot=True, regression_line=True):
    rets = pd.Series(trader.rets.flatten(), index=trader.returns.index)

    # Only keep years with more than 100 trading days
    rets = rets.groupby(rets.index.year).filter(lambda x: len(x) > 100)

    means = rets.resample("Y").mean() * 252
    stds = rets.resample("Y").std() * np.sqrt(252)

    SRs = means / stds

    if plot:
        # Fit regression line to SRs
        coefficients = np.polyfit(SRs.index.year + 1, SRs.values, 1)
        sr_pred = np.polyval(coefficients, SRs.index.year)

        plt.plot(SRs, marker="o")
        if regression_line:
            plt.plot(SRs.index, sr_pred, "--", color="red", alpha=0.5, label="Trend")
        plt.ylabel("Sharpe ratio")
        plt.legend()
    return SRs


def turnover(weights):
    """
    Computes average turnover for a sequence of weights,
    i.e., mean of |w_{t+1}-w_{t}|_1
    """
    w_diff = weights[1:] - weights[:-1]
    w_old = weights[:-1]

    daily_turnover = np.mean(
        np.sum(np.abs(w_diff), axis=1) / np.sum(np.abs(w_old), axis=1)
    )

    yearly_turnover = 252 * daily_turnover

    return yearly_turnover * 100


def _single_log_likelihood(r, Sigma, m):
    n = len(r)
    Sigma_inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    return (
        -n / 2 * np.log(2 * np.pi)
        - 1 / 2 * np.log(det)
        - 1 / 2 * (r - m).T @ Sigma_inv @ (r - m)
    )


def log_likelihood_low_rank(returns, Sigmas, means=None):
    """
    Computes the log-likelihoods

    param returns: pandas DataFrame of returns
    param Sigmas: dictionary of covariance matrices where each covariance matrix
                  is a namedtuple with fields "F" and "d"
    param means: pandas DataFrame of means

    Note: Sigmas[time] is covariance prediction for returns[time+1]
        same for means.loc[time]
    """
    returns = returns.shift(-1)

    ll = []
    m = np.zeros_like(returns.iloc[0].values).reshape(-1, 1)

    times = []

    for time, low_rank in Sigmas.items():
        # TODO: forming the covariance matrix is bad...
        cov = low_rank.F @ (low_rank.F).T + np.diag(low_rank.d)

        if not returns.loc[time].isna()[0]:
            if means is not None:
                m = means.loc[time].values.reshape(-1, 1)
            ll.append(
                _single_log_likelihood(
                    returns.loc[time].values.reshape(-1, 1), cov.values, m
                )
            )
            times.append(time)

    return pd.Series(ll, index=times).astype(float)


def log_likelihood_regularized(returns, Sigmas, means=None, r=None):
    """
    Helper function to avoid storing all the covariance matrices in memory for
    large universe experiments

    param returns: pandas DataFrame of returns
    param Sigmas: dictionary of covariance matrices
    param r: float, rank of low rank component

    Note: Sigmas[time] is covariance prediction for returns[time+1]
    """
    returns = returns.shift(-1)

    ll = []
    m = np.zeros_like(returns.iloc[0].values).reshape(-1, 1)

    times = []

    if r is not None:
        for time, cov in regularize_covariance(Sigmas, r=r):
            if not returns.loc[time].isna()[0]:
                if means is not None:
                    m = means.loc[time].values.reshape(-1, 1)
                ll.append(
                    _single_log_likelihood(
                        returns.loc[time].values.reshape(-1, 1), cov.values, m
                    )
                )
                times.append(time)
    else:
        for time, cov in Sigmas.items():
            if not returns.loc[time].isna()[0]:
                if means is not None:
                    m = means.loc[time].values.reshape(-1, 1)
                ll.append(
                    _single_log_likelihood(
                        returns.loc[time].values.reshape(-1, 1), cov.values, m
                    )
                )
                times.append(time)

    return pd.Series(ll, index=times).astype(float)


def log_likelihood(returns, Sigmas, means=None, scale=1):
    """
    Computes the log likelihhod assuming Gaussian returns with covariance matrix
    Sigmas and mean vector means

    param returns: numpy array where rows are vector of asset returns
    param Sigmas: numpy array of covariance matrix
    param means: numpy array of mean vector; if None, assumes zero mean
    """
    if means is None:
        means = np.zeros_like(returns)
        print("means shape: ", means.shape)

    T, n = returns.shape
    print("T: ", T)
    print("n: ", n)

    returns = returns.reshape(T, n, 1)
    means = means.reshape(T, n, 1)

    print("returns shape: ", returns.shape)
    print("means shape: ", means.shape)

    # print the first 2 terms of the returns and means
    print("returns[0]: ", returns[0])
    print("returns[1]: ", returns[1])
    print("means[0]: ", means[0])
    print("means[1]: ", means[1])

    returns = returns * scale
    means = means * scale
    Sigmas = Sigmas * scale**2

    dets = np.linalg.det(Sigmas).reshape(len(Sigmas), 1, 1)
    Sigma_invs = np.linalg.inv(Sigmas)

    return (
        -n / 2 * np.log(2 * np.pi) - 1 / 2 * np.log(dets) - 1/2
        * np.transpose(returns - means, axes=(0, 2, 1)) @ Sigma_invs @ (returns - means)
    ).flatten()


def log_likelihood_for_test(returns, Sigmas, dates, means=None, scale=1):
    """
    this function is equal to the other log_likelihood function, but it saves the results in a csv file, so i can see the determinants
    value behavior and the matrix product behavior when i modify the number of assets inside my portfolio
    """
    if means is None:
        means = np.zeros_like(returns)

    T, n = returns.shape

    returns = returns.reshape(T, n, 1)
    means = means.reshape(T, n, 1)

    returns = returns * scale
    means = means * scale
    Sigmas = Sigmas * scale**2

    dets = np.linalg.det(Sigmas).reshape(len(Sigmas), 1, 1)
    Sigma_invs = np.linalg.inv(Sigmas)

    matrixProduct = np.transpose(returns - means, axes=(0, 2, 1)) @ Sigma_invs @ (returns - means)

    # print shape of dets and matrixProduct
    print("dets shape: ", dets.shape)
    print("matrixProduct shape: ", matrixProduct.shape)
    
    # dets shape is (T, 1, 1) and matrixProduct shape is (T, 1, 1), convert them to a list of scalars
    dets = dets.flatten()
    matrixProduct = matrixProduct.flatten()

    firstConstantTerm = -n / 2 * np.log(2 * np.pi)
    determinants = 1 / 2 * np.log(dets)

    logLikelihoodValue = (- n / 2 * np.log(2 * np.pi) - 1 / 2 * np.log(dets) - 1 / 2 * matrixProduct).flatten()

    # now save the results in a csv file, where every row is a time t and the first column is the determinant value and the second column is the matrix product value
    
    results = np.column_stack((dets, matrixProduct, logLikelihoodValue))
    np.savetxt("detsAndMatrixProduct" + str(n) + "Assets.csv", results, delimiter=",")
    print("results saved in detsAndMatrixProduct.csv")

    # Create a DataFrame for matrixProduct with dates
    matrix_product_df = pd.DataFrame({'Date': dates, 'MatrixProduct': 1/2 * matrixProduct})

    first_constant_term_df = pd.DataFrame({'Date': dates, 'FirstConstantTerm': firstConstantTerm})

    log_determinants_df = pd.DataFrame({'Date': dates, 'LogDeterminants': determinants})

    return logLikelihoodValue, matrix_product_df, first_constant_term_df, log_determinants_df


def log_likelihood_sequential(returns, Sigmas, means=None, scale=1):
    """
    Computes Gaussian log likelihood sequentially
    """
    if means is None:
        means = np.zeros_like(returns)

    T, n = returns.shape

    returns = returns.reshape(T, n, 1)
    means = means.reshape(T, n, 1)

    returns = returns * scale
    means = means * scale
    Sigmas = Sigmas * scale**2

    ll = np.zeros(T)

    for t in range(T):
        r = returns[t].reshape(n, 1)
        m = means[t].reshape(n, 1)
        Sigma = Sigmas[t]
        det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)

        ll[t] = (
            -n / 2 * np.log(2 * np.pi)
            - 1 / 2 * np.log(det)
            - 1 / 2 * (r - m).T @ Sigma_inv @ (r - m)
        )


def rolling_window(returns, memory, min_periods=20):
    min_periods = max(min_periods, 1)

    times = returns.index
    assets = returns.columns

    returns = returns.values

    Sigmas = np.zeros((returns.shape[0], returns.shape[1], returns.shape[1]))
    Sigmas[0] = np.outer(returns[0], returns[0])

    for t in range(1, returns.shape[0]):
        alpha_old = 1 / min(t + 1, memory)
        alpha_new = 1 / min(t + 2, memory)

        if t >= memory:
            Sigmas[t] = alpha_new / alpha_old * Sigmas[t - 1] + alpha_new * (
                np.outer(returns[t], returns[t])
                - np.outer(returns[t - memory], returns[t - memory])
            )
        else:
            Sigmas[t] = alpha_new / alpha_old * Sigmas[t - 1] + alpha_new * (
                np.outer(returns[t], returns[t])
            )

    Sigmas = Sigmas[min_periods - 1 :]
    times = times[min_periods - 1 :]

    return {
        times[t]: pd.DataFrame(Sigmas[t], index=assets, columns=assets)
        for t in range(len(times))
    }


def from_row_to_covariance(row, n):
    """
    Convert upper diagonal part of covariance matrix to a covariance matrix
    """
    Sigma = np.zeros((n, n))

    # set upper triangular part
    upper_mask = np.triu(np.ones((n, n)), k=0).astype(bool)
    Sigma[upper_mask] = row

    # set lower triangular part
    lower_mask = np.tril(np.ones((n, n)), k=0).astype(bool)
    Sigma[lower_mask] = Sigma.T[lower_mask]
    return Sigma


def from_row_matrix_to_covariance(M, n):
    """
    Convert Tx(n(n+1)/2) matrix of upper diagonal parts of covariance matrices to a Txnxn matrix of covariance matrices
    """
    Sigmas = []
    T = M.shape[0]
    for t in range(T):
        Sigmas.append(from_row_to_covariance(M[t], n)) # this is a list of covariance matrices
    return np.array(Sigmas)


def add_to_diagonal(Sigmas, lamda):
    """
    Adds lamda*diag(Sigma) to each covariance (Sigma) matrix in Sigmas

    param Sigmas: dictionary of covariance matrices
    param lamda: scalar
    """
    for key in Sigmas.keys():
        Sigmas[key] = Sigmas[key] + lamda * np.diag(np.diag(Sigmas[key]))

    return Sigmas


def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y


def em_low_rank_log_likelihood(returns, Sigmas, rank):
    Sigmas_low_rank = dict(regularize_covariance(Sigmas, r=rank, low_rank_format=True))
    Sigmas_em = dict(em_regularize_covariance(Sigmas, Sigmas_low_rank))

    return log_likelihood_low_rank(returns, Sigmas_em)
