import numpy as np
import pandas as pd
from collections import namedtuple

### This is vectorized iterated EWMA

# named tuple
IEWMA = namedtuple('IEWMA', ['mean', 'covariance'])


def get_next_ewma(EWMA, y_last, t, beta):
    """
    param EWMA: EWMA at time t-1
    param y_last: observation at time t-1
    param t: current time step
    param beta: EWMA exponential forgetting parameter

    returns: EWMA estimate at time t (note that this does not depend on y_t)
    """

    old_weight = (beta-beta**t)/(1-beta**t)
    new_weight = (1-beta) / (1-beta**t)

    return old_weight*EWMA + new_weight*y_last

def ewma(y, halflife):
    """
    y: array with measurements for times t=1,2,...,T=len(y)
    halflife: EWMA half life

    returns: list of EWMAs for times t=2,3,...,T+1 = len(y)


    Note: We define EWMA_t as a function of the 
    observations up to time t-1. This means that
    y = [y_1,y_2,...,y_T] (for some T), while
    EWMA = [EWMA_2, EWMA_3, ..., EWMA_{T+1}]
    This way we don't get a "look-ahead bias" in the EWMA
    """

    beta = np.exp(-np.log(2)/halflife)
    EWMA_t = 0
    EWMAs = []
    for t in range(1,y.shape[0]+1): # First EWMA is for t=2 
        y_last = y[t-1] # Note zero-indexing
        EWMA_t = get_next_ewma(EWMA_t, y_last, t, beta)
        EWMAs.append(EWMA_t)
    return np.array(EWMAs)


def _get_realized_covs(returns):
    """
    param returns: numpy array where rows are vector of asset returns for t=0,1,...
        returns has shape (T, n) where T is the number of days and n is the number of assets

    returns: (numpy array) list of r_t*r_t' (matrix multiplication) for all days, i.e,
        "daily realized covariances"
    """
    T = returns.shape[0]
    n = returns.shape[1]
    returns = returns.reshape(T,n,1)

    return returns @ returns.transpose(0,2,1)
    

def _get_realized_vars(returns):
    """
    param returns: numpy array where rows are vector of asset returns for t=0,1,...
        returns has shape (T, n) where T is the number of days and n is the number of assets

    returns: (numpy array) list of diag(r_t^2) for all days, i.e,\
        "daily realized variances"
    """


    variances = []
    
    for t in range(returns.shape[0]):
        r_t = returns[t, :].reshape(-1,)
        variances.append(np.diag(r_t**2))
    return np.array(variances)

def _get_r_adj(returns, D_inv):
    """
    param D_inv: Txnxn numpy array of inverse of diagonal volatility matrices
    param param returns: array of returns to whiten for t=1,2,..., 

    returns: numpy array with r_tilde_hats as rows
    """
    T = returns.shape[0]
    n = returns.shape[1]
    returns = returns.reshape(T,n,1)
    return D_inv @ returns

def _refactor_to_corr(Sigmas):
    """
    Returns correlation matrices from covariance matrices

    param Sigmas: Txnxn numpy array of covariance matrices
    """
    V = np.sqrt(np.diagonal(Sigmas, axis1=1, axis2=2))
    outer_V = np.array([np.outer(v, v) for v in V])
    return Sigmas / outer_V




def iterated_ewma(returns, vola_halflife, cov_halflife, lower=None, upper=None,\
    min_periods=20, mean=False):
        """
        param returns: pandas dataframe with returns for each asset
        param vola_halflife: half life for volatility
        param cov_halflife: half life for covariance
        param lower: lower bound for adjusted return cutoff
        param upper: upper bound for adjusted return cutoff
        param min_periods: minimum number of periods to use for EWMA
        param mean: whether to estimate mean; if False, assumes zero mean data

        returns: dictionary with covariance matrix predictions for each day\
            each key (time step) in the dictionary corresponds to the
            prediction for the following (next) key (time step)
        """
        # TODO: How to handle lower=None, upper=None?
        lower = lower or -1000
        upper = upper or 1000

        if mean:
            min_periods = max(min_periods, 3) # Need to remove two entries
            returns_mean = ewma(returns.values, halflife=vola_halflife)
        else:
            returns_mean = np.zeros_like(returns)



        ### Scaling  
        realized_vars = _get_realized_vars(returns.values-returns_mean)
        if mean:
            # The first entry contains zeros
            realized_vars = realized_vars[1:]
            returns_mean = returns_mean[1:]
            returns = returns.iloc[1:].copy()


        V = ewma(realized_vars, halflife=vola_halflife)
        D = np.sqrt(V)
        D_inv_diags = 1/np.diagonal(D, axis1=1, axis2=2)
        D_inv = np.stack([np.diag(V) for V in D_inv_diags]) # Gets L^T s.t. LL^T = V_hat^{-1}
        returns_adj = _get_r_adj(returns.values-returns_mean, D_inv).clip(min=lower, max=upper)

        ### Full whitening, outer products in T x n x n tensor
        if mean:
            returns_adj_mean = ewma(returns_adj, halflife=cov_halflife)
        else:
            returns_adj_mean = np.zeros_like(returns_adj)
        realized_covs = _get_realized_covs(returns_adj-returns_adj_mean)
        if mean:
            # The first entry contains zeros only
            realized_covs = realized_covs[1:]
            returns_adj_mean = returns_adj_mean[1:]
            returns_adj = returns_adj[1:]
            returns_mean = returns_mean[1:]
            returns = returns.iloc[1:].copy()
            D = D[1:]

        # Create correlation matrix
        R_tilde = ewma(realized_covs, halflife=cov_halflife)
        R = _refactor_to_corr(R_tilde)

        # Remove first min_periods-1 entries
        if mean:
            # We already removed the first two entries
            burnin = min_periods-3
        else:  
            burnin = min_periods-1
        R = R[burnin:, :, :]
        D = D[burnin:, :, :]
        returns_mean = returns_mean[burnin:, :]
        # returns_adj = returns_adj[burnin:, :, :]
        returns_adj_mean = returns_adj_mean[burnin:, :]
        times = returns.index[burnin:]

        # Engle 2002 formula
        Sigmas = D @ R @ D

        if mean:
            T, n = R.shape[0], R.shape[1]    

            means = returns_mean + (D @ returns_adj_mean).reshape(T,n)
            means = {times[i]: pd.Series(means[i], index = returns.columns) for i in range(len(times))}
            covariances = {times[i]: pd.DataFrame(Sigmas[i], index = returns.columns, columns = returns.columns) for i in range(len(times))}

            return IEWMA(mean=means, covariance=covariances)
        else:

            return {times[i]: pd.DataFrame(Sigmas[i], index = returns.columns, columns = returns.columns) for i in range(len(times))}

    

