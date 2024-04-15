import numpy as np
import pandas as pd

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


# read from this file the covariance matrices calculated for the test phase
mgarch_cond_cov = pd.read_csv("experiments/mgarch_predictors_from_R/stocks/mgarch_stocks_adj.csv", index_col=None)

# here we are obtaining the covariance matrix calculated for every day from the csv file;
Sigmas = from_row_matrix_to_covariance(mgarch_cond_cov.values,25) / 10000 # returns.shape[1] gives the number of columns in the returns DataFrame, which corresponds to the number of assets in the portfolio

# print the shape of first element of sigmas
print(Sigmas[0].shape) # (25, 25)

# now for every element in Sigmas calculate the trace of the matrix and store it in a list
traces = [np.trace(Sigma) for Sigma in Sigmas]

# do the mean of the trace 
mean_trace = np.mean(traces)

print(mean_trace) # 0.000225