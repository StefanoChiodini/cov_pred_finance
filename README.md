# [cvxcovariance](http://www.cvxgrp.org/cov_pred_finance)

[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cov_pred_finance/badge.svg)](https://coveralls.io/github/cvxgrp/cov_pred_finance)
[![PyPI version](https://badge.fury.io/py/cvxcovariance.svg)](https://badge.fury.io/py/cvxcovariance)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
[![PyPI download month](https://img.shields.io/pypi/dm/cvxcovariance.svg)](https://pypi.python.org/pypi/cvxcovariance/)


The `cvxcovariance` package
provides simple tools for creating an estimate $\hat\Sigma_t$ of the covariance $\Sigma_t$ of the $n$-dimensional return vectors $r_t$, $t=1,2,\ldots$, based on the observed returns $r_1, \ldots, r_{t-1}$. (Here $r_t$ is the return from $t-1$ to $t$.) The covariance predictor $\hat\Sigma_t$ is generated by blending $K$ different "expert" predictors $\hat\Sigma_t^{(1)},\ldots,\hat\Sigma_t^{(K)}$, by solving a convex optimization problem at each time step.

For a detailed description of the methodology, see our manuscript [A Simple Method for Predicting
Covariance Matrices of Financial Returns](https://web.stanford.edu/~boyd/papers/cov_pred_finance.html) (in particular Section 3).

In the simplest case the user provides a $T\times n$ pandas DataFrame
of returns $r_1,\ldots,r_T$ and $K$ half-life pairs, and gets back covariance predictors for each time
step. (The $K$ experts are computed as iterated exponentially weighted moving average (IEWMA) predictors as described in Section 2.6 of the [paper](https://web.stanford.edu/~boyd/papers/cov_pred_finance.html).) In the more general case, the user provides the $K$ expert predictors $\hat\Sigma_t^{(1)},\ldots,\hat\Sigma_t^{(K)}$, $t=1,\ldots,T$, and these are blended together by solving the convex optimization problems. In either case the result is returned as an iterator object over namedtuples: `Result = namedtuple("Result", ["time", "mean", "covariance", "weights"])`.


Note: at time $t$ the user is provided with $\Sigma_{t+1}$,
$\textit{i.e.}$, the covariance matrix for the next time step. So `Result.covariance` returns the covariance prediction for `time+1`.

## Installation
To install the package, run the following command in the terminal:

```bash
pip install cvxcovariance
```

## Usage
There are two alternative ways to use the package. The first is to use the
`from_ewmas` function to create a combined multiple IEWMA (CM-IEWMA) predictor. The second is to provide your own covariance experts, via dictionaries, and pass them to the `from_sigmas` function. Both functions return an object of the `_CovarianceCombination` class, which can be used to solve the covariance combination problem.

### CM-IEWMA
The `from_ewmas` function takes as input a pandas DataFrame of
returns and the IEWMA half-life pairs (each pair consists of one half-life for volatility estimation and one for correlation estimation), and returns an iterator object that
iterates over the CM-IEWMA covariance predictors defined via namedtuples. Through the namedtuple you can access the `time`, `mean`, `covariance`, and `weights` attributes. `time` is the timestamp. `mean` is the estimated mean of the return at the $\textit{next}$ timestamp, $\textit{i.e.}$ `time+1`, if the user wants to estimate the mean; per default the mean is set to zero, which is a reasonable assumption for many financial returns. `covariance` is the estimated covariance matrix for the $\textit{next}$ timestamp, $\textit{i.e.}$ `time+1`. `weights` are the $K$ weights attributed to the experts. Here is an example:

```python
import pandas as pd
from cvx.covariance.combination import from_ewmas

# Load return data
returns = pd.read_csv("data/ff5.csv", index_col=0, header=0, parse_dates=True).iloc[:1000]
n = returns.shape[1]

# Define half-life pairs for K=3 experts, (halflife_vola, halflife_cov)
halflife_pairs = [(10, 21), (21, 63), (63, 125)]

# Define the covariance combinator
combinator = from_ewmas(returns,
                        halflife_pairs,
                        min_periods_vola=n,  # min periods for volatility estimation
                        min_periods_cov=3 * n)  # min periods for correlation estimation (must be at least n)

# Solve combination problem and loop through combination results to get predictors
covariance_predictors = {}
for predictor in combinator.solve(window=10):  # lookback window in convex optimization problem
    # From predictor we can access predictor.time, predictor.mean (=0 here), predictor.covariance, and predictor.weights
    covariance_predictors[predictor.time] = predictor.covariance
```
Here `covariance_predictors[t]` is the covariance prediction for time $t+1$, $\textit{i.e.}$, it is uses knowledge of $r_1,\ldots,r_t$.

### General covariance combination
The `from_sigmas` function takes as input a pandas DataFrame of
returns and a dictionary of covariance predictors `{key: {time:
sigma}`, where `key` is the key of an expert predictor and `{time:
sigma}` is the expert predictions. For example, here we combine two EWMA covariance predictors from pandas:

```python
import pandas as pd
from cvx.covariance.combination import from_sigmas

# Load return data
returns = pd.read_csv("data/ff5.csv", index_col=0, header=0, parse_dates=True).iloc[:1000]
n = returns.shape[1]

# Define 21 and 63 day EWMAs as dictionaries (K=2 experts)
ewma21 = returns.ewm(halflife=21, min_periods=5 * n).cov().dropna()
expert1 = {time: ewma21.loc[time] for time in ewma21.index.get_level_values(0).unique()}
ewma63 = returns.ewm(halflife=63, min_periods=5 * n).cov().dropna()
expert2 = {time: ewma63.loc[time] for time in ewma63.index.get_level_values(0).unique()}

# Create expert dictionary
experts = {1: expert1, 2: expert2}

# Define the covariance combinator
combinator = from_sigmas(sigmas=experts, returns=returns)

# Solve combination problem and loop through combination results to get predictors
covariance_predictors = {}
for predictor in combinator.solve(window=10):
    # From predictor we can access predictor.time, predictor.mean (=0 here), predictor.covariance, and predictor.weights
    covariance_predictors[predictor.time] = predictor.covariance
```
Here `covariance_predictors[t]` is the covariance prediction for time $t+1$, $\textit{i.e.}$, it is uses knowledge of $r_1,\ldots,r_t$.

## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
poetry install
```

to replicate the virtual environment we have defined in pyproject.toml.

## Kernel

We install [JupyterLab](https://jupyter.org) within your new virtual
environment. Executing

```bash
./create_kernel.sh
```

constructs a dedicated
[Kernel](https://docs.jupyter.org/en/latest/projects/kernels.html) for the
project.

## Citing
If you want to reference our paper in your research, please consider citing us by using the following BibTeX:

```BibTeX
@misc{johansson2023covariance,
      title={A Simple Method for Predicting Covariance Matrices of Financial Returns},
      author={Kasper Johansson and Mehmet Giray Ogut and Markus Pelger and Thomas Schmelzer and Stephen Boyd},
      year={2023},
      eprint={2305.19484},
      archivePrefix={arXiv},
}
```
