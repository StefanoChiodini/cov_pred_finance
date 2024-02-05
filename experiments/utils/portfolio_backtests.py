from __future__ import annotations

import warnings
from abc import ABC
from abc import abstractmethod

import pandas as pd
from tqdm import trange

# Mute specific warning
warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")

from .trading_model import *


def _get_causal(returns, Sigmas, start_date, end_date, mus=None):
    """
    this function restituisce returns and Sigmas with keys in the interval [start_date, end_date].
    In particular: This function is a preparatory step that gets the historical returns and corresponding covariance matrices (and expected returns if provided) ready for the period over which backtesting is to be conducted. It ensures the data is causally aligned, meaning past data is used to predict future returns, adhering to a realistic trading scenario where future information is not known at the time of making predictions.
    """
    returns_temp = returns.loc[start_date:end_date].shift(-1).dropna()
    Sigmas_temp = {time: Sigmas[time] for time in returns_temp.index}

    if mus is not None:
        mus_temp = pd.DataFrame({time: mus.loc[time] for time in returns_temp.index}).T
        return returns_temp, Sigmas_temp, mus_temp

    return returns_temp, Sigmas_temp


def _create_table_helper(metrics, prescient=True):
    '''
    this function is used to create the table for the results, the printed code is in latex format
    '''

    print("\\begin{tabular}{lcccc}")
    print("   \\toprule")
    print("   {Predictor} & {Return} & {Risk} & {Sharpe} & {Drawdown} \\\\")
    print("   \\midrule")

    for name, metric in metrics.items():
        if name != "PRESCIENT":
            print(
                "   {} & {:.1f}\\% & {:.1f}\\% & {:.1f} & {:.0f}\\% \\\\".format(
                    name,
                    metric.mean_return * 100,
                    metric.risk * 100,
                    metric.sharpe,
                    metric.drawdown * 100,
                )
            )
    print("   \\hline")
    if prescient:
        metric = metrics["PRESCIENT"]
        print(
            "   {} & {:.1f}\\% & {:.1f}\\% & {:.1f} & {:.0f}\\% \\\\".format(
                name,
                metric.mean_return * 100,
                metric.risk * 100,
                metric.sharpe,
                metric.drawdown * 100,
            )
        )
    print("   \\bottomrule")
    print("\\end{tabular}")


def create_table(traders, sigma_tar, rf, excess, prescient=True):
    """
    this function calls the function get_metrics from the Trader Class to calculate the metrics for each predictor
    param traders: dict of Trader Class objects
    param sigma_tar: target volatility
    param rf: risk free rate
    param excess: True if excess returns is used in computation of Sharpe ratio etc., False otherwise
    """
    #print("traders in create table: ", traders)
    #print("sigma_tar in create table: ", sigma_tar)
    metrics = {}
    # FOR LOOP: iterate through the traders dictionary
    for name, trader in traders.items():
        
        # print the name of the predictor
        print("\n\nname: ", name)
        print("the trader object is: ", trader)
        if sigma_tar:
            # call the get_metrics function from the Trader Class to calculate the metrics for each predictor; here we are calling the get_metrics function on the trader object
            metrics[name] = trader.get_metrics(
                diluted_with_cash=True, sigma_des=sigma_tar, rf=rf, excess=excess
            )
        else:  # TODO: for now this means already trades cash
            metrics[name] = trader.get_metrics(
                diluted_with_cash=False, rf=rf, excess=excess
            )
    print("metrics before calling _create_table_helper function: ", metrics)
    _create_table_helper(metrics, prescient)


class PortfolioBacktest(ABC):
    def __init__(
        self,
        returns,
        cov_predictors,
        names,
        mean_predictors=None,
        start_date=None,
        end_date=None,
    ):
        """
        param returns: pd.DataFrame with returns
        param cov_predictors: list of covariance predictors
        param names: list of names of the predictors
        param mean_predictors: list of mean predictors
        param start_date: start date of the backtest
        param end_date: end date of the backtest

        Note for cov_predictor in cov_predictors: cov_predictor[time] is the
        covariance predictor for time+1, i.e., cov_predictor[time] used
        information up to and including returns[time] (the Portfolio backtest
        Class takes care of the shifting)
        """
        for cov_predictor in cov_predictors:
            assert set(cov_predictor.keys()).issubset(set(returns.index))
        if mean_predictors is not None:
            for mean_predictor in mean_predictors:
                assert set(cov_predictor.keys()).issubset(set(mean_predictor.index))
        assert start_date is None or start_date in list(cov_predictor.keys())
        assert end_date is None or end_date in list(cov_predictor.keys())

        self.returns = returns
        self.cov_predictors = cov_predictors
        self.mean_predictors = mean_predictors
        self.names = names
        self.start_date = start_date or list(cov_predictor.keys())[0]
        self.end_date = end_date or list(cov_predictor.keys())[-1]

    @abstractmethod
    def backtest(self):
        pass


class EqWeighted(PortfolioBacktest):
    def __init__(self, returns, cov_predictors, names, start_date=None, end_date=None):
        super().__init__(
            # cov_preditors is a list of dictionaries; where each dictionary is a covariance predictor
            # the keys of the dictionary are the dates and the values are the covariance matrices
            returns, cov_predictors, names, start_date=start_date, end_date=end_date
        )

    #  This function aims to conduct a backtest for an equally weighted (eq_weighted) portfolio strategy using different sets of covariance predictors 
    def backtest(self):
        adjust_factor = 1
        traders_eq_w = {}

        # this for loop is used to iterate through the different covariance predictors
        for i in trange(len(self.cov_predictors)):
            # it is assigning the i-th dictionary from the self.cov_predictors list to the variable Sigma_hats. This variable now contains the specific set of covariance predictions (with their corresponding timestamps)
            # so now sigma_hats is a dictionary with keys being the timestamps and values being the covariance matrices
            Sigma_hats = self.cov_predictors[i]

            '''
            now call the get causal function using the returns(the one take from the csv file), sigma_hats, start_date and end_date parameters
            From this function we get returns and covariance matrices with keys in the interval [start_date, end_date].
            Of course returns are still the ones from the csv file, but now they are shifted by one day and the keys are in the interval [start_date, end_date]
            and the covariance matrices are the ones from the specific predictor (Sigma_hats) and the keys are in the interval [start_date, end_date]
            '''
            
            returns_temp, Sigmas_temp = _get_causal(
                self.returns, Sigma_hats, self.start_date, self.end_date
            )

            # here just initialize a trader instance: an instance of the Trader class
            trader = Trader(returns_temp, Sigmas_temp)

            # here is calling the backtest function from the Trader class to conduct the backtest for the eq_weighted portfolio strategy
            trader.backtest(portfolio_type="eq_weighted", adjust_factor=adjust_factor)

            traders_eq_w[self.names[i]] = trader
            # after this assignment, traders_eq_w is a dictionary with keys being the names of the predictors and values being the trader instances
            # and every trader instance contains returns(the one take from the csv file), covariance matrices and the weights(weights that must be assigned to every asset) for the eq_weighted portfolio strategy
            # the weights are the portfolio weights: so equal weights are equeally distributed among the assets; minimum variance weights are the weights that minimize the variance of the portfolio; etc.
        return traders_eq_w


class MinRisk(PortfolioBacktest):
    def __init__(self, returns, cov_predictors, names, start_date=None, end_date=None):
        super().__init__(
            returns, cov_predictors, names, start_date=start_date, end_date=end_date
        )

    def backtest(self, additonal_cons):
        adjust_factor = 1
        traders_min_risk = {}
        for i in trange(len(self.cov_predictors)):
            Sigma_hats = self.cov_predictors[i]
            returns_temp, Sigmas_temp = _get_causal(
                self.returns, Sigma_hats, self.start_date, self.end_date
            )
            trader = Trader(returns_temp, Sigmas_temp)
            trader.backtest(
                portfolio_type="min_risk",
                adjust_factor=adjust_factor,
                additonal_cons=additonal_cons,
            )
            traders_min_risk[self.names[i]] = trader

        return traders_min_risk


class MaxDiverse(PortfolioBacktest):
    def __init__(self, returns, cov_predictors, names, start_date=None, end_date=None):
        super().__init__(
            returns, cov_predictors, names, start_date=start_date, end_date=end_date
        )

    def backtest(self, additonal_cons):
        adjust_factor = 1
        traders_max_diverse = {}
        for i in trange(len(self.cov_predictors)):
            Sigma_hats = self.cov_predictors[i]
            returns_temp, Sigmas_temp = _get_causal(
                self.returns, Sigma_hats, self.start_date, self.end_date
            )
            trader = Trader(returns_temp, Sigmas_temp)
            trader.backtest(
                portfolio_type="max_diverse",
                adjust_factor=adjust_factor,
                additonal_cons=additonal_cons,
            )
            traders_max_diverse[self.names[i]] = trader

        return traders_max_diverse


class RiskParity(PortfolioBacktest):
    def __init__(self, returns, cov_predictors, names, start_date=None, end_date=None):
        super().__init__(
            returns, cov_predictors, names, start_date=start_date, end_date=end_date
        )

    def backtest(self):
        adjust_factor = 1
        traders_risk_par = {}
        for i in trange(len(self.cov_predictors)):
            Sigma_hats = self.cov_predictors[i]
            returns_temp, Sigmas_temp = _get_causal(
                self.returns, Sigma_hats, self.start_date, self.end_date
            )
            trader = Trader(returns_temp, Sigmas_temp)
            trader.backtest(portfolio_type="risk_parity", adjust_factor=adjust_factor)
            traders_risk_par[self.names[i]] = trader

        return traders_risk_par


class MeanVariance(PortfolioBacktest):
    def __init__(
        self,
        returns,
        cov_predictors,
        names,
        mean_predictors,
        start_date=None,
        end_date=None,
    ):
        super().__init__(
            returns,
            cov_predictors,
            names,
            mean_predictors=mean_predictors,
            start_date=start_date,
            end_date=end_date,
        )

    def backtest(self, additonal_cons, sigma_tar, rhos=None):
        adjust_factor = 1
        traders_mean_var = {}
        for i in trange(len(self.cov_predictors)):
            mu_hats = self.mean_predictors[i]
            Sigma_hats = self.cov_predictors[i]
            returns_temp, Sigmas_temp, mus_temp = _get_causal(
                self.returns, Sigma_hats, self.start_date, self.end_date, mus=mu_hats
            )
            trader = Trader(returns_temp, Sigmas_temp, r_hats=mus_temp)
            trader.backtest(
                portfolio_type="mean_variance",
                adjust_factor=adjust_factor,
                additonal_cons=additonal_cons,
                sigma_des=sigma_tar,
                rhos=rhos,
            )
            traders_mean_var[self.names[i]] = trader

        return traders_mean_var
