from __future__ import annotations

import multiprocessing as mp
from collections import namedtuple

import cvxpy as cp
import numpy as np
import pandas as pd


def _get_L_inv(covariance):
    Theta = np.linalg.inv(covariance)
    return np.linalg.inv(np.linalg.cholesky(Theta))


metrics = namedtuple("metrics", ["mean_return", "risk", "sharpe", "drawdown"])


def construct_rho(returns, n_devs=2):
    stdevs = returns.ewm(halflife=125).std().shift(1)

    # set first row to zeros
    stdevs.iloc[0] = 0

    # Replace nans with zeros
    stdevs = stdevs.fillna(0)

    return n_devs * stdevs


class Trader:
    def __init__(self, returns, Sigma_hats, r_hats=None, rhos=None):
        """
        param R: nxT pandas dataframe of T returns from n assets; index is date.
        param Sigma_hats: dictionary of causal covariance matrices; keys are dates; values are n x n pandas dataframes(so values are covariance matrixes). They are causal in the sense that
            Sigma_hat[time] does NOT depend on returns[time]
        param r_hats: pandas DataFrame of causal expected returns, i.e., r_hat[time]
            does NOT depend on returns[time]
        """
        # here returns are still the returns of the csv file but filtered with the backtest period
        self.returns = returns

        # here sigmas are the sigmas calculated with the predictor mathematical model formula
        self.Sigma_hats = Sigma_hats

        # r_ hats are the expected returns 
        self.r_hats = r_hats

        # It might be used later to indicate whether the portfolio has been diluted with cash or not.
        self.diluted = False

        assert list(self.returns.index) == list(self.Sigma_hats.keys())

        # Generate list of standard deviations; so sigma_hats is a list of standard deviations
        sigma_hats = []
        for t in self.returns.index:
            Sigma_hat_t = self.Sigma_hats[t].values
            sigma_hat_t = np.sqrt(np.diag(Sigma_hat_t))
            sigma_hats.append(sigma_hat_t)

        if r_hats is not None:
            assert list(self.returns.index) == list(self.r_hats.index)
            assert list(self.returns.columns) == list(self.r_hats.columns)

        if rhos is not None:
            assert list(self.returns.index) == list(self.rhos.index)
            assert list(self.returns.columns) == list(self.rhos.columns)

        # Generate dataframe of standard deviations
        sigma_hats = np.array(sigma_hats)
        self.sigma_hats = pd.DataFrame(
            sigma_hats, index=self.returns.index, columns=self.returns.columns
        )
        assert list(self.returns.index) == list(self.sigma_hats.index)
        assert list(self.returns.columns) == list(self.sigma_hats.columns)

        # Generate Choleksy factors: self.L_inv_hats: A dictionary to store the inverse of the Cholesky factor (L_inv) of the covariance matrix for each time period. This is likely used for optimization or risk management calculations later in the class.
        self.L_inv_hats = {}
        for time in returns.index:
            self.L_inv_hats[time] = _get_L_inv(
                pd.DataFrame(
                    self.Sigma_hats[time].values, index=self.assets, columns=self.assets
                )
            )

    @property
    def assets(self):
        return self.returns.columns

    @property
    def n(self):
        return self.returns.shape[1]

    @property
    def T(self):
        return self.returns.shape[0]

    def solve_min_risk(self, prob, w, L_inv_param, L_inv, sigma_param=None, sigma=None):
        """
        Solves the minimum risk problem for a given whiteners.

        param prob: cvxpy problem object
        param w: cvxpy variable object
        param Lt_inv_param: cvxpy parameter parameter
        param Lt: whitener
        """
        L_inv_param.value = L_inv
        if sigma_param is not None and sigma is not None:
            sigma_param.value = sigma
        if self.C_speedup:
            prob.register_solve("cpg", cpg_solve)  # TODO: remove?
            prob.solve(method="cpg", updated_params=["L_inv_param"])
        else:
            prob.solve(verbose=False)

        return w.value, prob.objective.value

    def solve_risk_parity(self, prob, w, L_inv_param, L_inv):
        """
        Solves the risk parity problem for a given covariance matrix.

        param prob: cvxpy problem object
        param w: cvxpy variable object
        param Sigma_t_param: cvxpy parameter parameter
        param Sigma_t: covariance matrix
        """
        L_inv_param.value = L_inv
        prob.solve()
        w_normalized = w.value / np.sum(w.value)

        return w_normalized

    def solve_mean_variance(
        self, prob, w, L_inv_param, r_hat_param, L_inv, r_hat, rho_param, rho
    ):
        """
        Solves the mean variance problem.

        param prob: cvxpy problem object
        param w: cvxpy variable object
        param Sigma_t_param: cvxpy parameter parameter
        param Sigma_t: covariance matrix
        """

        if rho_param is not None:
            rho_param.value = rho.reshape(-1, 1)

        L_inv_param.value = L_inv
        r_hat_param.value = np.vstack([r_hat.reshape(-1, 1), 0])
        r_hat_param = np.zeros(r_hat_param.shape)
        prob.solve(ignore_dpp=True, solver="ECOS")

        return w.value, prob.objective.value

    def get_vol_cont_w(self, ws, obj, sigma_des):
        """
        Computes the weights of the volatility controlled portfolio.

        param ws: list of weights of the minimum risk portfolios
        param obj: list of objective values of the minimum risk portfolios
        param sigma_des: desired volatility of the portfolio
        """
        sigma_des = sigma_des * np.sqrt(self.adjust_factor)
        sigma_des = sigma_des / np.sqrt(252)
        # sigma_r = np.sqrt(obj)

        sigma_r = obj
        w_cash = (1 - sigma_des / sigma_r).reshape(-1, 1)
        ws = (1 - w_cash) * ws

        return np.hstack([ws, w_cash])

    def solve_max_diverse(self, prob, z, L_inv_param, sigma_param, L_inv, sigma):
        """
        Solves the maximum diversification optimization problem.

        param prob: cvxpy problem object
        param z: cvxpy variable object
        param L_inv_param: cvxpy parameter parameter
        param L_inv: inverse whitener
        param sigma_param: cvxpy parameter parameter
        param sigma: diagonal of the covariance matrix

        """
        L_inv_param.value = L_inv
        sigma_param.value = sigma
        prob.solve(solver="ECOS")

        w = z.value / np.sum(z.value)
        return w

    def solve_max_sharpe(self, prob, z, L_inv_param, r_hat_param, L_inv, r_hat):
        """
        Solves the maximum diversification optimization problem.

        param prob: cvxpy problem object
        param z: cvxpy variable object
        param L_inv_param: cvxpy parameter parameter
        param L_inv: inverse whitener
        param r_hat_param: cvxpy parameter parameter
        param r_hat: return

        """
        L_inv_param.value = L_inv.T
        r_hat_param.value = r_hat
        prob.solve(solver="ECOS")

        # call assertion error if fails
        assert prob.status == "optimal", r_hat

        # print(prob.status)

        w = z.value / np.sum(z.value)
        return w

    def dilute_with_cash(self, sigma_des=0.1):
        """
        Dilutes the portfolio with cash to achieve a desired volatility.
        """
        if self.diluted:
            print("Already trades cash...")
            return None
        ws_new = []
        print("changing weights to achieve desired volatility, using sigma hat obtained from the covariance formula of the paper")
        
        # The function iterates over each date (t) in the index of the returns. For each date,
        # it extracts the covariance matrix (Sigma_hat_t) and the current weights (w_t) of the portfolio.
        # It calculates the current portfolio's volatility (sigma_hat -> he square root of this variance is the standard deviation, also known as volatility) 
        # using the weights and the covariance matrix. Then theta is calculated: theta is a scaling factor to adjust the weights of the assets in the portfolio.
        # Finally, it computes the new weights (w_t_new) by multiplying the original weights by theta and adding the remaining weight to cash(1- theta).
        for i, t in enumerate(self.returns.index):
            Sigma_hat_t = self.Sigma_hats[t].values
            w_t = self.ws[i].reshape(-1, 1)
            sigma_hat = np.sqrt(w_t[: self.n].T @ Sigma_hat_t @ w_t[: self.n])

            '''
            Ratio des/hat this ratio compares the desired volatility to the current volatility. If the current volatility is higher than desired, this ratio will be less than 1, indicating that the portfolio's risk needs to be reduced. Conversely, if the current volatility is lower than desired, the ratio will be greater than 1, suggesting an increase in risk is acceptable.
            Square Root of Annualization Factor: Applying the square root to the annualization factor (1/252) converts the annual adjustment back to a daily scale. This is necessary because the portfolio adjustments are typically made on a daily basis, and the target volatility is an annualized figure.

            This resulting factor is then used to scale the current weights of the assets in the portfolio. The new weights will be a fraction (
            of the original weights, with the remaining portion of the portfolio allocated to a risk-free asset (like cash) to achieve the desired volatility level.
            '''
            theta = np.sqrt(1 / 252) * (sigma_des / sigma_hat)

            w_t_new = np.vstack([theta * w_t, 1 - theta])
            ws_new.append(w_t_new.flatten())

        self.ws_diluted = np.array(ws_new)
        self.diluted = True

    def get_risk_adj_returns(self, sigma_des=0.1):
        """
        Scales returns to adjusted risk level sigma_des.
        """
        a = sigma_des / (np.sqrt(252) * np.std(self.rets))

        self.rets_adjusted = a * self.rets

    def get_risk_adj_portfolio_growth(self, sigma_des=0.1):
        """
        Computes the risk adjusted portfolio growth from the daily (total)  returns.
        """
        self.get_risk_adj_returns(sigma_des=sigma_des)
        Vs = [np.array([1])]
        for t, r_t in enumerate(self.rets_adjusted):
            Vs.append(Vs[t] * (1 + r_t / self.adjust_factor))
        self.Vs_adjusted = np.array(Vs[1:])

    def backtest(
        self,
        portfolio_type="min_risk",
        cons=[],
        sigma_des=None,
        adjust_factor=1,
        additonal_cons={"short_lim": 1.6, "upper_bound": 0.15, "lower_bound": -0.1},
        C_speedup=False,
        kappa=None,
        rhos=None,
    ):
        """
        this function is used to calculate the correct/best weight for each asset in the portfolio.
        It creates a ws matrix that is the matrix of weights
        param portfolio_type: type of portfolio to backtest. Options are "min_risk", "vol_cont", "risk_parity", "mean_variance".
        param cons: list of constraints to impose on the optimization problem.
        param rhos: pandas DataFrame of uncertainty in expected returns
            if None, then uncertainty is set to zero. Causal estimates
        """
        self.portfolio_type = portfolio_type
        self.adjust_factor = adjust_factor
        self.additonal_cons = additonal_cons
        self.C_speedup = C_speedup

        # initialize ws and obj
        #ws = np.zeros((self.T, self.n)) # TODO IN CASE OF ERROR UNCOMMENT THIS
        #obj = np.zeros(self.T) # TODO IN CASE OF ERROR UNCOMMENT THIS

        # TODO: ugly to have this here
        if rhos is not None:
            self.rhos = rhos.loc[self.returns.index]
        else:
            self.rhos = rhos

        if portfolio_type == "eq_weighted":
            # the method creates a weight matrix ws where each asset in the portfolio has an equal weight. The weights are determined by dividing one by the number of assets (self.n).
            ws = np.ones((self.T, self.n)) / self.n

        if (portfolio_type == "min_risk" or portfolio_type == "vol_cont" or portfolio_type == "robust_min_risk"):
            # get the minimum risk portfolio
            w = cp.Variable((self.n, 1))
            L_inv_param = cp.Parameter((self.n, self.n))
            # risk = cp.norm(L_inv_param @ w, 2)

            if portfolio_type == "robust_min_risk":
                assert kappa != None
                sigma_hat_param = cp.Parameter((self.n, 1), nonneg=True)
                risk = cp.norm2(
                    cp.sum(cp.multiply(L_inv_param, w.T), axis=1)
                ) + kappa * cp.square(sigma_hat_param.T @ cp.abs(w))
            else:
                risk = cp.norm2(cp.sum(cp.multiply(L_inv_param, w.T), axis=1))

            if self.C_speedup and portfolio_type != "robust_min_risk":
                print(
                    "Using prespecified CVXPYgen formulation... Ignoring new constraints"
                )
                n = 49
                w = cp.Variable((n, 1), name="w")
                L_inv_param = cp.Parameter((n, n), name="L_inv_param")
                # risk = cp.norm2(cp.sum(cp.multiply(L_inv_param, w.T), axis=1))
                risk = cp.norm(L_inv_param @ w, 2)
                ones = np.ones((n, 1))
                cons = [ones.T @ w <= 1, ones.T @ w >= 1]
                additonal_cons = {
                    "short_lim": 1.6,
                    "upper_bound": 0.15,
                    "lower_bound": -0.1,
                }
                if [*additonal_cons.keys()]:
                    if "short_lim" in additonal_cons.keys():
                        cons += [cp.norm(w, 1) <= additonal_cons["short_lim"]]
                    if "upper_bound" in additonal_cons.keys():
                        cons += [w <= additonal_cons["upper_bound"]]
                    if "lower_bound" in additonal_cons.keys():
                        cons += [w >= additonal_cons["lower_bound"]]

                prob = cp.Problem(cp.Minimize(risk), cons)
                prob.register_solve("cpg", cpg_solve)
            else:
                # add constraints
                cons += [cp.sum(w) == 1]
                if [*additonal_cons.keys()]:
                    # print("Adding additional constraints")
                    if "short_lim" in additonal_cons.keys():
                        cons += [cp.norm(w, 1) <= additonal_cons["short_lim"]]
                    if "upper_bound" in additonal_cons.keys():
                        cons += [w <= additonal_cons["upper_bound"]]
                    if "lower_bound" in additonal_cons.keys():
                        cons += [w >= additonal_cons["lower_bound"]]
                    # cons += [w >= -0.15]
                    # cons += [w <= 0.3]
                    # cons += [w >= 0]

                prob = cp.Problem(cp.Minimize(risk), cons)

                # if self.portfolio_type == "robust_min_risk":
                # TODO: always define this? Not used for min_risk!
                all_sigma_hats = [
                    self.sigma_hats.values[t].reshape(-1, 1) for t in range(self.T)
                ]

                # Solve problem with random inputs once to speed up later solves
                L_inv_param.value = [*self.L_inv_hats.values()][0]
                if portfolio_type == "robust_min_risk":
                    sigma_hat_param.value = all_sigma_hats[0]
                prob.solve()

            # solve the problem for each date
            all_w = [w for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            if self.portfolio_type == "robust_min_risk":
                all_sigma_hat_param = [sigma_hat_param for _ in range(self.T)]
            else:
                all_sigma_hat_param = [None for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]

            pool = mp.Pool()
            ws_and_obj = pool.starmap(
                self.solve_min_risk,
                zip(
                    all_prob,
                    all_w,
                    all_L_inv_param,
                    [*self.L_inv_hats.values(), all_sigma_hat_param, all_sigma_hats],
                ),
            )
            pool.close()
            pool.join()

            # get the weights and objective values
            ws, obj = zip(*ws_and_obj)
            ws = np.array(ws)
            obj = np.array(obj)
            self.obj = np.array(obj)

            if portfolio_type == "vol_cont":
                assert sigma_des is not None
                self.diluted = True  # Trades cash
                ws = self.get_vol_cont_w(ws, obj, sigma_des)

        elif portfolio_type == "mean_variance":
            assert self.r_hats is not None
            assert sigma_des is not None
            sigma_des = sigma_des * self.adjust_factor

            w = cp.Variable((self.n + 1, 1))  # Last asset is cash
            L_inv_param = cp.Parameter((self.n, self.n))
            r_hat_param = cp.Parameter((self.n + 1, 1))  # Last asset is cash

            # risk = cp.norm(L_inv_param @ w[:-1], 2)
            risk = cp.norm2(cp.sum(cp.multiply(L_inv_param, w[:-1].T), axis=1))

            if self.rhos is not None:
                rho_param = cp.Parameter((self.n, 1), nonneg=True)
                ret = r_hat_param.T @ w - rho_param.T @ cp.abs(w[:-1])
            else:
                ret = r_hat_param.T @ w

            # add constraints
            cons += [cp.sum(w) == 1]
            cons += [risk <= sigma_des / np.sqrt(252)]

            if [*additonal_cons.keys()]:
                if "short_lim" in additonal_cons.keys():
                    cons += [cp.norm(w[:-1], 1) <= additonal_cons["short_lim"]]
                if "upper_bound" in additonal_cons.keys():
                    cons += [w[:-1] <= additonal_cons["upper_bound"]]
                if "lower_bound" in additonal_cons.keys():
                    cons += [w[:-1] >= additonal_cons["lower_bound"]]
                # cons += [w >= 0]

            prob = cp.Problem(cp.Maximize(ret), cons)

            # Solve problem with random inputs once to speed up later solves
            L_inv_param.value = [*self.L_inv_hats.values()][0]
            r_hat_param.value = np.vstack([self.r_hats.values[0].reshape(-1, 1), 0])

            if self.rhos is not None:
                rho_param.value = self.rhos.values[0].reshape(-1, 1)
            # prob.solve() TODO: uncomment if using dpp

            # solve the problem for each date
            all_w = [w for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            all_r_hat_param = [r_hat_param for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]

            if self.rhos is not None:
                all_rho_param = [rho_param for _ in range(self.T)]
                rhos = self.rhos.values
            else:
                all_rho_param = [None for _ in range(self.T)]
                rhos = [None for _ in range(self.T)]

            pool = mp.Pool()
            ws_and_obj = pool.starmap(
                self.solve_mean_variance,
                zip(
                    all_prob,
                    all_w,
                    all_L_inv_param,
                    all_r_hat_param,
                    [*self.L_inv_hats.values()],
                    self.r_hats.values,
                    all_rho_param,
                    rhos,
                ),
            )
            pool.close()
            pool.join()

            # get the weights and objective values
            ws, obj = zip(*ws_and_obj)
            ws = np.array(ws)
            obj = np.array(obj)
            self.obj = np.array(obj)

        elif portfolio_type == "max_diverse":
            L_inv_param = cp.Parameter((self.n, self.n))
            sigma_param = cp.Parameter((self.n, 1))

            z = cp.Variable((self.n, 1))  # Change of variables solution
            obj = cp.norm2(cp.sum(cp.multiply(L_inv_param, z.T), axis=1))

            M = additonal_cons["upper_bound"]
            cons = [z >= 0, z <= M * cp.sum(z), sigma_param.T @ z == 1]
            prob = cp.Problem(cp.Minimize(obj), cons)

            # Solve problem once to speed up later solves
            L_inv_param.value = [*self.L_inv_hats.values()][0]
            sigma_param.value = np.ones((self.n, 1))
            prob.solve(solver="ECOS")

            all_z = [z for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            all_sigma_param = [sigma_param for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]

            pool = mp.Pool()
            all_sigma_hats = [
                self.sigma_hats.values[t].reshape(-1, 1) for t in range(self.T)
            ]
            ws = pool.starmap(
                self.solve_max_diverse,
                zip(
                    all_prob,
                    all_z,
                    all_L_inv_param,
                    all_sigma_param,
                    [*self.L_inv_hats.values()],
                    all_sigma_hats,
                ),
            )
            pool.close()
            pool.join()

            ws = np.array(ws)

        elif portfolio_type == "risk_parity":
            w = cp.Variable((self.n, 1))
            L_inv_param = cp.Parameter((self.n, self.n))

            # risk = cp.norm(L_inv_param @ w, 2)
            risk = cp.norm2(cp.sum(cp.multiply(L_inv_param, w.T), axis=1))
            prob = cp.Problem(cp.Minimize(1 / 2 * risk - cp.log(cp.geo_mean(w))))

            # Solve once for speedup later
            # Solve problem with random inputs once to speed up later solves
            L_inv_param.value = [*self.L_inv_hats.values()][0]
            prob.solve()

            all_w = [w for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]

            pool = mp.Pool()
            ws = pool.starmap(
                self.solve_risk_parity,
                zip(all_prob, all_w, all_L_inv_param, [*self.L_inv_hats.values()]),
            )
            pool.close()
            pool.join()
            ws = np.array(ws)

        elif portfolio_type == "max_sharpe":
            assert self.r_hats is not None

            L_inv_param = cp.Parameter((self.n, self.n))
            r_hat_param = cp.Parameter((self.n, 1))

            z = cp.Variable((self.n, 1))  # Change of variables solution
            obj = cp.norm2(cp.sum(cp.multiply(L_inv_param, z.T), axis=1))

            M = additonal_cons["upper_bound"]
            # cons = [z >= 0, z <= M*cp.sum(z), r_hat_param.T@z == 1]
            cons = [z >= 0, r_hat_param.T @ z == 1]

            prob = cp.Problem(cp.Minimize(obj), cons)

            # Solve problem once to speed up later solves
            L_inv_param.value = [*self.Lt_inv_hats.values()][0].T
            r_hat_param.value = np.ones((self.n, 1))
            prob.solve(solver="ECOS")

            all_z = [z for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            all_r_hat_param = [r_hat_param for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]
            all_r_hats = [self.r_hats.values[t].reshape(-1, 1) for t in range(self.T)]

            pool = mp.Pool()

            ws = pool.starmap(
                self.solve_max_sharpe,
                zip(
                    all_prob,
                    all_z,
                    all_L_inv_param,
                    all_r_hat_param,
                    [*self.Lt_inv_hats.values()],
                    all_r_hats,
                ),
            )
            pool.close()
            pool.join()

            ws = np.array(ws)

        # weights of the portfolio saved in the trader class; ws is a matrix of weights where each row is the weights of the portfolio at a given time
        # weights are a crucial component as they determine how much of the portfolio's total capital is allocated to each asset
        self.ws = ws.reshape(self.T, -1)

    def get_total_returns(self, diluted_with_cash=False, sigma_des=None, rf=None):
        """
        Computes total daily returns of the portfolio from the weights and indivudal asset returns.
        """
        rets = []
        # print the predictor name, weights, and returns
        for t in range(self.T):
            # are the weights of the portfolio at time t; not the weights assigned from the predictor but from the portfolio
            # so for example in the equal weighted portfolio, the weights are always 1/25(because we have 25 assets in the stock portfolio)
            # and the weights are always 1/25 even if the predictor is for example EWMA;->
            # name:  EWMA
            # w_t:  [0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04
            # 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04]
            w_t = self.ws[t] # these are the weights obtained from the backtest function; and are the portfolio weights; not the predictor weights(so the covariance matrix formula of the paper is not used here
                
            # these are the returns obtained from the SP500_top25_adjusted.csv file
            r_t = self.returns.iloc[t].values.reshape(-1, 1)

            if (self.portfolio_type == "vol_cont" or self.portfolio_type == "mean_variance" or self.portfolio_type == "mean_variance2" or diluted_with_cash):
                if diluted_with_cash:
                    if not self.diluted == True:
                        assert sigma_des is not None
                        self.dilute_with_cash(sigma_des)
                    w_t = self.ws_diluted[t]
                        
                if rf is None:
                    r_t = np.vstack([r_t, 0])  # Add cash return (last)
                else:
                    rf_t = rf.iloc[t]
                    r_t = np.vstack([r_t, rf_t])
            rets.append(np.dot(w_t, r_t))
            temp = t
        
        # testing purpose only ########################################################
        #w_t = self.ws[temp]
        #r_t = self.returns.iloc[temp].values.reshape(-1, 1)
        #print("w_t: ", w_t) # TODO: remove this print
        #print("r_t: ", r_t) # TODO: remove this print
        #print("w_t shape: ", w_t.shape) # TODO: remove this print
        #print("r_t shape: ", r_t.shape) # TODO: remove this print
        # sum all the weights inside w_t and compare with 1
        sum_w_t = 0
        for i in range(len(w_t)):
            sum_w_t += w_t[i]
        
        print("sum_w_t: ", sum_w_t) # TODO: remove this print
        ################################################################################
        
        #print("rets content: ", rets) # TODO: remove this print
        #rets content:  [array([0.00566938]), array([-0.00129673]), array([0.00805442]), array([-4.19301648e-05]), array([0.
        #print("rets shape: ", np.array(rets).shape) # TODO: remove this print
        #rets shape:  (2770, 1)

        self.rets = np.array(rets) #This array represents the total daily returns of the portfolio over the entire period self.T.


    def get_portfolio_growth(self):
        """
        Computes the portfolio growth from the daily (total) returns.
        Note: run get_total_returns() first.
        """
        Vs = [np.array([1])]
        for t, r_t in enumerate(self.rets):
            Vs.append(Vs[t] * (1 + r_t / self.adjust_factor))

        self.Vs = np.array(Vs[1:]) # This array represents the cumulative growth of the portfolio over the entire period.

        #print("self.Vs content: ", self.Vs) # TODO: remove this print
        #print("self.Vs shape: ", self.Vs.shape) # TODO: remove this print

    def compute_max_drawdown(self):
        """
        Computes the maximum drawdown of the portfolio.
        """
        max_dd = 0
        peak = self.Vs[0]
        for V in self.Vs:
            if V > peak:
                peak = V
            dd = (peak - V) / peak
            if dd > max_dd:
                max_dd = dd
        return float(max_dd)

    def get_metrics(
        self, diluted_with_cash=False, sigma_des=0.1, rf=None, excess=False
    ):
        """
        param adjust_factor: factor to results by, \
            e.g., if we have a model for 100*r_daily, the volatility\
                should be adjusted by 1/sqrt(100)


        Computes the avg return, stdev, sharpe ratio, and max drawdown of the portfolio.
        Note: Run backtest first.
        """
        # diluted_with_cash = False # TODO: REMOVE ABSOLUTELY THIS PRINT -> if enabled i will have always the same performance results in every indicator
        # print("self.adjust_factor: ", self.adjust_factor) # TODO: remove this print -> it is 1
        self.get_total_returns(
            diluted_with_cash=diluted_with_cash, sigma_des=sigma_des, rf=rf
        )
        self.get_portfolio_growth()
        if rf is not None:
            rf = rf.values.reshape(-1, 1)
        if excess:
            assert rf is not None
            excess_returns = self.rets / self.adjust_factor - rf
            mean_return = np.mean(excess_returns) * 252
            stdev = np.std(self.rets) * np.sqrt(252)
            # mean_return = (np.mean(self.rets/self.adjust_factor) - np.mean(rf))* 252
            # stdev = np.std(self.rets / self.adjust_factor-rf) * np.sqrt(252)
        else:
            mean_return = np.mean(self.rets / self.adjust_factor) * 252
            stdev = np.std(self.rets / self.adjust_factor) * np.sqrt(252)
        sharpe_ratio = mean_return / stdev

        return metrics(mean_return, stdev, sharpe_ratio, self.compute_max_drawdown())

        # print(f"Mean annual return: {mean_return:.2%}")
        # print(f"Annual risk: {stdev:.2%}")
        # print(f"Sharpe ratio: {sharpe_ratio:.3}")
        # print(f"Maximum drawdown: {self.compute_max_drawdown():.2%}")
