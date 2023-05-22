import pandas as pd
import pytest
from collections import namedtuple

from cvx.covariance.ewma import iterated_ewma as iterated_ewma_1
from cvx.covariance.ewma import regularize_covariance
from cvx.covariance.iterated_ewma_vec import iterated_ewma as iterated_ewma_2

IEWMA = namedtuple("iewma", ["mean", "covariance"])

@pytest.fixture
def returns(prices):
    return 100*prices.pct_change().fillna(0.0)

def _evaluate_nonzero_mean(iewma1, iewma2):
    # Test mean 
    mean1 = iewma1.mean
    mean2 = iewma2.mean
    assert mean1.keys() == mean2.keys()
    for time in mean1.keys():
        pd.testing.assert_series_equal(mean1[time], mean2[time], check_names=False)
    
    # Test covariance
    covariance1 = iewma1.covariance
    covariance2 = iewma2.covariance
    assert covariance1.keys() == covariance2.keys()
    for time in covariance1.keys():
        pd.testing.assert_frame_equal(covariance1[time], covariance2[time])

def test_cov_zero_mean(returns):
    iewma1 = list(iterated_ewma_1(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, mean=False, clip_at=4.2))
    iewma1 = {item.time: item.covariance for item in iewma1}

    iewma2 = iterated_ewma_2(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, clip_at=4.2)

    assert iewma1.keys() == iewma2.keys()
    for time in iewma1.keys():
        pd.testing.assert_frame_equal(iewma1[time], iewma2[time])

def test_cov_nonzero_mean(returns):

    iewma1 = list(iterated_ewma_1(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, clip_at=4.2, mean =True))
    means = {item.time: item.mean for item in iewma1}
    covariances = {item.time: item.covariance for item in iewma1}
    iewma1 = IEWMA(mean=means, covariance=covariances)

    iewma2 = iterated_ewma_2(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, clip_at=4.2, mean=True)

    _evaluate_nonzero_mean(iewma1, iewma2)

def test_regularization_zero_mean(returns):

    iewma1 = list(iterated_ewma_1(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, clip_at=4.2))
    Sigmas = {item.time: item.covariance for item in iewma1}
    iewma1 = dict(regularize_covariance(Sigmas, r=5))

    iewma2 = iterated_ewma_2(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, clip_at=4.2,
    low_rank=5)

    assert iewma1.keys() == iewma2.keys()
    for time in iewma1.keys():
        pd.testing.assert_frame_equal(iewma1[time], iewma2[time])

def test_regularization_nonzero_mean(returns):
    iewma1 = list(iterated_ewma_1(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, mean =True, clip_at=4.2))
    means = {item.time: item.mean for item in iewma1}
    covariances = {item.time: item.covariance for item in iewma1}
    covariances = dict(regularize_covariance(covariances, r=5))
    iewma1 = IEWMA(mean=means, covariance=covariances)

    iewma2 = iterated_ewma_2(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, mean=True,
    clip_at=4.2, low_rank=5)

    _evaluate_nonzero_mean(iewma1, iewma2)

