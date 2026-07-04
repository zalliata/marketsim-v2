import math
from marketsim.agents.momentum_reversal import momentum_signal, reversal_signal, z_score


def test_momentum_sums_window_skipping_recent():
    # returns oldest..newest; lookback=3 skip=1 -> sum of indices [len-4, len-1)
    r = [0.1, 0.2, 0.3, 0.4, 0.5]  # skip last (0.5), sum 0.2+0.3+0.4
    assert math.isclose(momentum_signal(r, lookback=3, skip=1), 0.9)


def test_momentum_insufficient_history():
    assert momentum_signal([0.1, 0.2], lookback=12, skip=1) == 0.0


def test_reversal_negates_recent_return():
    assert math.isclose(reversal_signal([0.1, -0.3], lookback=1), 0.3)


def test_zscore_constant_is_zero():
    assert z_score(5.0, [5.0, 5.0, 5.0, 5.0]) == 0.0


def test_zscore_standardizes():
    hist = [0.0, 2.0, 4.0]           # mean 2, pstd sqrt(8/3)
    z = z_score(4.0, hist)
    assert math.isclose(z, (4.0 - 2.0) / math.sqrt(8 / 3), rel_tol=1e-9)
