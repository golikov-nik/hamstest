"""
Accuracy tests for Estimator with the Mann-Whitney U test.

With ranks=list(range(n+m)), the maximum Mann-Whitney U score is n*m + n*(n-1)//2,
achieved by value=[0]*m + [1]*n (the n largest ranks go to group 1). Exactly
one permutation achieves this, so exact_p = 1 / C(n+m, n).
"""

import math

import pytest
from hamstest.estimate import Estimator

from tests.conftest import MannWhitneyUTest, exact_log_p, mannwhitneyu_max_target

SAMPLE_SIZE = 101
SEED = 0
SIGMA_MULTIPLIER = 3.0
FIXED_TOLERANCE = 1.0

MWU_CASES = [
    pytest.param(5, 5, id="small_n5m5"),  # exact log_p ≈ -5.529
    pytest.param(10, 10, id="medium_n10m10"),  # exact log_p ≈ -12.127
]


@pytest.mark.parametrize("n, m", MWU_CASES)
def test_mannwhitneyu_estimate_accuracy(n: int, m: int) -> None:
    """Estimator must recover the known exact log p-value within tolerance."""
    ranks = list(range(n + m))
    target = mannwhitneyu_max_target(n, m)
    log_p_exact = exact_log_p(n, m)

    log_p_est, log_err = Estimator(
        MannWhitneyUTest(n, m, ranks), sample_size=SAMPLE_SIZE, seed=SEED
    ).estimate(target)

    diff = abs(log_p_est - log_p_exact)
    if math.isnan(log_err):
        assert diff < FIXED_TOLERANCE, (
            f"Mann-Whitney U n={n} m={m}: |log_p_est - log_p_exact| = {diff:.4f} "
            f"exceeds fixed tolerance {FIXED_TOLERANCE}. "
            f"log_p_exact={log_p_exact:.4f}, log_p_est={log_p_est:.4f}"
        )
    else:
        tolerance = SIGMA_MULTIPLIER * log_err
        assert diff < tolerance, (
            f"Mann-Whitney U n={n} m={m}: |log_p_est - log_p_exact| = {diff:.4f} "
            f"exceeds {SIGMA_MULTIPLIER}*sigma = {tolerance:.4f}. "
            f"log_p_exact={log_p_exact:.4f}, log_p_est={log_p_est:.4f}"
        )


@pytest.mark.parametrize("n, m", MWU_CASES)
def test_mannwhitneyu_estimate_sign(n: int, m: int) -> None:
    """log_p_est must be negative (p-value is between 0 and 1)."""
    ranks = list(range(n + m))
    target = mannwhitneyu_max_target(n, m)
    log_p_est, _ = Estimator(
        MannWhitneyUTest(n, m, ranks), sample_size=SAMPLE_SIZE, seed=SEED
    ).estimate(target)
    assert log_p_est < 0.0, f"log_p_est={log_p_est} should be negative"


def test_mannwhitneyu_deterministic_with_seed() -> None:
    """Two Estimators with the same seed must return identical results."""
    n, m = 5, 5
    ranks = list(range(n + m))
    target = mannwhitneyu_max_target(n, m)
    r1 = Estimator(MannWhitneyUTest(n, m, ranks), sample_size=SAMPLE_SIZE, seed=42).estimate(target)
    r2 = Estimator(MannWhitneyUTest(n, m, ranks), sample_size=SAMPLE_SIZE, seed=42).estimate(target)
    assert r1[0] == r2[0], "log_p differs between runs with same seed"
    assert r1[1] == r2[1], "log_err differs between runs with same seed"


def test_mannwhitneyu_max_target_formula() -> None:
    """mannwhitneyu_max_target helper must equal sum of the top n ranks."""
    for n, m in [(5, 5), (10, 10), (3, 7)]:
        ranks = list(range(n + m))
        expected = sum(sorted(ranks, reverse=True)[:n])
        assert mannwhitneyu_max_target(n, m) == expected, (
            f"n={n} m={m}: formula gives {mannwhitneyu_max_target(n, m)}, expected {expected}"
        )
