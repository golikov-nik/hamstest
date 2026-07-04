"""
Accuracy tests for Estimator with KSTest.

For each (n, m) with target = n*m (the maximum KS score), exactly one
permutation achieves score >= target, so exact_p = 1 / C(n+m, n). We verify
that Estimator.estimate() returns a log_p within statistical tolerance.
"""

import math

import numpy as np
import pytest
from hamstest.estimate import Estimator
from hamstest.permutation_tests.kstest import KSTest
from hamstest.test import SimplePermutationTest

from tests.conftest import exact_log_p, ks_max_target

SAMPLE_SIZE = 101
SEED = 0
SIGMA_MULTIPLIER = 3.0
FIXED_TOLERANCE = 1.0  # natural log units; used when log_err is nan

KS_CASES = [
    pytest.param(5, 5, id="small_n5m5"),  # exact log_p ≈ -5.529
    pytest.param(10, 10, id="medium_n10m10"),  # exact log_p ≈ -12.127
    pytest.param(15, 15, id="larger_n15m15"),  # exact log_p ≈ -18.860
]


@pytest.mark.parametrize("n, m", KS_CASES)
def test_ks_estimate_accuracy(n: int, m: int) -> None:
    """Estimator must recover the known exact log p-value within tolerance."""
    target = ks_max_target(n, m)
    log_p_exact = exact_log_p(n, m)

    log_p_est, log_err = Estimator(KSTest(n, m), sample_size=SAMPLE_SIZE, seed=SEED).estimate(
        target
    )

    diff = abs(log_p_est - log_p_exact)
    if math.isnan(log_err):
        assert diff < FIXED_TOLERANCE, (
            f"KS n={n} m={m}: |log_p_est - log_p_exact| = {diff:.4f} "
            f"exceeds fixed tolerance {FIXED_TOLERANCE}. "
            f"log_p_exact={log_p_exact:.4f}, log_p_est={log_p_est:.4f}"
        )
    else:
        tolerance = SIGMA_MULTIPLIER * log_err
        assert diff < tolerance, (
            f"KS n={n} m={m}: |log_p_est - log_p_exact| = {diff:.4f} "
            f"exceeds {SIGMA_MULTIPLIER}*sigma = {tolerance:.4f}. "
            f"log_p_exact={log_p_exact:.4f}, log_p_est={log_p_est:.4f}"
        )


@pytest.mark.parametrize("n, m", KS_CASES)
def test_ks_estimate_sign(n: int, m: int) -> None:
    """log_p_est must be negative (p-value is between 0 and 1)."""
    target = ks_max_target(n, m)
    log_p_est, _ = Estimator(KSTest(n, m), sample_size=SAMPLE_SIZE, seed=SEED).estimate(target)
    assert log_p_est < 0.0, f"log_p_est={log_p_est} should be negative"


def test_ks_deterministic_with_seed() -> None:
    """Two Estimators with the same seed must return identical results."""
    n, m = 5, 5
    target = ks_max_target(n, m)
    r1 = Estimator(KSTest(n, m), sample_size=SAMPLE_SIZE, seed=42).estimate(target)
    r2 = Estimator(KSTest(n, m), sample_size=SAMPLE_SIZE, seed=42).estimate(target)
    assert r1[0] == r2[0], "log_p differs between runs with same seed"
    assert r1[1] == r2[1], "log_err differs between runs with same seed"


def test_ks_two_sided_not_supported() -> None:
    """KSTest exposes only one-sided alternatives."""
    with pytest.raises(ValueError, match="greater.*less"):
        KSTest(5, 5, alternative="two-sided")


def test_ks_test_uses_simple_permutation_test_with_native_hooks() -> None:
    test = KSTest(2, 2)
    subset = test.create_subset(np.array([1, 0, 1, 0], dtype=np.int32))

    assert isinstance(test, SimplePermutationTest)
    assert subset.native_kind() == "ks"
    assert subset.export_native_state()["alt"] == 1
