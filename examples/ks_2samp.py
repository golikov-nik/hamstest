"""Compare hamstest and SciPy on a one-sided two-sample KS test."""

from __future__ import annotations

import math

from hamstest.estimate import Estimator
from hamstest.permutation_tests.kstest import KSTest
from scipy import stats


def estimate_with_hamstest(x, y, alternative: str, *, sample_size: int = 301, seed: int = 0):
    scipy_result = stats.ks_2samp(x, y, alternative=alternative, method="exact")
    target_score = round(scipy_result.statistic * len(x) * len(y))

    if target_score == 0:
        return 1.0, 0.0

    log_pvalue, log_error = Estimator(
        KSTest(len(x), len(y), alternative=alternative),
        sample_size=sample_size,
        seed=seed,
    ).estimate(int(target_score))

    return math.exp(log_pvalue), log_error


def main() -> None:
    x = [0, 1, 5, 6, 7, 8]
    y = [2, 3, 4, 9, 10, 11]

    print("alternative  scipy_stat  scipy_p   hamstest_p  log_error")
    for alternative in ("greater", "less"):
        scipy_result = stats.ks_2samp(x, y, alternative=alternative, method="exact")
        hamstest_p, log_error = estimate_with_hamstest(x, y, alternative, seed=2)

        print(
            f"{alternative:<11} "
            f"{scipy_result.statistic:10.4f} "
            f"{scipy_result.pvalue:8.4f} "
            f"{hamstest_p:11.4f} "
            f"{log_error:9.4f}"
        )


if __name__ == "__main__":
    main()
