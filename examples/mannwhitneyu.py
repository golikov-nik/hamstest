"""Compare hamstest and SciPy on one-sided Mann-Whitney U tests."""

from __future__ import annotations

import math

import numpy as np
from hamstest.estimate import Estimator
from hamstest.permutation_tests.mannwhitneyutest import MannWhitneyUTest
from scipy import stats


def estimate_with_hamstest(x, y, alternative: str, *, sample_size: int = 301, seed: int = 0):
    values = np.concatenate([x, y])
    ranks = (2 * stats.rankdata(values, method="average")).astype(np.int64)
    x_rank_sum = int(np.sum(ranks[: len(x)]))

    if alternative == "greater":
        test = MannWhitneyUTest(len(x), len(y), ranks)
        target_score = x_rank_sum
    elif alternative == "less":
        test = MannWhitneyUTest(len(y), len(x), ranks)
        target_score = int(np.sum(ranks) - x_rank_sum)
    else:
        raise ValueError("this example only uses 'greater' and 'less'")

    log_pvalue, log_error = Estimator(test, sample_size=sample_size, seed=seed).estimate(
        target_score
    )

    return math.exp(log_pvalue), log_error


def main() -> None:
    x = [5, 6, 7, 8]
    y = [1, 2, 3, 4]

    print("alternative  scipy_stat  scipy_p   hamstest_p  log_error")
    for alternative in ("greater", "less"):
        scipy_result = stats.mannwhitneyu(x, y, alternative=alternative, method="exact")
        hamstest_p, log_error = estimate_with_hamstest(x, y, alternative, seed=7)

        print(
            f"{alternative:<11} "
            f"{scipy_result.statistic:10.1f} "
            f"{scipy_result.pvalue:8.4f} "
            f"{hamstest_p:11.4f} "
            f"{log_error:9.4f}"
        )


if __name__ == "__main__":
    main()
