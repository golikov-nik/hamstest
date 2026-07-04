import math

from scipy import stats

from examples.ks_2samp import estimate_with_hamstest as estimate_ks_with_hamstest
from examples.mannwhitneyu import estimate_with_hamstest as estimate_mwu_with_hamstest


def test_ks_example_is_close_to_scipy() -> None:
    x = [-1.0, -0.2, 0.1, 0.4, 1.2]
    y = [-0.5, 0.3, 0.9, 1.1, 1.6]

    hamstest_p, log_error = estimate_ks_with_hamstest(
        x,
        y,
        "less",
        sample_size=101,
        seed=0,
    )
    scipy_p = stats.ks_2samp(x, y, alternative="less", method="exact").pvalue

    assert abs(math.log(hamstest_p) - math.log(scipy_p)) <= max(3.0 * log_error, 0.1)


def test_mannwhitneyu_example_is_close_to_scipy() -> None:
    x = [5, 6, 7, 8]
    y = [1, 2, 3, 4]

    hamstest_p, log_error = estimate_mwu_with_hamstest(
        x,
        y,
        "greater",
        sample_size=101,
        seed=3,
    )
    scipy_p = stats.mannwhitneyu(x, y, alternative="greater", method="exact").pvalue

    assert abs(math.log(hamstest_p) - math.log(scipy_p)) <= max(3.0 * log_error, 0.1)
