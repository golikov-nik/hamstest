import math

import scipy.special
from hamstest.permutation_tests.mannwhitneyutest import MannWhitneyUTest

__all__ = ["MannWhitneyUTest", "exact_log_p", "ks_max_target", "mannwhitneyu_max_target"]


def exact_log_p(n: int, m: int) -> float:
    """Natural log of 1/C(n+m, n).

    For both KS and Mann-Whitney U tests at the maximum possible score, exactly one
    out of C(n+m, n) permutations achieves that score, so this is the exact
    log p-value.
    """
    return -float(math.log(scipy.special.comb(n + m, n, exact=True)))


def ks_max_target(n: int, m: int) -> int:
    """Maximum KS score for group sizes n and m (equals n*m)."""
    return n * m


def mannwhitneyu_max_target(n: int, m: int) -> int:
    """Maximum Mann-Whitney U score for ranks=list(range(n+m))."""
    return n * m + n * (n - 1) // 2
