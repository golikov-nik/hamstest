try:
    from hamstest_adapter_ks import register as _reg_ks

    _reg_ks()
except ImportError:
    pass

import numpy as np

from hamstest.test import SimplePermutationTest


def _alternative_to_int(alternative: str | int) -> int:
    if isinstance(alternative, int):
        if alternative in (-1, 1):
            return alternative
        raise ValueError("KS alternative must be 'greater' or 'less'")

    alternative = alternative.lower()
    if alternative == "greater":
        return 1
    if alternative == "less":
        return -1
    raise ValueError("KS alternative must be 'greater' or 'less'")


def _ks_score(value: np.ndarray, group_size: int, alternative: int) -> int:
    total_size = len(value)
    prefix = 0
    max_prefix = 0
    min_prefix = 0

    for bit in value:
        prefix += int(bit) * total_size - group_size
        max_prefix = max(max_prefix, prefix)
        min_prefix = min(min_prefix, prefix)

    if alternative == 1:
        return int(max_prefix)
    return int(-min_prefix)


class KSTest(SimplePermutationTest):
    def __init__(self, n: int, m: int, *, alternative: str = "greater"):
        self.n = int(n)
        self.m = int(m)
        self.alternative = alternative
        self._alternative = _alternative_to_int(alternative)

    def shape(self) -> tuple[int, int]:
        return self.n, self.m

    def calculate_score(self, value: np.ndarray) -> int:
        return _ks_score(value, self.n, self._alternative)

    def native_kind(self) -> str:
        return "ks"

    def export_native_state(self, value: np.ndarray):
        return {"alt": self._alternative, "value": value}
