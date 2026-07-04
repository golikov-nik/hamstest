try:
    from hamstest_adapter_mannwhitneyu import register as _reg_mwu

    _reg_mwu()
except ImportError:
    pass

import numpy as np

from hamstest.test import AbstractPermutationTest, AbstractPermutationTestSubset


class MannWhitneyUTestSubset(AbstractPermutationTestSubset):
    def __init__(self, test: "MannWhitneyUTest", sum: int):
        self.test = test
        self.sum = sum

    def get_score(self) -> int:
        return self.sum

    def exchange_bits(self, bit_zero: int, bit_one: int) -> None:
        self.sum += self.test.ranks[bit_zero]
        self.sum -= self.test.ranks[bit_one]

    def copy(self) -> "AbstractPermutationTestSubset":
        return MannWhitneyUTestSubset(self.test, self.sum)

    def native_kind(self) -> str:
        return "mannwhitneyu"

    def export_native_state(self):
        return {"sum": int(self.sum), "ranks": np.asarray(self.test.ranks, dtype=np.int64)}

    @classmethod
    def from_subset(cls, test: "MannWhitneyUTest", value: np.ndarray):
        return cls(test, np.dot(value, test.ranks))


class MannWhitneyUTest(AbstractPermutationTest):
    def __init__(self, n, m, ranks):
        self.n = n
        self.m = m
        self.ranks = ranks

    def shape(self) -> tuple[int, int]:
        return self.n, self.m

    def create_subset(self, value: np.ndarray) -> AbstractPermutationTestSubset:
        return MannWhitneyUTestSubset.from_subset(self, value)
