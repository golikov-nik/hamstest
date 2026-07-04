import numpy as np

from hamstest.test import AbstractPermutationTestSubset, PermutationTestWithHashes

try:
    from hamstest_native import fastpath as _native_fastpath
except ImportError:
    _native_fastpath = None

TestValue = tuple[int, int]


class PermutationTestSubset:
    def __init__(
        self,
        zero_pos,
        one_pos,
        hash: int,
        test: PermutationTestWithHashes,
        abstract_subset: AbstractPermutationTestSubset,
    ):
        self.zero_pos = zero_pos
        self.zero_pos.setflags(write=True)
        self.one_pos = one_pos
        self.one_pos.setflags(write=True)
        self.hash = hash
        self.test = test
        self.abstract_subset = abstract_subset
        self.statistics_value = abstract_subset.get_score()

    def get_sum(self) -> int:
        return self.statistics_value

    def get_value(self) -> TestValue:
        return self.statistics_value, self.hash

    def value_bigger(self, bound: TestValue) -> bool:
        return self.get_value() > bound

    def copy(self):
        return PermutationTestSubset(
            self.zero_pos.copy(),
            self.one_pos.copy(),
            self.hash,
            self.test,
            self.abstract_subset.copy(),
        )

    def _perturb_native_iters(
        self, iters: int, bits_zero: list[int], bits_one: list[int], bound: TestValue
    ) -> int | None:
        if _native_fastpath is None:
            return None

        kind = getattr(self.abstract_subset, "native_kind", lambda: None)()
        kind = kind.strip().lower() if isinstance(kind, str) else None
        if not kind:
            return None

        try:
            succ, new_hash = _native_fastpath.run_bitvector_iters(
                kind,
                self.zero_pos,
                self.one_pos,
                np.uint64(self.hash),
                self.test.hashes,
                int(bound[0]),
                np.uint64(bound[1]),
                int(iters),
                bits_zero,
                bits_one,
                self.abstract_subset.export_native_state(),
                0,
            )
        except RuntimeError as exc:
            if str(exc).startswith("No adapter registered for kind:"):
                return None
            raise

        self.hash = int(new_hash)
        N = int(len(self.zero_pos) + len(self.one_pos))
        new_value = np.zeros(N, dtype=np.int32)
        new_value[self.one_pos] = 1
        self.abstract_subset = self.test.test.create_subset(new_value)
        self.statistics_value = self.abstract_subset.get_score()
        return int(succ)

    def perturb_if_bigger_iters(
        self, iters: int, bits_zero: list[int], bits_one: list[int], bound: TestValue
    ) -> int:
        if iters != len(bits_zero) or iters != len(bits_one):
            raise ValueError("iters must match bits_zero and bits_one lengths")

        succ = self._perturb_native_iters(iters, bits_zero, bits_one, bound)
        if succ is not None:
            return succ

        successes = 0
        for t in range(iters):
            bit_zero_pos = bits_zero[t]
            bit_one_pos = bits_one[t]
            bit_zero = self.zero_pos[bit_zero_pos]
            if bit_one_pos == len(self.one_pos):
                successes += 1
                continue
            bit_one = self.one_pos[bit_one_pos]

            old_hash = self.hash
            old_score = self.statistics_value
            self.hash ^= int(self.test.hashes[bit_zero])
            self.hash ^= int(self.test.hashes[bit_one])
            self.abstract_subset.exchange_bits(bit_zero, bit_one)
            self.statistics_value = self.abstract_subset.get_score()

            if self.value_bigger(bound):
                self.zero_pos[bit_zero_pos], self.one_pos[bit_one_pos] = (
                    self.one_pos[bit_one_pos],
                    self.zero_pos[bit_zero_pos],
                )
                successes += 1
                continue

            self.abstract_subset.exchange_bits(bit_one, bit_zero)
            self.hash = old_hash
            self.statistics_value = old_score
        return int(successes)
