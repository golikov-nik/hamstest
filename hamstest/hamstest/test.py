from abc import ABC, abstractmethod

import numpy as np


class AbstractPermutationTestSubset(ABC):
    @abstractmethod
    def get_score(self) -> int:
        pass

    @abstractmethod
    def exchange_bits(self, bit_zero: int, bit_one: int) -> None:
        pass

    @abstractmethod
    def copy(self) -> "AbstractPermutationTestSubset":
        pass


class AbstractPermutationTest(ABC):
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        pass

    @property
    def size(self) -> int:
        return self.shape()[0] + self.shape()[1]

    @abstractmethod
    def create_subset(self, value: np.ndarray) -> AbstractPermutationTestSubset:
        pass


class SimplePermutationTestSubset(AbstractPermutationTestSubset):
    def __init__(self, test: "SimplePermutationTest", value: np.ndarray):
        self.test = test
        self.value = value

    def get_score(self) -> int:
        return self.test.calculate_score(self.value)

    def exchange_bits(self, bit_zero: int, bit_one: int) -> None:
        self.value[bit_zero] ^= 1
        self.value[bit_one] ^= 1

    def copy(self) -> "AbstractPermutationTestSubset":
        return SimplePermutationTestSubset(self.test, self.value.copy())

    def native_kind(self) -> str | None:
        native_kind = getattr(self.test, "native_kind", None)
        if native_kind is None:
            return None
        return native_kind()

    def export_native_state(self) -> object | None:
        export_native_state = getattr(self.test, "export_native_state", None)
        if export_native_state is None:
            return None
        return export_native_state(self.value)


class SimplePermutationTest(AbstractPermutationTest):
    @abstractmethod
    def calculate_score(self, value: np.ndarray) -> int:
        pass

    def create_subset(self, value: np.ndarray) -> AbstractPermutationTestSubset:
        return SimplePermutationTestSubset(self, value)


class PermutationTestWithHashes:
    def __init__(self, test: AbstractPermutationTest, hashes: np.ndarray):
        if len(hashes) != test.size:
            raise ValueError("hash count must match permutation test size")
        self.test = test
        self.hashes = hashes

    def gen_rand_value(self, rnd: np.random.Generator) -> tuple[np.ndarray, int]:
        a = np.zeros(self.test.size, dtype=np.int32)
        a[: self.test.shape()[0]] = 1
        rnd.shuffle(a)
        result_hash = int(np.bitwise_xor.reduce(self.hashes[a.astype(bool)]))
        return a, result_hash
