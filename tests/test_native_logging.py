import logging

import hamstest.subset as subset_module
import numpy as np
from hamstest.estimate import Estimator
from hamstest.permutation_tests.kstest import KSTest
from hamstest.subset import PermutationTestSubset
from hamstest.test import (
    AbstractPermutationTest,
    AbstractPermutationTestSubset,
    PermutationTestWithHashes,
)


class _RegisteredFastpath:
    def is_test_registered(self, name: str) -> bool:
        return name == "ks"


class _MissingAdapterFastpath:
    def run_bitvector_iters(self, *args, **kwargs):
        raise RuntimeError("No adapter registered for kind: toy")


class _ToySubset(AbstractPermutationTestSubset):
    def __init__(self, value: np.ndarray):
        self.value = value.copy()

    def get_score(self) -> int:
        return int(self.value[0])

    def exchange_bits(self, bit_zero: int, bit_one: int) -> None:
        self.value[bit_zero] ^= 1
        self.value[bit_one] ^= 1

    def copy(self) -> "_ToySubset":
        return _ToySubset(self.value)

    def native_kind(self) -> str:
        return "toy"

    def export_native_state(self):
        return {}


class _ToyTest(AbstractPermutationTest):
    def shape(self) -> tuple[int, int]:
        return 1, 1

    def create_subset(self, value: np.ndarray) -> AbstractPermutationTestSubset:
        return _ToySubset(value)


def setup_function() -> None:
    subset_module._NATIVE_STATUS_BY_KEY.clear()


def test_estimator_logs_registered_native_adapter_status(caplog, monkeypatch) -> None:
    monkeypatch.setattr(subset_module, "_native_fastpath", _RegisteredFastpath())
    monkeypatch.setattr(subset_module, "_native_fastpath_import_error", None)

    with caplog.at_level(logging.DEBUG, logger="hamstest"):
        Estimator(KSTest(2, 2), sample_size=11, seed=0).estimate(0)

    assert "hamstest native adapter active for kind 'ks'" in caplog.text
    assert "native fast path is available" in caplog.text


def test_missing_runtime_adapter_logs_python_fallback(caplog, monkeypatch) -> None:
    monkeypatch.setattr(subset_module, "_native_fastpath", _MissingAdapterFastpath())
    monkeypatch.setattr(subset_module, "_native_fastpath_import_error", None)

    test = PermutationTestWithHashes(_ToyTest(), np.array([1, 2], dtype=np.uint64))
    subset = PermutationTestSubset(
        np.array([1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        1,
        test,
        _ToySubset(np.array([1, 0], dtype=np.int32)),
    )

    with caplog.at_level(logging.DEBUG, logger="hamstest"):
        subset.perturb_if_bigger_iters(
            1,
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            (-1, 0),
        )

    assert "hamstest native adapter inactive for kind 'toy'" in caplog.text
    assert "no adapter registered" in caplog.text
