import logging

import numpy as np

from hamstest.test import AbstractPermutationTestSubset, PermutationTestWithHashes

try:
    from hamstest_native import fastpath as _native_fastpath
except ImportError as exc:
    _native_fastpath = None
    _native_fastpath_import_error = exc
else:
    _native_fastpath_import_error = None

TestValue = tuple[int, int]

_logger = logging.getLogger(__name__)
_MISSING = object()
_NATIVE_STATUS_BY_KEY: dict[str, bool | None] = {}


def _native_kind(abstract_subset: AbstractPermutationTestSubset) -> str | None:
    native_kind = getattr(abstract_subset, "native_kind", None)
    if native_kind is None:
        return None

    kind = native_kind()
    return kind.strip().lower() if isinstance(kind, str) else None


def native_adapter_status(
    abstract_subset: AbstractPermutationTestSubset,
) -> tuple[str | None, bool | None, str]:
    kind = _native_kind(abstract_subset)
    if not kind:
        subset_name = type(abstract_subset).__name__
        return (
            None,
            False,
            f"hamstest native adapter inactive for {subset_name}: no native_kind hook",
        )

    if _native_fastpath is None:
        detail = "native extension unavailable"
        if _native_fastpath_import_error is not None:
            detail = f"{detail}: {_native_fastpath_import_error}"
        return kind, False, f"hamstest native adapter inactive for kind {kind!r}: {detail}"

    is_test_registered = getattr(_native_fastpath, "is_test_registered", None)
    if callable(is_test_registered):
        if is_test_registered(kind):
            return (
                kind,
                True,
                f"hamstest native adapter active for kind {kind!r}: native fast path is available",
            )
        return (
            kind,
            False,
            f"hamstest native adapter inactive for kind {kind!r}: no adapter registered",
        )

    return (
        kind,
        None,
        f"hamstest native adapter status for kind {kind!r} is unknown before perturbation",
    )


def _native_status_key(kind: str | None, abstract_subset: AbstractPermutationTestSubset) -> str:
    if kind:
        return f"kind:{kind}"
    subset_type = type(abstract_subset)
    return f"class:{subset_type.__module__}.{subset_type.__qualname__}"


def _log_native_status(
    kind: str | None,
    abstract_subset: AbstractPermutationTestSubset,
    active: bool | None,
    message: str,
) -> None:
    key = _native_status_key(kind, abstract_subset)
    previous = _NATIVE_STATUS_BY_KEY.get(key, _MISSING)
    if previous is active:
        return

    _NATIVE_STATUS_BY_KEY[key] = active
    _logger.debug(message)


def log_native_adapter_status(abstract_subset: AbstractPermutationTestSubset) -> None:
    kind, active, message = native_adapter_status(abstract_subset)
    _log_native_status(kind, abstract_subset, active, message)


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
            log_native_adapter_status(self.abstract_subset)
            return None

        kind = _native_kind(self.abstract_subset)
        if not kind:
            log_native_adapter_status(self.abstract_subset)
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
                _log_native_status(
                    kind,
                    self.abstract_subset,
                    False,
                    f"hamstest native adapter inactive for kind {kind!r}: no adapter registered",
                )
                return None
            raise

        _log_native_status(
            kind,
            self.abstract_subset,
            True,
            f"hamstest native adapter active for kind {kind!r}: using native fast path",
        )
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
