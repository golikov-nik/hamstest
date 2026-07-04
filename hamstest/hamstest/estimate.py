import dataclasses
from enum import Enum

import numpy as np

from hamstest.subset import PermutationTestSubset, TestValue
from hamstest.test import AbstractPermutationTest, PermutationTestWithHashes
from hamstest.utils import beta_mean_log, beta_var_log


@dataclasses.dataclass
class Level:
    bound: TestValue
    low_scores: list[TestValue]
    high_scores: list[TestValue]


class ResamplingOptions(Enum):
    FULL = 1
    PARTIAL = 2


class BoundarySelectionOptions(Enum):
    MEDIAN = 1
    MIN = 2


class Estimator:
    def __init__(
        self,
        test: AbstractPermutationTest,
        sample_size: int = 101,
        seed: int | None = None,
        move_scale: float = 1,
        resampling_options: ResamplingOptions = ResamplingOptions.FULL,
        boundary_selection_options: BoundarySelectionOptions = BoundarySelectionOptions.MEDIAN,
    ):
        self.sample_size = sample_size
        self.rnd = np.random.default_rng(seed)
        self.move_scale = move_scale
        self.test = PermutationTestWithHashes(
            test, np.asarray(self.rnd.integers(2**63, size=test.size), dtype=np.uint64)
        )
        self.resampling_options = resampling_options
        self.boundary_selection_options = boundary_selection_options

    def resample(
        self, vals: list[PermutationTestSubset]
    ) -> tuple[list[PermutationTestSubset] | None, int | None, TestValue | None, Level | None]:
        stats = []
        for i in range(self.sample_size):
            stats.append((vals[i].get_value(), i))
        stats.sort()

        start_from = 0
        if self.boundary_selection_options == BoundarySelectionOptions.MEDIAN:
            central_value = stats[self.sample_size // 2][0]
            for i in range(self.sample_size):
                if stats[i][0] >= central_value:
                    start_from = i
                    break
        elif self.boundary_selection_options != BoundarySelectionOptions.MIN:
            raise ValueError("unknown boundary selection option")

        if start_from == 0:
            while start_from < len(vals) and stats[start_from][0] == stats[0][0]:
                start_from += 1

        if start_from == len(vals):
            return None, None, None, None

        bound = stats[start_from - 1][0]

        def gen_new_sample():
            ind = self.rnd.integers(0, self.sample_size - start_from) + start_from
            return vals[stats[ind][1]].copy()

        new_samples = []
        low_scores = []
        high_scores = []
        for i in range(start_from):
            low_scores.append(stats[i][0])
            new_samples.append(gen_new_sample())
        for i in range(start_from, self.sample_size):
            high_scores.append(stats[i][0])
            new_samples.append(vals[stats[i][1]].copy())

        return new_samples, start_from, bound, Level(bound, low_scores, high_scores)

    def gen_zeros(self, cnt):
        return np.asarray(
            self.rnd.integers(low=0, high=self.test.test.shape()[1], size=cnt), dtype=np.int32
        )

    def gen_ones(self, cnt):
        return np.asarray(
            self.rnd.integers(low=0, high=self.test.test.shape()[0] + 1, size=cnt), dtype=np.int32
        )

    def perturbate(
        self,
        vals: list[PermutationTestSubset],
        cur_bound: TestValue,
        start_from: int,
        last_acceptance_rate: float,
    ) -> None:
        n_iterations = 0
        n_accepted = 0
        n_total = 0
        sample_bound = (
            self.sample_size if self.resampling_options == ResamplingOptions.FULL else start_from
        )
        need_accepted = max(1, (self.move_scale * sample_bound * self.test.test.shape()[0]) // 2)

        safe_rate = max(last_acceptance_rate, 1e-4)
        attempt_iters = max(
            1, int(round(self.test.test.shape()[0] * self.move_scale * 0.5 / safe_rate))
        )
        while n_accepted < need_accepted:
            for sample_id in range(sample_bound):
                n_accepted += vals[sample_id].perturb_if_bigger_iters(
                    attempt_iters,
                    self.gen_zeros(attempt_iters),
                    self.gen_ones(attempt_iters),
                    cur_bound,
                )
                n_total += attempt_iters

            n_iterations += 1

        for _ in range(n_iterations):
            for sample_id in range(sample_bound):
                n_accepted += vals[sample_id].perturb_if_bigger_iters(
                    attempt_iters,
                    self.gen_zeros(attempt_iters),
                    self.gen_ones(attempt_iters),
                    cur_bound,
                )
                n_total += attempt_iters

        return n_accepted / n_total

    def get_pval(self, levels: list[Level], target: int):
        # P(>= target)
        log_p = 0
        log_var = 0

        extended_target: TestValue = (target, 0)
        for level in levels:
            if extended_target <= level.bound:
                cnt_last = len(level.high_scores)
                for x in level.low_scores:
                    if x >= extended_target:
                        cnt_last += 1
                numerator = cnt_last
                log_p += beta_mean_log(numerator, self.sample_size)
                log_var += beta_var_log(numerator, self.sample_size)
                return log_p, np.sqrt(log_var)

            n_high = len(level.high_scores)
            n_high += 1
            log_p += beta_mean_log(n_high, self.sample_size)
            log_var += beta_var_log(n_high, self.sample_size)

        last_level = levels[-1]
        cnt_last = 0
        for x in last_level.high_scores:
            if x >= extended_target:
                cnt_last += 1

        if cnt_last == 0:
            log_p += beta_mean_log(1, len(last_level.high_scores))
            return log_p, float("nan")

        log_p += beta_mean_log(cnt_last, len(last_level.high_scores))
        log_var += beta_var_log(cnt_last, len(last_level.high_scores))
        return log_p, np.sqrt(log_var)

    def estimate(self, target: int, eps: float = 0) -> tuple[float, float | None]:
        def gen_rand_subset() -> PermutationTestSubset:
            value, hash = self.test.gen_rand_value(self.rnd)
            zero_pos = np.where(value == 0)[0].astype(np.int32)
            one_pos = np.where(value == 1)[0].astype(np.int32)
            return PermutationTestSubset(
                zero_pos, one_pos, hash, self.test, self.test.test.create_subset(value)
            )

        vals = [gen_rand_subset() for _ in range(self.sample_size)]

        levels = []
        vals, start_from, cur_bound, new_level = self.resample(vals)
        if vals is None or start_from is None or cur_bound is None or new_level is None:
            raise RuntimeError("could not build an initial level from the sampled permutations")
        levels.append(new_level)

        last_acceptance_rate = 1
        adjLogPval = 0
        while cur_bound[0] < target:
            adjLogPval += beta_mean_log(len(levels[-1].high_scores) + 1, self.sample_size)
            if eps != 0 and adjLogPval < np.log(eps):
                break

            last_acceptance_rate = self.perturbate(
                vals, cur_bound, start_from, last_acceptance_rate
            )

            new_vals, new_start_from, new_bound, new_level = self.resample(vals)
            if new_bound is None:
                break

            levels.append(new_level)
            vals = new_vals
            cur_bound = new_bound
            start_from = new_start_from

        return self.get_pval(levels, target)
