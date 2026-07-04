from dataclasses import dataclass
from math import log

import pandas as pd
from joblib import Parallel, delayed

from hamstest.estimate import BoundarySelectionOptions, Estimator, ResamplingOptions
from hamstest.test import AbstractPermutationTest

_LN10 = log(10)


@dataclass
class Task:
    test: AbstractPermutationTest
    target: int
    sample_size: int
    move_scale: float
    resampling_options: ResamplingOptions
    boundary_selection_options: BoundarySelectionOptions


def run_once(args: tuple[int, Task, int]):
    task_index, task, seed = args
    est = Estimator(
        task.test,
        task.sample_size,
        seed,
        task.move_scale,
        task.resampling_options,
        task.boundary_selection_options,
    )
    return task_index, est.estimate(task.target)


def run_multiple(count: int, tasks: list[Task], *, n_jobs: int | None = None):
    """Run each task `count` times and return one DataFrame per task."""
    jobs: list[tuple[int, Task, int]] = [
        (task_index, task, seed) for seed in range(count) for task_index, task in enumerate(tasks)
    ]

    results_by_task: list[list] = [[] for _ in tasks]
    raw = Parallel(n_jobs=n_jobs)(delayed(run_once)(job) for job in jobs)

    for task_index, result in raw:
        results_by_task[task_index].append(result)

    result_dfs = []
    for cur_res in results_by_task:
        results_df = (
            pd.DataFrame(cur_res, columns=["log_prob", "log_err"])
            .sort_values("log_prob")
            .reset_index(drop=True)
        )
        results_df["lower"] = results_df["log_prob"] - 2 * results_df["log_err"]
        results_df["upper"] = results_df["log_prob"] + 2 * results_df["log_err"]
        result_dfs.append(results_df)

    return result_dfs


def make_success_rate_table_multi(
    scales: list[int],
    count: int,
    test: AbstractPermutationTest,
    target: int,
    exact_prob_log: float,
    sample_size: int,
    options,
    *,
    n_jobs: int | None = None,
):
    list_options = list(options.items())
    tasks = []
    for _name, (resampling_options, boundary_selection_options) in list_options:
        for move_scale in scales:
            tasks.append(
                Task(
                    test,
                    target,
                    sample_size,
                    move_scale,
                    resampling_options,
                    boundary_selection_options,
                )
            )

    results_dfs = run_multiple(count, tasks, n_jobs=n_jobs)
    all_dfs = []
    for i, (method, _) in enumerate(list_options):
        my_res = results_dfs[i * len(scales) : (i + 1) * len(scales)]
        res_scale = []
        for j in range(len(scales)):
            results_df = my_res[j]
            hits = (results_df["lower"] <= exact_prob_log) & (results_df["upper"] >= exact_prob_log)
            rate = hits.sum() / results_df.shape[0]
            res_scale.append((scales[j], rate))
        cur_df = (
            pd.DataFrame(res_scale, columns=["move_scale", "success_rate"])
            .sort_values("move_scale")
            .reset_index(drop=True)
        )
        cur_df["method"] = method
        all_dfs.append(cur_df)
    return pd.concat(all_dfs)


def make_boxplot_data(
    count: int,
    test: AbstractPermutationTest,
    target: int,
    exact_log_p: float,
    sample_size: int,
    move_scale: float,
    options,
    *,
    n_jobs: int | None = None,
):
    resampling_options, boundary_selection_options = options
    task = Task(
        test, target, sample_size, move_scale, resampling_options, boundary_selection_options
    )
    df = run_multiple(count, [task], n_jobs=n_jobs)[0]
    return -df["log_prob"].values / _LN10, -exact_log_p / _LN10


def make_confidence_interval_data(
    count: int,
    test: AbstractPermutationTest,
    target: int,
    exact_prob_log: float,
    sample_size: int,
    move_scale: float,
    options,
    *,
    n_jobs: int | None = None,
):
    resampling_options, boundary_selection_options = options
    task = Task(
        test, target, sample_size, move_scale, resampling_options, boundary_selection_options
    )
    results_df = run_multiple(count, [task], n_jobs=n_jobs)[0]
    lower = -results_df["lower"].values / _LN10
    upper = -results_df["upper"].values / _LN10
    exact_neg_log10 = -exact_prob_log / _LN10
    return results_df.index, lower, upper, exact_neg_log10
