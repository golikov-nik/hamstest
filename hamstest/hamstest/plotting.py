import matplotlib.pyplot as plt
import pandas as pd

from hamstest.experiments import make_boxplot_data, make_confidence_interval_data
from hamstest.test import AbstractPermutationTest


def style_boxplot(bp, color: str = "red", lw: float = 1.6) -> None:
    for box in bp["boxes"]:
        box.set(facecolor="none", edgecolor=color, linewidth=lw)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set(color=color, linewidth=lw)


_style_boxplot = style_boxplot


def plot_rate_df(df: pd.DataFrame, title: str):
    plot_df = (
        df.groupby(["move_scale", "method"], as_index=False)["success_rate"]
        .mean()
        .assign(success_pct=lambda d: d["success_rate"] * 100)
    )

    method_styles = {
        "full-median": dict(marker="o", linewidth=2, markersize=6),
        "partial-median": dict(marker="^", linewidth=2, markersize=6),
        "partial-min": dict(marker="s", linewidth=2, markersize=6),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, g in plot_df.groupby("method"):
        g = g.sort_values("move_scale")
        ax.plot(
            g["move_scale"].astype(str),
            g["success_pct"],
            label=method,
            **method_styles.get(method, {}),
        )

    ax.axhline(95, linestyle="--", linewidth=1, color="k")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Success percentage")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")
    ax.set_title(title)

    return ax


def plot_boxplot(estimates, exact_neg_log10: float, plot_title: str = ""):
    fig, ax = plt.subplots(figsize=(3, 5))
    bp = ax.boxplot([estimates], widths=0.4, patch_artist=True, showfliers=False)
    style_boxplot(bp, color="red", lw=2.0)

    ax.scatter(
        1,
        exact_neg_log10,
        marker="x",
        s=80,
        linewidths=2,
        color="tab:blue",
        zorder=5,
        label="exact",
    )
    ax.set_xticks([])
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel(r"$-\log_{10}(\mathrm{p\!-\!value})$")
    ax.set_title(plot_title)
    ax.legend(loc="lower right", framealpha=0.9)

    return ax


def make_boxplot(
    count: int,
    test: AbstractPermutationTest,
    target: int,
    exact_log_p: float,
    sample_size: int,
    move_scale: float,
    options,
    plot_title: str = "",
    *,
    n_jobs: int | None = None,
):
    estimates, exact_neg_log10 = make_boxplot_data(
        count, test, target, exact_log_p, sample_size, move_scale, options, n_jobs=n_jobs
    )
    return plot_boxplot(estimates, exact_neg_log10, plot_title)


def plot_confidence_intervals(x, lower, upper, exact_neg_log10: float, plot_title: str = ""):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in x:
        ax.plot([i, i], [upper[i], lower[i]], color="blue")

    ax.axhline(y=exact_neg_log10, color="red", linestyle="--", label="exact")
    ax.set_xlabel("Index")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{p\text{-}value})$")
    ax.set_title(f"Confidence Intervals\n{plot_title}")
    ax.legend()

    return ax


def make_confidence_interval_plot(
    count: int,
    test: AbstractPermutationTest,
    target: int,
    exact_prob_log: float,
    sample_size: int,
    move_scale: float,
    options,
    plot_title: str = "",
    *,
    n_jobs: int | None = None,
):
    x, lower, upper, exact_neg_log10 = make_confidence_interval_data(
        count, test, target, exact_prob_log, sample_size, move_scale, options, n_jobs=n_jobs
    )
    return plot_confidence_intervals(x, lower, upper, exact_neg_log10, plot_title)
