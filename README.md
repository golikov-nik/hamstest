# hamstest

`hamstest` estimates arbitrarily small p-values in two-sample permutation tests using hash-augmented adaptive multilevel splitting Monte Carlo.

## Installation

```bash
pip install hamstest
```

Install the optional plotting helpers used by the tutorial and paper notebooks with:

```bash
pip install "hamstest[plotting]"
```

Installing the example C++ adapters for KS and Mann-Whitney U tests:

```bash
pip install hamstest
pip install pybind11 wheel setuptools
pip install ./hamstest_adapter_ks --no-build-isolation
pip install ./hamstest_adapter_mannwhitneyu --no-build-isolation
```

The adapters are intentionally packaged like third-party extensions: they include headers from the installed `hamstest` package rather than from this source checkout. Once `hamstest` is published, plain `pip install ./hamstest_adapter_ks` will work because pip can install `hamstest` and `pybind11` into the isolated build environment. In a local checkout, use `--no-build-isolation` after installing `hamstest` and the build tools first.

When reinstalling an adapter after `hamstest` is already installed locally, use `--no-deps` too:

```bash
pip install pybind11 wheel setuptools
pip install ./hamstest_adapter_ks --no-build-isolation --no-deps --force-reinstall
pip install ./hamstest_adapter_mannwhitneyu --no-build-isolation --no-deps --force-reinstall
```

To start a new C++ adapter package:

```bash
hamstest-adapter-init my_hamstest_adapter --kind my_test
```

The generated package contains the `pyproject.toml`, `setup.py`, Python registration hook, and a starter `src/adapter.cpp`. You only need to replace the statistic-specific C++ functions.

## Development

```bash
pip install -e './hamstest[dev,plotting]'
ruff format --check .
ruff check .
pytest
```

## Quick start

```python
from hamstest import Estimator, ResamplingOptions, BoundarySelectionOptions
from hamstest.permutation_tests.kstest import KSTest

n, m = 100, 900
test = KSTest(n, m)
target = int(0.333 * n * m)   # score threshold corresponding to target p-value

est = Estimator(test, sample_size=101, move_scale=1.0)
log_p, log_err = est.estimate(target)

import math
print(f"log10(p) = {log_p / math.log(10):.2f} ± {log_err / math.log(10):.2f}")
```

## Implementing a custom test

Subclass two abstract classes:

```python
from hamstest import AbstractPermutationTest, AbstractPermutationTestSubset
import numpy as np

class MyTestSubset(AbstractPermutationTestSubset):
    def get_score(self) -> int:
        # Return the test statistic for the current permutation.
        ...

    def exchange_bits(self, bit_zero: int, bit_one: int) -> None:
        # Swap elements at bit_zero and bit_one between groups.
        # Possibly recalculate internal data, e.g. sum in Mann-Whitney U test
        ...

    def copy(self) -> 'MyTestSubset':
        ...

class MyTest(AbstractPermutationTest):
    def shape(self) -> tuple[int, int]:
        # Returns (n, m): sizes of two groups
        return self.n, self.m

    def create_subset(self, value: np.ndarray) -> MyTestSubset:
        # value is a 0/1 array of length n+m with exactly n ones.
        return MyTestSubset(...)
```

See [hamstest/hamstest/permutation_tests/kstest.py](hamstest/hamstest/permutation_tests/kstest.py) and [hamstest/hamstest/permutation_tests/mannwhitneyutest.py](hamstest/hamstest/permutation_tests/mannwhitneyutest.py) for complete implementations shipped with the package. The small repo-only scripts in [examples/ks_2samp.py](examples/ks_2samp.py) and [examples/mannwhitneyu.py](examples/mannwhitneyu.py) print side-by-side comparisons with SciPy.

## Notebooks

- [notebooks/tutorial.ipynb](notebooks/tutorial.ipynb) — step-by-step guide to implementing a test and running the estimator

The tutorial uses `hamstest.experiments` and `hamstest.plotting`, so install the `plotting` extra before running it.

## Repo layout

```
hamstest/                  Python package + C++ dispatcher (pip install ./hamstest)
hamstest/hamstest/permutation_tests/ Reference permutation-test implementations
hamstest/hamstest/experiments.py Experiment runners and result tables
hamstest/hamstest/plotting.py Plotting helpers
hamstest_adapter_ks/       KS test C++ adapter
hamstest_adapter_mannwhitneyu/ Mann-Whitney U test C++ adapter
examples/                  Repo-only comparison scripts
notebooks/                 Tutorial
```
