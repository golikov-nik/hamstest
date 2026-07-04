# hamstest

`hamstest` estimates arbitrarily small p-values in two-sample permutation tests using hash-augmented adaptive multilevel splitting Monte Carlo.

## Installation

```bash
pip install hamstest
```

Native C++ adapters for the bundled KS and Mann-Whitney U tests are optional:

```bash
pip install "hamstest[adapters]"
```

To check whether a native adapter is active while running the estimator, enable
debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

Plotting helpers used by notebooks are optional:

```bash
pip install "hamstest[plotting]"
```

## Quick start

```python
from hamstest import Estimator
from hamstest.permutation_tests.kstest import KSTest

test = KSTest(100, 900)
target_score = int(0.333 * 100 * 900)
estimator = Estimator(test, sample_size=101, seed=0)
log_pvalue, log_pvalue_stderr = estimator.estimate(target_score)
```

`test` should implement the permutation-test interfaces exported by
`hamstest`: `AbstractPermutationTest` and `AbstractPermutationTestSubset`.
The package also ships reference implementations in `hamstest.permutation_tests`.

## Adapter packages

The package includes the C++ dispatcher and a helper command for starting custom
native adapters:

```bash
hamstest-adapter-init my_hamstest_adapter --kind my_test
```

Example adapters for KS and Mann-Whitney U tests are developed in the same
GitHub repository and are published separately. They can be installed together
with `hamstest[adapters]` or as individual packages:

```bash
pip install hamstest-adapter-ks
pip install hamstest-adapter-mannwhitneyu
```

## Project

Source code and issue tracking are available at
https://github.com/golikov-nik/hamstest.
