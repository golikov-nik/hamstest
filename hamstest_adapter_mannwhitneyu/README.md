# hamstest-adapter-mannwhitneyu

Native C++ adapter for `hamstest.permutation_tests.mannwhitneyutest.MannWhitneyUTest`.

## Installation

```bash
pip install hamstest-adapter-mannwhitneyu
```

The package depends on `hamstest>=0.1.0,<0.2.0`. When installed, importing
`hamstest.permutation_tests.mannwhitneyutest.MannWhitneyUTest` registers this
adapter and lets `hamstest` use the native fast path automatically.

## Usage

```python
from hamstest import Estimator
from hamstest.permutation_tests.mannwhitneyutest import MannWhitneyUTest

n, m = 20, 30
ranks = list(range(n + m))
test = MannWhitneyUTest(n, m, ranks)
estimator = Estimator(test, sample_size=101, seed=0)
log_pvalue, log_pvalue_stderr = estimator.estimate(735)
```

If the adapter is not installed or cannot be loaded, `hamstest` falls back to
the pure Python implementation.

## Project

Source code and issue tracking are available at
https://github.com/golikov-nik/hamstest.
