# hamstest-adapter-ks

Native C++ adapter for `hamstest.permutation_tests.kstest.KSTest`.

## Installation

```bash
pip install hamstest-adapter-ks
```

The package depends on `hamstest>=0.1.0,<0.2.0`. When installed, importing
`hamstest.permutation_tests.kstest.KSTest` registers this adapter and lets
`hamstest` use the native fast path automatically.

## Usage

```python
from hamstest import Estimator
from hamstest.permutation_tests.kstest import KSTest

test = KSTest(100, 900)
estimator = Estimator(test, sample_size=101, seed=0)
log_pvalue, log_pvalue_stderr = estimator.estimate(30000)
```

If the adapter is not installed or cannot be loaded, `hamstest` falls back to
the pure Python implementation.

## Project

Source code and issue tracking are available at
https://github.com/golikov-nik/hamstest.
