from hamstest_native.build import setup_adapter

setup_adapter(
    name="hamstest-adapter-ks",
    package="hamstest_adapter_ks",
    extension="_adapter",
    description="Native C++ adapter for hamstest Kolmogorov-Smirnov permutation tests",
    install_requires=("hamstest>=0.1.0,<0.2.0",),
)
