from hamstest_native.build import setup_adapter

setup_adapter(
    name="hamstest-adapter-mannwhitneyu",
    package="hamstest_adapter_mannwhitneyu",
    extension="_mannwhitneyu_adapter",
    description="Native C++ adapter for hamstest Mann-Whitney U permutation tests",
    install_requires=("hamstest>=0.1.0,<0.2.0",),
)
