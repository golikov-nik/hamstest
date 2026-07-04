from hamstest_native.build import setup_adapter

setup_adapter(
    name="hamstest-adapter-mannwhitneyu",
    package="hamstest_adapter_mannwhitneyu",
    extension="_mannwhitneyu_adapter",
    description="Mann-Whitney U test C++ adapter for hamstest-native",
)
