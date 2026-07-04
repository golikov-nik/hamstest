import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

extra_compile_args = ["/O2", "/DNDEBUG"] if sys.platform == "win32" else ["-O3", "-DNDEBUG"]

ext_modules = [
    Pybind11Extension(
        "hamstest_native._hamstest_native",
        ["src/native.cpp"],
        cxx_std=17,
        include_dirs=["."],
        extra_compile_args=extra_compile_args,
    )
]


setup(
    packages=find_packages(),
    package_data={"hamstest_native": ["adapter.hpp"]},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
