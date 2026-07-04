from setuptools import find_packages, setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pybind11 is required to build a hamstest adapter. "
        "Install it in this environment first when using --no-build-isolation: "
        "python -m pip install pybind11 wheel setuptools"
    ) from exc

from . import get_include


def setup_adapter(
    *,
    name: str,
    package: str,
    extension: str,
    sources=("src/adapter.cpp",),
    version: str = "0.1.0",
    description: str | None = None,
    install_requires=("hamstest>=0.1.0",),
    cxx_std: int = 17,
    **kwargs,
):
    ext_modules = [
        Pybind11Extension(
            f"{package}.{extension}",
            list(sources),
            cxx_std=cxx_std,
            include_dirs=[get_include()],
        )
    ]

    setup(
        name=name,
        version=version,
        description=description,
        python_requires=">=3.10",
        install_requires=list(install_requires),
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        **kwargs,
    )
