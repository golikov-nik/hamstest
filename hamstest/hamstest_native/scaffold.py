from __future__ import annotations

import argparse
from pathlib import Path

ADAPTER_CPP = """#include <pybind11/pybind11.h>
#include <hamstest_native/adapter.hpp>

namespace py = pybind11;

struct Params {};

static void* build(PyObject* /*state*/, int /*n*/) {
    return new Params();
}

static void destroy(void* params) {
    delete static_cast<Params*>(params);
}

static long long score_from_value(const void* /*params*/, const int* /*value*/, int /*n*/) {
    return 0;
}

static hamstest_native::adapter_v1 ADAPTER {
    build,
    destroy,
    score_from_value,
    nullptr,
    nullptr,
    nullptr,
};

PYBIND11_MODULE(@EXTENSION@, m) {
    m.doc() = "@DESCRIPTION@";
    hamstest_native::bind_adapter_module(m, ADAPTER);
}
"""


def package_name_from_distribution(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def write_file(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_adapter_project(
    target: Path,
    *,
    distribution: str,
    package: str,
    extension: str,
    kind: str,
    force: bool,
) -> None:
    description = f"{kind} adapter for hamstest"

    write_file(
        target / "pyproject.toml",
        f"""[build-system]
requires = ["setuptools>=69", "wheel", "pybind11>=2.12.0", "hamstest>=0.1.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{distribution}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.10"
dependencies = ["hamstest>=0.1.0"]
""",
        force=force,
    )

    write_file(
        target / "setup.py",
        f"""from hamstest_native.build import setup_adapter


setup_adapter(
    name="{distribution}",
    package="{package}",
    extension="{extension}",
    description="{description}",
)
""",
        force=force,
    )

    write_file(
        target / package / "__init__.py",
        f"""from hamstest_native import make_register


register = make_register(__name__, ".{extension}", "{kind}")
""",
        force=force,
    )

    write_file(
        target / "src" / "adapter.cpp",
        ADAPTER_CPP.replace("@EXTENSION@", extension).replace("@DESCRIPTION@", description),
        force=force,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a hamstest C++ adapter package")
    parser.add_argument("target", help="Directory to create")
    parser.add_argument("--name", help="Distribution name, for example my-hamstest-adapter")
    parser.add_argument("--package", help="Import package name")
    parser.add_argument("--extension", default="_adapter", help="Extension module name")
    parser.add_argument("--kind", help="Default hamstest native kind")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args(argv)

    target = Path(args.target)
    distribution = args.name or target.name.replace("_", "-")
    package = args.package or package_name_from_distribution(distribution)
    kind = args.kind or package_name_from_distribution(
        distribution.removeprefix("hamstest-adapter-")
    )

    create_adapter_project(
        target,
        distribution=distribution,
        package=package,
        extension=args.extension,
        kind=kind,
        force=args.force,
    )


if __name__ == "__main__":
    main()
