from importlib import import_module


def make_register(package: str, extension: str, default_name: str):
    def register(name: str = default_name):
        from .fastpath import register_test

        module = import_module(extension, package=package)
        register_test(name, module.make_adapter_capsule())

    return register
