from pathlib import Path

from .registration import make_register as make_register

CAPSULE_NAME = "hamstest_native.adapter_v1"

try:
    from .fastpath import register_test as register_test
    from .fastpath import run_bitvector_iters as run_bitvector_iters
except ImportError:

    def run_bitvector_iters(*args, **kwargs):
        raise ImportError("hamstest native extension is not built or installed")

    def register_test(*args, **kwargs):
        raise ImportError("hamstest native extension is not built or installed")


def get_include() -> str:
    return str(Path(__file__).resolve().parent.parent)
