try:
    from . import _hamstest_native as _nat
except ImportError:
    import _hamstest_native as _nat  # pragma: no cover


def run_bitvector_iters(
    kind,
    zero_pos,
    one_pos,
    current_hash,
    hashes,
    bound_score,
    bound_hash,
    iters,
    bits_zero,
    bits_one,
    state,
    seed=0,
):
    return _nat.bitvector_iters(
        kind,
        zero_pos,
        one_pos,
        current_hash,
        hashes,
        bound_score,
        bound_hash,
        iters,
        bits_zero,
        bits_one,
        state,
        seed,
    )


def register_test(name: str, capsule):
    return _nat.register_test(name, capsule)


# Expose capsule name for adapters to use (optional nicety)
CAPSULE_NAME = "hamstest_native.adapter_v1"
