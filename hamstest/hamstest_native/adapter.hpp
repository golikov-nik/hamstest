#pragma once

#include <Python.h>
#include <pybind11/pybind11.h>

namespace hamstest_native {

using build_params_fn = void* (*)(PyObject* py_state_dict, int n);
using destroy_params_fn = void (*)(void* params);
using score_from_value_fn = long long (*)(const void* params, const int* value, int n);
using apply_swap_fn = bool (*)(void* params, int a, int b);
using current_score_fn = long long (*)(const void* params);
using check_ge_bound_fn = bool (*)(const void* params, const int* value, int n, long long bound);

struct adapter_v1 {
    build_params_fn build;
    destroy_params_fn destroy;
    score_from_value_fn score_from_value;
    apply_swap_fn apply_swap;
    current_score_fn current_score;
    check_ge_bound_fn check_ge_bound;
};

inline constexpr const char* capsule_name = "hamstest_native.adapter_v1";

inline pybind11::capsule make_adapter_capsule(adapter_v1& adapter) {
    return pybind11::capsule(static_cast<void*>(&adapter), capsule_name);
}

inline void bind_adapter_module(pybind11::module_& module, adapter_v1& adapter) {
    module.def("make_adapter_capsule", [&adapter]() {
        return make_adapter_capsule(adapter);
    });
}

}  // namespace hamstest_native
