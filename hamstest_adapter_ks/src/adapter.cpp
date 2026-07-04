#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <hamstest_native/adapter.hpp>

#include <algorithm>

namespace py = pybind11;

namespace {

struct KSParams {
    int alternative;
    int N;
    int n;
};

int count_ones(py::array_t<int> value) {
    const auto view = value.unchecked<1>();
    int count = 0;
    for (ssize_t i = 0; i < view.shape(0); ++i) {
        count += view(i);
    }
    return count;
}

void* build_params(PyObject* state, int N) {
    auto dict = py::reinterpret_borrow<py::dict>(py::handle(state));
    const int alternative = dict.contains("alt") ? dict["alt"].cast<int>() : 1;
    if (alternative != 1 && alternative != -1) {
        throw py::value_error("KS alternative must be 'greater' or 'less'");
    }

    auto value = dict["value"].cast<py::array_t<int>>();
    return new KSParams{
        alternative,
        N,
        count_ones(value),
    };
}

void destroy_params(void* params) {
    delete static_cast<KSParams*>(params);
}

long long score_from_value(const void* params, const int* value, int N) {
    const auto* ks = static_cast<const KSParams*>(params);
    const long long scale = ks->N;
    const long long offset = ks->n;
    long long prefix_sum = 0;
    long long max_prefix = 0;
    long long min_prefix = 0;

    for (int i = 0; i < N; ++i) {
        prefix_sum += static_cast<long long>(value[i]) * scale - offset;
        max_prefix = std::max(max_prefix, prefix_sum);
        min_prefix = std::min(min_prefix, prefix_sum);
    }

    return ks->alternative == 1 ? max_prefix : -min_prefix;
}

bool score_reaches_bound(
    const void* params,
    const int* value,
    int N,
    long long bound
) {
    const auto* ks = static_cast<const KSParams*>(params);
    const long long scale = ks->N;
    const long long offset = ks->n;
    long long prefix_sum = 0;
    long long max_prefix = 0;
    long long min_prefix = 0;

    for (int i = 0; i < N; ++i) {
        prefix_sum += static_cast<long long>(value[i]) * scale - offset;
        if (prefix_sum > max_prefix) {
            max_prefix = prefix_sum;
            if (ks->alternative == 1 && max_prefix >= bound) {
                return true;
            }
        }
        if (prefix_sum < min_prefix) {
            min_prefix = prefix_sum;
            if (ks->alternative == -1 && -min_prefix >= bound) {
                return true;
            }
        }
    }

    return ks->alternative == 1 ? max_prefix >= bound : -min_prefix >= bound;
}

hamstest_native::adapter_v1 adapter {
    build_params,
    destroy_params,
    score_from_value,
    nullptr,
    nullptr,
    score_reaches_bound,
};

}  // namespace

PYBIND11_MODULE(_adapter, m) {
    m.doc() = "KS adapter for hamstest";
    hamstest_native::bind_adapter_module(m, adapter);
}
