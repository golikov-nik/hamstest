#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <hamstest_native/adapter.hpp>

#include <vector>

namespace py = pybind11;

namespace {

struct MannWhitneyUParams {
    long long current_sum;
    std::vector<long long> ranks;
};

void* build_params(PyObject* state, int /*N*/) {
    auto dict = py::reinterpret_borrow<py::dict>(py::handle(state));
    auto ranks_array = dict["ranks"].cast<py::array_t<long long>>();
    const auto ranks_view = ranks_array.unchecked<1>();

    auto* params = new MannWhitneyUParams();
    params->current_sum = dict.contains("sum") ? dict["sum"].cast<long long>() : 0;
    params->ranks.resize(static_cast<size_t>(ranks_view.shape(0)));
    for (py::ssize_t i = 0; i < ranks_view.shape(0); ++i) {
        params->ranks[static_cast<size_t>(i)] = ranks_view(i);
    }
    return params;
}

void destroy_params(void* params) {
    delete static_cast<MannWhitneyUParams*>(params);
}

long long score_from_value(const void* params, const int* value, int N) {
    const auto* mwu = static_cast<const MannWhitneyUParams*>(params);
    long long score = 0;
    for (int i = 0; i < N; ++i) {
        if (value[i]) {
            score += mwu->ranks[static_cast<size_t>(i)];
        }
    }
    return score;
}

bool apply_swap(void* params, int zero_index, int one_index) {
    auto* mwu = static_cast<MannWhitneyUParams*>(params);
    mwu->current_sum += mwu->ranks[static_cast<size_t>(zero_index)]
        - mwu->ranks[static_cast<size_t>(one_index)];
    return true;
}

long long current_score(const void* params) {
    return static_cast<const MannWhitneyUParams*>(params)->current_sum;
}

hamstest_native::adapter_v1 adapter {
    build_params,
    destroy_params,
    score_from_value,
    apply_swap,
    current_score,
    nullptr,
};

}  // namespace

PYBIND11_MODULE(_mannwhitneyu_adapter, m) {
    m.doc() = "Mann-Whitney U adapter for hamstest-native";
    hamstest_native::bind_adapter_module(m, adapter);
}
