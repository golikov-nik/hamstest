#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hamstest_native/adapter.hpp"

namespace py = pybind11;

namespace {

using Adapter = hamstest_native::adapter_v1;

std::unordered_map<std::string, Adapter> g_adapters;

Adapter find_adapter(const std::string& kind) {
    const auto adapter = g_adapters.find(kind);
    if (adapter == g_adapters.end()) {
        throw std::runtime_error("No adapter registered for kind: " + kind);
    }
    return adapter->second;
}

template <typename T, int Flags>
std::vector<T> copy_array(py::array_t<T, Flags> array) {
    const auto view = array.template unchecked<1>();
    std::vector<T> values(static_cast<size_t>(view.shape(0)));
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        values[static_cast<size_t>(i)] = view(i);
    }
    return values;
}

template <typename T, int Flags>
void write_array(py::array_t<T, Flags> array, const std::vector<T>& values) {
    auto view = array.template mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        view(i) = values[static_cast<size_t>(i)];
    }
}

std::vector<int> value_from_positions(const std::vector<int>& one_positions, int N) {
    std::vector<int> value(static_cast<size_t>(N), 0);
    for (int index : one_positions) {
        value[static_cast<size_t>(index)] = 1;
    }
    return value;
}

class AdapterParams {
public:
    AdapterParams(const Adapter& adapter, py::dict state, int N)
        : adapter_(adapter),
          params_(adapter_.build ? adapter_.build(state.ptr(), N) : nullptr) {}

    AdapterParams(const AdapterParams&) = delete;
    AdapterParams& operator=(const AdapterParams&) = delete;

    ~AdapterParams() {
        if (adapter_.destroy && params_) {
            adapter_.destroy(params_);
        }
    }

    void* get() const {
        return params_;
    }

private:
    const Adapter& adapter_;
    void* params_;
};

long long score_from_value(
    const Adapter& adapter,
    const void* params,
    const std::vector<int>& value
) {
    if (!adapter.score_from_value) {
        throw std::runtime_error("adapter missing score_from_value");
    }
    return adapter.score_from_value(params, value.data(), static_cast<int>(value.size()));
}

bool score_hash_exceeds_bound(
    const Adapter& adapter,
    void* params,
    const std::vector<int>& value,
    long long bound_score,
    uint64_t bound_hash,
    uint64_t current_hash
) {
    if (adapter.current_score || !adapter.check_ge_bound) {
        const long long score = adapter.current_score
            ? adapter.current_score(params)
            : score_from_value(adapter, params, value);
        return score > bound_score || (score == bound_score && current_hash > bound_hash);
    }

    const long long score_to_reach = current_hash > bound_hash
        ? bound_score
        : bound_score + 1;
    return adapter.check_ge_bound(
        params,
        value.data(),
        static_cast<int>(value.size()),
        score_to_reach
    );
}

}  // namespace

py::tuple bitvector_iters(
    const std::string& kind,
    py::array_t<int, py::array::c_style | py::array::forcecast> zero_pos_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> one_pos_arr,
    uint64_t current_hash,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> hashes_arr,
    long long bound_score,
    uint64_t bound_hash,
    int iters,
    py::array_t<int, py::array::c_style | py::array::forcecast> bits_zero_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> bits_one_arr,
    py::object state_obj,
    uint64_t /*seed unused*/
) {
    const Adapter adapter = find_adapter(kind);

    std::vector<int> zero_positions = copy_array(zero_pos_arr);
    std::vector<int> one_positions = copy_array(one_pos_arr);
    const std::vector<int> zero_choices = copy_array(bits_zero_arr);
    const std::vector<int> one_choices = copy_array(bits_one_arr);
    const std::vector<uint64_t> hashes = copy_array(hashes_arr);

    auto state = state_obj.cast<py::dict>();
    const int N = static_cast<int>(zero_positions.size() + one_positions.size());
    std::vector<int> value = value_from_positions(one_positions, N);
    AdapterParams params(adapter, state, N);

    uint64_t hash = current_hash;
    int successes = 0;
    const int requested_iters = std::max(iters, 0);
    const auto iteration_count = std::min<size_t>(
        static_cast<size_t>(requested_iters),
        std::min(zero_choices.size(), one_choices.size())
    );

    {
        // Keep Python-facing writes below outside the GIL release scope.
        py::gil_scoped_release release;

        for (size_t step = 0; step < iteration_count; ++step) {
            const int zero_choice = zero_choices[step];
            const int one_choice = one_choices[step];

            const int zero_index = zero_positions[static_cast<size_t>(zero_choice)];
            if (one_choice == static_cast<int>(one_positions.size())) {
                ++successes;
                continue;
            }

            const int one_index = one_positions[static_cast<size_t>(one_choice)];
            const int old_zero_value = value[static_cast<size_t>(zero_index)];
            const int old_one_value = value[static_cast<size_t>(one_index)];

            value[static_cast<size_t>(zero_index)] = 1;
            value[static_cast<size_t>(one_index)] = 0;
            const bool adapter_updated = adapter.apply_swap
                ? adapter.apply_swap(params.get(), zero_index, one_index)
                : false;

            hash ^= hashes[static_cast<size_t>(zero_index)];
            hash ^= hashes[static_cast<size_t>(one_index)];

            if (score_hash_exceeds_bound(
                    adapter,
                    params.get(),
                    value,
                    bound_score,
                    bound_hash,
                    hash
                )) {
                std::swap(
                    zero_positions[static_cast<size_t>(zero_choice)],
                    one_positions[static_cast<size_t>(one_choice)]
                );
                ++successes;
            } else {
                if (adapter_updated) {
                    adapter.apply_swap(params.get(), one_index, zero_index);
                }
                value[static_cast<size_t>(zero_index)] = old_zero_value;
                value[static_cast<size_t>(one_index)] = old_one_value;
                hash ^= hashes[static_cast<size_t>(one_index)];
                hash ^= hashes[static_cast<size_t>(zero_index)];
            }
        }
    }

    write_array(zero_pos_arr, zero_positions);
    write_array(one_pos_arr, one_positions);

    return py::make_tuple(successes, py::int_(hash));
}

void register_test(const std::string& name, py::capsule cap) {
    if (std::strcmp(cap.name(), hamstest_native::capsule_name) != 0) {
        throw std::runtime_error("Invalid adapter capsule name");
    }
    auto* ptr = static_cast<hamstest_native::adapter_v1*>(cap.get_pointer());
    if (!ptr || !ptr->score_from_value) {
        throw std::runtime_error("Adapter is null or missing score_from_value");
    }
    g_adapters[name] = *ptr;
}

PYBIND11_MODULE(_hamstest_native, m) {
    m.doc() = "Permutation test native core with dynamic adapter registry";
    m.def(
        "bitvector_iters",
        &bitvector_iters,
        py::arg("kind"),
        py::arg("zero_pos"),
        py::arg("one_pos"),
        py::arg("current_hash"),
        py::arg("hashes"),
        py::arg("bound_score"),
        py::arg("bound_hash"),
        py::arg("iters"),
        py::arg("bits_zero"),
        py::arg("bits_one"),
        py::arg("state"),
        py::arg("seed")
    );
    m.def("register_test", &register_test, py::arg("name"), py::arg("capsule"));
}
