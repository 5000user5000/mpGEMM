#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>

struct ErrorStats {
    double mse;
    double max_error;
};

// Computes MSE and max absolute error between two same-sized flat arrays.
inline ErrorStats measure_error(const std::vector<float>& ref,
                                const std::vector<float>& test) {
    size_t N = ref.size();
    double sum_sq = 0.0;
    double max_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double diff = double(test[i]) - double(ref[i]);
        sum_sq += diff * diff;
        max_err = std::max(max_err, std::abs(diff));
    }
    return { sum_sq / N, max_err };
}
