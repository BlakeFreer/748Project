#include "window.hpp"

#include <Eigen/Core>

Eigen::ArrayXd BlackmanWindow(int N) {
    constexpr double PI = 3.14159265358979323;
    // https://numpy.org/doc/stable/reference/routines.window.html
    return Eigen::ArrayXd::LinSpaced(N, 0, N - 1).unaryExpr([&](double n) {
        return 0.42 - 0.5 * std::cos(2. * PI * n / (N - 1)) +
               0.08 * std::cos(4. * PI * n / (N - 1));
    });
}