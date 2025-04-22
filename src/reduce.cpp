#include "reduce.hpp"

#include <Eigen/Core>

Eigen::ArrayXd FlattenFeature(Eigen::ArrayXXd feature) {
    feature.resize(feature.size(), 1);
    return feature;
}