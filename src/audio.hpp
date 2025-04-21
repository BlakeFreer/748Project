#pragma once

#include <Eigen/Core>

struct AudioFile {
    int sample_rate;
    Eigen::ArrayXXd data;

    AudioFile(std::string filename);
};