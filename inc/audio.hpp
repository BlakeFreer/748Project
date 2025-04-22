#pragma once

#include <Eigen/Core>

struct AudioFile {
    int sample_rate;
    Eigen::ArrayXd data;

    AudioFile(std::string filename);
};