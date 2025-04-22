#include "audio.hpp"

#include <filesystem>
#include <format>
#include <iostream>

#include "sndfile.hh"

AudioFile::AudioFile(std::string filename) {
    namespace fs = std::filesystem;

    if (!fs::exists(filename)) {
        auto msg = std::format("Audio file {} does not exist.",
                               fs::absolute(filename).string());
        throw std::runtime_error(msg);
    }

    SndfileHandle f(filename);
    if (f.error()) {
        auto msg = std::format("Failed to open {}.", filename);
        throw std::runtime_error(msg);
    }

    size_t num_chn = f.channels();
    size_t num_frames = f.frames();

    if (num_chn > 1) {
        std::cout << std::format(
                         "{} has more than 1 channel. Only the first "
                         "will be kept.",
                         filename)
                  << std::endl;
    }

    sample_rate = f.samplerate();
    double* buffer = new double[num_chn * num_frames]();
    f.readf(buffer, num_frames);

    Eigen::ArrayXXd all_channels =
        Eigen::Map<Eigen::ArrayXXd>(buffer, num_chn, num_frames);

    data = all_channels.row(0);
}