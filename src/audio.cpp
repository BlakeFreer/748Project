#include "audio.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>

#include "sndfile.hh"

AudioFile::AudioFile(std::string filename) {
    namespace fs = std::filesystem;

    if (!fs::exists(filename)) {
        throw std::runtime_error("Audio file " + filename + " does not exist.");
    }

    SndfileHandle f(filename);
    if (f.error()) {
        throw std::runtime_error("Failed to open " + filename + ".");
    }

    size_t num_chn = f.channels();
    size_t num_frames = f.frames();

    if (num_chn > 1) {
        std::cout << filename
                  << " has more than 1 channel. Only the first "
                     "will be kept."
                  << std::endl;
    }

    sample_rate = f.samplerate();
    double* buffer = new double[num_chn * num_frames]();
    f.readf(buffer, num_frames);

    Eigen::ArrayXXd all_channels =
        Eigen::Map<Eigen::ArrayXXd>(buffer, num_chn, num_frames);

    data = all_channels.row(0);
}