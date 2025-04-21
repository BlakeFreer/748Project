#include "audio.hpp"

#include <filesystem>
#include <format>

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

    sample_rate = f.samplerate();
    double* buffer = new double[num_chn * num_frames]();
    f.readf(buffer, num_frames);

    data = Eigen::Map<Eigen::ArrayXXd>(buffer, num_chn, num_frames);
    data.transposeInPlace();
}