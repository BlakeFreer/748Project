#pragma once

#include <Eigen/Core>

#include "audio.hpp"

Eigen::ArrayXXcd STFT(const AudioFile aud, int fftn, int hop);

constexpr double hz2mel(double hz);
constexpr double mel2hz(double mel);

Eigen::ArrayXXd CreateMelFilterbanks(int num_filters, double sample_rate,
                                     int nfft, double lowfreq, double highfreq);

Eigen::ArrayXd ExtractFeature(AudioFile aud);
