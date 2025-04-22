#include "extraction.hpp"

#include <fftw3.h>

#include <Eigen/Core>
#include <cmath>
#include <iostream>

#include "audio.hpp"
#include "csv.hpp"
#include "window.hpp"

// Rows are bins, columns are frames
Eigen::ArrayXXcd STFT(const AudioFile aud, int fftn, int hop) {
    // Pad audio to align with window and hop
    int num_frames = (aud.data.size() - fftn + hop) / hop;
    int padding = fftn + hop * (num_frames - 1) - aud.data.size();

    assert(0 <= padding < hop);

    Eigen::ArrayXd audio = aud.data;  // make a copy to modify
    if (padding > 0) {
        audio.conservativeResize(audio.size() + padding);
        for (auto& x : audio.tail(padding)) {
            x = 0;
        }
    }  // else no padding is needed (hops and window align perfectly)

    Eigen::ArrayXd window = BlackmanWindow(fftn);
    window /= window.sum();  // normalized window to unit mass

    int num_bins = fftn / 2 + 1;
    Eigen::ArrayXXcd stft(num_bins, num_frames);

    fftw_plan fft;
    for (int i = 0; i < num_frames; i++) {
        Eigen::ArrayXd sample = audio(Eigen::seqN(i * hop, fftn));
        Eigen::ArrayXd windowed_sample = window * sample;

        fft = fftw_plan_dft_r2c_1d(
            fftn, windowed_sample.data(),
            reinterpret_cast<fftw_complex*>(stft.col(i).data()), FFTW_ESTIMATE);
        fftw_execute(fft);
    }
    fftw_free(fft);

    return stft;
}

// Using HTLK MFCC-FB24 [Ganchev]
constexpr double hz2mel(double hz) {
    return 2595 * std::log10(1 + hz / 700.);
}

constexpr double mel2hz(double mel) {
    return 700 * (std::pow(10, mel / 2595.) - 1);
}

// Each row is a filter bank. Each column is an fft bin.
Eigen::ArrayXXd CreateMelFilterbanks(int num_filters, double sample_rate,
                                     int nfft, double lowfreq,
                                     double highfreq) {
    double mel_center_delta =
        (hz2mel(highfreq) - hz2mel(lowfreq)) / (num_filters + 1);

    int nbins = nfft / 2 + 1;

    Eigen::ArrayXd fft_freqs =
        Eigen::ArrayXd::LinSpaced(nbins, 0, sample_rate / 2);

    Eigen::ArrayXXd filters = Eigen::ArrayXXd::Zero(num_filters, nbins);

    for (int j = 1; j <= num_filters; j++) {
        // Compute vertices of filter
        double f_low = mel2hz(mel_center_delta * (j - 1));
        double f_center = mel2hz(mel_center_delta * (j));
        double f_high = mel2hz(mel_center_delta * (j + 1));

        // Create the triangle filter
        for (int i = 0; i < nbins; i++) {
            double f = fft_freqs[i];

            if (f_low <= f <= f_center) {
                filters(j - 1, i) = (f - f_low) / (f_center - f_low);
            } else if (f_center <= f <= f_high) {
                filters(j - 1, i) = (f - f_high) / (f_high - f_center);
            } else {
                continue;  // these cells are already 0
            }
        }
    }
    return filters;
}

Eigen::ArrayXd ExtractFeature(AudioFile aud) {
    // This window size and hop is recommended by Fine et al.
    const float kStepSec = 0.01;
    const float kWindowSec = 0.025;
    int hop = aud.sample_rate * kStepSec;
    int fftn = aud.sample_rate * kWindowSec;

    Eigen::ArrayXXcd stft = STFT(aud, fftn, hop);
    Eigen::ArrayXXd power_spectrum = stft.abs().pow(2);

    SaveCSV("power.csv", power_spectrum);

    assert(!power_spectrum.isNaN().any());

    // from Ganchev
    const int kNumFilters = 24;
    const double lowfreq = 0;
    const double highfreq = 4000;

    Eigen::ArrayXXd mel_filterbanks = CreateMelFilterbanks(
        kNumFilters, aud.sample_rate, fftn, lowfreq, highfreq);

    // Apply the filterbank to each frame of the STFT to get kNumFilters data
    // points per frame

    assert(mel_filterbanks.cols() == power_spectrum.rows());
    Eigen::ArrayXXd filtered_power(kNumFilters, power_spectrum.cols());

    for (int j = 0; j < kNumFilters; j++) {
        for (int i = 0; i < power_spectrum.cols(); i++) {
            Eigen::ArrayXd filter = mel_filterbanks.row(j);
            filtered_power(j, i) = (power_spectrum.col(i) * filter).sum();
        }
    }

    assert(!filtered_power.isNaN().any());
    filtered_power = filtered_power.log10();
    assert(!filtered_power.isNaN().any());

    // Pool the columns to reduce time resolution
    // (Adjust kNumFilters to change the frequency resolution)

    const int kNumPeriods = 8;  // independent of duration so that all features
                                // have same dimensionality.

    double breaks = static_cast<double>(filtered_power.cols()) / kNumPeriods;

    std::cout << "File is " << aud.data.size() / float(aud.sample_rate)
              << " sec long" << std::endl;
    std::cout << "There are " << stft.cols() << " frames" << std::endl;

    Eigen::ArrayXXd pooled(kNumFilters, kNumPeriods);
    for (int i = 0; i < kNumPeriods; i++) {
        int low_i = std::round(i * breaks);
        int high_i = std::round((i + 1) * breaks);
        std::cout << low_i << "\t" << high_i << std::endl;
        auto seq = Eigen::seq(low_i, high_i - 1);
        for (int j = 0; j < kNumFilters; j++) {
            pooled(j, i) = filtered_power.row(j)(seq).mean();
            filtered_power.row(j)(seq) = NAN;
        }
    }

    // Check used all elements exactly once
    assert(filtered_power.isNaN().all());
    // assert(!(pooled.isNaN()).any());

    pooled.resize(pooled.size(), 1);

    return pooled;
}
