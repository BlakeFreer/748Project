#include "extraction.hpp"

#include <fftw3.h>

#include <Eigen/Core>
#include <cmath>

#include "audio.hpp"
#include "window.hpp"

// Rows are bins, columns are frames
Eigen::ArrayXXcd STFT(Eigen::ArrayXd signal, int fftn, int hop) {
    // Pad audio to align with window and hop
    int num_frames = (signal.size() - fftn + hop - 1) / hop + 1;
    int padding = fftn + hop * (num_frames - 1) - signal.size();

    // std::cout << "num_frames: " << num_frames << std::endl;
    // std::cout << "signal.size(): " << signal.size() << std::endl;
    // std::cout << "fftn: " << fftn << std::endl;
    // std::cout << "hop: " << hop << std::endl;
    // std::cout << "padding: " << padding << std::endl;

    assert(0 <= padding);
    assert(padding < hop);

    if (padding > 0) {
        signal.conservativeResize(signal.size() + padding);
        for (auto& x : signal.tail(padding)) {
            x = 0;
        }
    }  // else no padding is needed (hops and window align perfectly)

    Eigen::ArrayXd window = BlackmanWindow(fftn);
    window /= window.sum();  // normalized window to unit mass

    int num_bins = fftn / 2 + 1;
    Eigen::ArrayXXcd stft(num_bins, num_frames);

    fftw_plan fft;
    for (int i = 0; i < num_frames; i++) {
        Eigen::ArrayXd sample = signal(Eigen::seqN(i * hop, fftn));
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

            if (f_low <= f && f <= f_center) {
                filters(j - 1, i) = (f - f_low) / (f_center - f_low);
            } else if (f_center <= f && f <= f_high) {
                filters(j - 1, i) = (f_high - f) / (f_high - f_center);
            } else {
                continue;  // these cells are already 0
            }
        }
    }
    return filters;
}

Eigen::ArrayXXd ExtractFeature(AudioFile aud) {
    /***************************************************************
        Normalize Amplitude
    ***************************************************************/
    double max_amplitude = aud.data.abs().maxCoeff();
    assert(max_amplitude > 0);
    Eigen::ArrayXd audio = aud.data / max_amplitude;

    /***************************************************************
        Power Spectrum
    ***************************************************************/
    // Window size and hop is recommended by Fine et al.
    const double kStepSec = 0.01;
    const double kWindowSec = 0.025;
    int hop = kStepSec * aud.sample_rate;
    int fftn = kWindowSec * aud.sample_rate;

    Eigen::ArrayXXd power_spectrum = STFT(audio, fftn, hop).abs2();
    // SaveCSV("power.csv", power_spectrum);

    /***************************************************************
     Mel Filterbank
     ***************************************************************/
    // values from Ganchev
    const int kNumFilters = 24;
    const double lowfreq = 0;
    const double highfreq = 4000;
    Eigen::ArrayXXd mel_filterbanks = CreateMelFilterbanks(
        kNumFilters, aud.sample_rate, fftn, lowfreq, highfreq);

    assert(mel_filterbanks.cols() == power_spectrum.rows());

    // Computes kNumFilters datapoints per frame.
    Eigen::ArrayXXd filtered_power(kNumFilters, power_spectrum.cols());

    for (int j = 0; j < kNumFilters; j++) {
        Eigen::ArrayXd filter = mel_filterbanks.row(j);
        for (int i = 0; i < power_spectrum.cols(); i++) {
            filtered_power(j, i) = (power_spectrum.col(i) * filter).sum();
        }
    }
    // SaveCSV("mel_power_binned.csv", filtered_power);

    /***************************************************************
        Pool to reduce time resolution
    ***************************************************************/
    // Adjust kNumFilters to change the frequency resolution

    // independent of duration so that all features have same dimensionality.
    const int kNumPeriods = 8;

    double breaks = static_cast<double>(filtered_power.cols()) / kNumPeriods;

    Eigen::ArrayXXd pooled_power(kNumFilters, kNumPeriods);
    for (int i = 0; i < kNumPeriods; i++) {
        int low_i = std::round(i * breaks);
        int high_i = std::round((i + 1) * breaks);
        auto seq = Eigen::seq(low_i, high_i - 1);

        for (int j = 0; j < kNumFilters; j++) {
            pooled_power(j, i) = filtered_power.row(j)(seq).mean();
            filtered_power.row(j)(seq) = NAN;
        }
    }

    // Check used all elements exactly once
    assert(filtered_power.isNaN().all());
    assert(!(pooled_power.isNaN()).any());

    /***************************************************************
        Take log10 of pooled_power power
    ***************************************************************/
    const double epsilon = 1e-8;  // to avoid log(0)
    Eigen::ArrayXXd pooled = (pooled_power + epsilon).log10();
    assert(!pooled.isNaN().any());

    return pooled;
}
