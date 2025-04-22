#include <iostream>

#include "audio.hpp"
#include "csv.hpp"
#include "extraction.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./extract <filename>" << std::endl;
        exit(2);
    }

    fs::path filename(argv[1]);

    AudioFile aud(filename.string());

    Eigen::ArrayXd feature = ExtractFeature(aud);

    SaveCSV(filename.replace_extension(".feat"), feature);

    return 0;
}