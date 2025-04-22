#include <iostream>

#include "audio.hpp"
#include "extraction.hpp"
#include "fileio.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./extract <filename>" << std::endl;
        exit(2);
    }

    fs::path filename(argv[1]);
    AudioFile aud(filename.string());

    Eigen::ArrayXXd feature = ExtractFeature(aud);

    fs::path outfile = filename.replace_extension(".feat");
    SaveCSV(outfile, feature);

    std::cout << outfile << std::endl;

    return 0;
}
