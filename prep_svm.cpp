#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "fileio.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./prep-svm <files.txt>" << std::endl;
        exit(2);
    }

    std::vector<fs::path> reduced_files = ReadFileListing(argv[1]);

    fs::path outfile(argv[1]);
    outfile.replace_extension(".svm");

    std::ofstream out(outfile);
    if (!out.is_open()) {
        std::cerr << "Failed to create " << outfile << std::endl;
        exit(1);
    }

    for (const auto& f : reduced_files) {
        if (f.extension() != ".reduced") {
            std::cerr << "Expected a .reduced file. Got " << f.extension()
                      << std::endl;
            exit(1);
        }

        // assume the label is the first character
        int label = f.stem().string()[0] - '0';

        Eigen::ArrayXd feature = LoadCSV(f);

        out << label << " ";
        for (int i = 0; i < feature.size(); i++) {
            out << i + 1 << ":" << feature(i) << " ";
        }
        out << std::endl;
    }

    out.close();

    return 0;
}