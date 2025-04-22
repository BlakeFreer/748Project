#include "reduce.hpp"

#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "csv.hpp"

namespace fs = std::filesystem;

std::vector<fs::path> GetFeatureFiles(fs::path list_txt);

struct Args {
    std::vector<fs::path> feature_files;
    fs::path basis_file;
    fs::path mean_file;
    int dims;

    const std::string USAGE = "Usage: ./reduce <feats.txt> <basis-stem> <dims>";

    Args(int argc, char* argv[]) {
        if (argc != 4) {
            std::cerr << USAGE << std::endl;
            exit(2);
        }

        feature_files = GetFeatureFiles(argv[1]);
        basis_file = fs::path(argv[2]).replace_extension(".basis");
        mean_file = fs::path(argv[2]).replace_extension(".mean");
        dims = std::stoi(argv[3]);

        Validate();
    }

private:
    void Validate() {
        for (const auto& f : {basis_file, mean_file}) {
            if (!fs::exists(f)) {
                std::cerr << "Could not find file " << f << std::endl;
                exit(2);
            }
        }

        if (dims <= 0) {
            std::cerr << "Dimensions (" << dims << ") must be positive."
                      << std::endl;
            exit(2);
        }
    }
};

int main(int argc, char* argv[]) {
    Args args(argc, argv);

    Eigen::VectorXd mean = LoadCSV(args.mean_file);
    Eigen::MatrixXd basis = LoadCSV(args.basis_file);
    assert(mean.size() == basis.rows());
    assert(basis.rows() == basis.cols());

    for (const auto& f : args.feature_files) {
        Eigen::VectorXd feature = FlattenFeature(LoadCSV(f));

        assert(feature.size() == mean.size());

        Eigen::VectorXd reduced =
            basis.rightCols(args.dims).transpose() * (feature - mean);

        fs::path reduced_file = f;
        reduced_file.replace_extension(".reduced");
        SaveCSV(reduced_file, reduced);
        std::cout << reduced_file << std::endl;
    }
    return 0;
}

std::string StripQuotes(std::string s) {
    if (s.size() < 2) {
        return s;
    }

    if ((s.front() == '"' && s.back() == '"') ||
        (s.front() == '\'' && s.back() == '\'')) {
        return s.substr(1, s.size() - 2);
    }

    return s;
}

std::vector<fs::path> GetFeatureFiles(fs::path list_txt) {
    if (!fs::exists(list_txt)) {
        std::cerr << list_txt << " does not exist." << std::endl;
    }

    std::vector<fs::path> feature_files;

    std::string read_buffer;
    std::ifstream file(list_txt);
    int line_no = 1;
    while (std::getline(file, read_buffer)) {
        fs::path file(StripQuotes(read_buffer));

        if (!fs::exists(file)) {
            std::cerr << "Could not find " << file << " (line #" << line_no
                      << ")." << std::endl;
            exit(1);
        }

        feature_files.push_back(fs::path(file));

        line_no++;
    }
    return feature_files;
}
