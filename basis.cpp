#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "csv.hpp"
#include "reduce.hpp"

namespace fs = std::filesystem;

std::vector<fs::path> GetFeatureFiles(fs::path list_txt);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./basis <feats.txt>" << std::endl;
        exit(2);
    }

    fs::path infile(argv[1]);
    std::vector<fs::path> feature_files = GetFeatureFiles(infile);

    /***************************************************************
        Load features
    ***************************************************************/

    // Could reduce memory overhead by reading files twice - once to calculate
    // mean and again to compute covar by stacking XX^T. Accepting the memory
    // to reduce time spent reading files.

    // Rows = feature vec, col = dimensions
    Eigen::MatrixXd features(feature_files.size(), 0);  // cols set later

    for (int i = 0; i < feature_files.size(); i++) {
        Eigen::ArrayXd feature = FlattenFeature(LoadCSV(feature_files[i]));

        if (features.cols() == 0) {
            features.conservativeResize(features.rows(), feature.size());
        }

        assert(feature.size() == features.cols());
        features.row(i) = feature;
    }

    Eigen::VectorXd mean = features.colwise().mean();
    assert(mean.size() == features.cols());
    features.rowwise() -= mean.transpose();

    Eigen::MatrixXd covar = features.transpose() * features;
    covar /= features.rows();  // normalization

    // Assert symmetric and has intended dimensions
    assert(covar.cols() == covar.rows());
    assert(covar.cols() == mean.size());
    assert((covar - covar.transpose())
               .isApprox(Eigen::MatrixXd::Zero(covar.rows(), covar.cols())));

    /***************************************************************
        Eigenvalue decomposition
    ***************************************************************/
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(covar);

    /***************************************************************
        Save output
    ***************************************************************/
    fs::path scree_file = infile;
    scree_file.replace_extension(".scree");
    SaveCSV(scree_file, es.eigenvalues());
    std::cout << scree_file << std::endl;

    fs::path basis_file = infile;
    basis_file.replace_extension(".basis");
    SaveCSV(basis_file, es.eigenvectors());
    std::cout << basis_file << std::endl;

    fs::path mean_file = infile;
    mean_file.replace_extension(".mean");
    SaveCSV(mean_file, mean);
    std::cout << mean_file << std::endl;
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
