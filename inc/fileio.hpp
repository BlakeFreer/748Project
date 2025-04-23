#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <vector>

void SaveCSV(std::filesystem::path filename, const Eigen::ArrayXXd& array,
             std::vector<std::string> header = {}, std::string delimiter = ",",
             int precision = Eigen::FullPrecision);

Eigen::ArrayXXd LoadCSV(std::filesystem::path filename, int skip_lines = 0,
                        char delimiter = ',');

std::vector<std::filesystem::path> ReadFileListing(
    std::filesystem::path list_txt);

void SaveImage(std::filesystem::path filename, Eigen::ArrayXXd values,
               double min, double max);