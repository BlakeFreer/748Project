#include "fileio.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "colour.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

void SaveCSV(fs::path filename, const Eigen::ArrayXXd& array,
             std::vector<std::string> header, std::string delimiter,
             int precision) {
    std::ofstream of(filename);

    if (!of.is_open()) {
        throw std::runtime_error("Failed to open " + filename.string() +
                                 " for writing.");
    }

    if (!header.empty()) {
        if (header.size() != array.cols()) {
            throw std::invalid_argument(
                "Header length (" + std::to_string(header.size()) +
                ") does not match number of columns in data (" +
                std::to_string(array.cols()) + ").");
        }
        for (int i = 0; i < header.size(); i++) {
            of << header[i];
            if (i == header.size() - 1) {
                of << "\n";
            } else {
                of << ',';
            }
        }
    }

    auto fmt =
        Eigen::IOFormat(precision, Eigen::DontAlignCols, delimiter, "\n");
    of << array.format(fmt);
}

namespace {
bool should_trim(char c) {
    return std::isspace(c) || c == '"';
}
std::string trim_token(const std::string& s) {
    size_t start = 0;
    size_t end = s.size();

    while (start < end && should_trim(s[start])) {
        ++start;
    }
    while (end > start && should_trim(s[end - 1])) {
        --end;
    }

    return s.substr(start, end - start);
}
}  // namespace

Eigen::ArrayXXd LoadCSV(fs::path filename, int skip_lines, char delimiter) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename.string());
    }

    std::vector<std::vector<double>> data;
    std::string line;

    int line_no = 0;
    for (; line_no < skip_lines; line_no++) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            try {
                row.push_back(std::stod(trim_token(token)));
            } catch (const std::invalid_argument& e) {
                std::ostringstream err_msg;
                err_msg << "Invalid value '" << token << "' in line "
                        << (line_no + 1) << " of " << filename.string() << ".";
                if (line_no == 0) {
                    err_msg << " Did you mean to pass skip_lines=1 to skip the "
                               "header?";
                }
                err_msg << " (" << e.what() << ")";
                throw std::invalid_argument(err_msg.str());
            }
        }
        data.push_back(row);
        line_no++;
    }

    file.close();

    if (data.empty()) return Eigen::ArrayXXd(0, 0);

    size_t rows = data.size();
    size_t cols = data[0].size();
    Eigen::ArrayXXd table(rows, cols);

    for (int r = 0; r < rows; r++) {
        if (data[r].size() != cols) {
            throw std::runtime_error("Inconsistent row sizes.");
        }
        for (int c = 0; c < cols; c++) {
            table(r, c) = data[r][c];
        }
    }
    return table;
}

static std::string StripQuotes(std::string s) {
    if (s.size() < 2) {
        return s;
    }

    if ((s.front() == '"' && s.back() == '"') ||
        (s.front() == '\'' && s.back() == '\'')) {
        return s.substr(1, s.size() - 2);
    }

    return s;
}

std::vector<fs::path> ReadFileListing(fs::path list_txt) {
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

void SaveImage(std::filesystem::path filename, Eigen::ArrayXXd values,
               double min, double max) {
    if (filename.extension() != ".png") {
        throw std::runtime_error("Only .png files are supported, not " +
                                 filename.extension().string());
    }

    auto palette = colour::sunset;
    palette.Rescale(min, max);

    int rows = values.rows();
    int cols = values.cols();

    uint8_t* image_buffer = new uint8_t[cols * rows * 3];
    for (int sp = 0; sp < cols; sp++) {
        for (int i = 0; i < rows; i++) {
            int idx = ((rows - 1 - i) * cols + sp) * 3;

            auto col = palette.Get(values(i, sp));

            image_buffer[idx] = col.red;
            image_buffer[idx + 1] = col.green;
            image_buffer[idx + 2] = col.blue;
        }
    }

    stbi_write_png(filename.string().c_str(), cols, rows, 3, image_buffer,
                   cols * 3);
}