#include "csv.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

void SaveCSV(std::filesystem::path filename, const Eigen::ArrayXXd& array,
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

Eigen::ArrayXXd LoadCSV(std::filesystem::path filename, int skip_lines,
                        char delimiter) {
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