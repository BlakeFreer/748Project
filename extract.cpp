#include <filesystem>
#include <iostream>
#include <optional>
#include <unordered_map>

namespace fs = std::filesystem;

namespace Algorithm {
enum class Type {
    STFT,
    MEL
};

const std::unordered_map<std::string, Type> keys{
    {"stft", Type::STFT},
    {"mel", Type::MEL},
};

Type Parse(std::string_view s) {
    using enum Type;
    std::string s_lower;
    std::transform(s.begin(), s.end(), std::back_inserter(s_lower), ::tolower);

    if (auto it = keys.find(s_lower); it != keys.end()) {
        return it->second;
    } else {
        std::string msg = "Invalid algorithm. Valid options are: ";
        for (const auto& v : Algorithm::keys) {
            msg.append(v.first);
            msg.append(" ");
        }
        throw std::runtime_error(msg);
    }
}

}  // namespace Algorithm

struct Config {
    fs::path file;
    Algorithm::Type algorithm;

    enum class ParseError {
        MissingArgument,
        InvalidFile
    };

    Config(int argc, char* argv[]) {
        using enum ParseError;

        if (argc < 3) throw std::runtime_error("Missing argument(s).");

        file = fs::path(argv[1]);
        if (!fs::exists(file)) throw std::runtime_error("File does not exist.");

        algorithm = Algorithm::Parse(argv[2]);
    }
};

int main(int argc, char* argv[]) {
    std::optional<Config> c;
    try {
        c.emplace(argc, argv);
    } catch (const std::runtime_error& e) {
        std::cerr << "Usage: ./extract <filename> <method>" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << c->file << std::endl;

    return 0;
}