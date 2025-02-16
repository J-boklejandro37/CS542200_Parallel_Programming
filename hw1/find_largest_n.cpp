#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <filesystem>

namespace fs = std::filesystem;

int parse_n_value(const std::string& content) {
    size_t n_pos = content.find("\"n\":");
    if (n_pos != std::string::npos) {
        size_t value_start = content.find_first_of("0123456789", n_pos);
        if (value_start != std::string::npos) {
            size_t value_end = content.find_first_not_of("0123456789", value_start);
            std::string n_value_str = content.substr(value_start, value_end - value_start);
            return std::stoi(n_value_str);
        }
    }
    return -1; // Return -1 if "n" is not found or cannot be parsed
}

int find_largest_n() {
    int largest_n = std::numeric_limits<int>::min();
    std::string largest_n_file;

    for (int i = 1; i <= 40; ++i) {
        std::string filename = (i < 10 ? "0" : "") + std::to_string(i) + ".txt";
        filename = "/home/pp24/share/hw1/testcases/" + filename;
        if (fs::exists(filename)) {
            std::ifstream file(filename);
            if (file.is_open()) {
                std::string content((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
                file.close();

                int n_value = parse_n_value(content);
                if (n_value > largest_n) {
                    largest_n = n_value;
                    largest_n_file = filename;
                }
            }
        }
    }

    if (!largest_n_file.empty()) {
        std::cout << "The largest 'n' value is " << largest_n << " in file " << largest_n_file << std::endl;
    } else {
        std::cout << "No valid files found or no 'n' values present" << std::endl;
    }

    return 0;
}

int main() {
    return find_largest_n();
}