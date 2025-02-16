#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    std::string filename = argv[1];
    std::ifstream file(filename, std::ios::binary);
    std::vector<float> numbers;
    float number;

    if (file.is_open()) {
        while (file.read(reinterpret_cast<char*>(&number), sizeof(float))) {
            numbers.push_back(number);
        }
        file.close();

        // Print the numbers to verify
        for (const auto& num : numbers) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Unable to open file" << std::endl;
    }

    return 0;
}