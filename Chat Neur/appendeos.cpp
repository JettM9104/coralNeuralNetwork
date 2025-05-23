#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main() {
    std::string enter;

    std::cout << "To confirm, please type <eos";
    std::cin >> enter;

    if (enter != "<EOS>") {
        std::cerr << "you prolly misclicked";
        return -1;
    }
    std::ifstream inputFile("text.txt");
    if (!inputFile) {
        std::cerr << "Error: Could not open text.txt for reading.\n";
        return 1;
    }

    std::vector<std::string> lines;
    std::string line;

    // Read lines and append <eos>
    while (std::getline(inputFile, line)) {
        lines.push_back(line + "<eos>");
    }
    inputFile.close();

    std::ofstream outputFile("text.txt");
    if (!outputFile) {
        std::cerr << "Error: Could not open text.txt for writing.\n";
        return 1;
    }

    for (const auto& modifiedLine : lines) {
        outputFile << modifiedLine << '\n';
    }

    outputFile.close();
    std::cout << "Finished appending <eos> to each line in text.txt.\n";
    return 0;
}
