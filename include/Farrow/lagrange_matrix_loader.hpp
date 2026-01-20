#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class LagrangeMatrixLoader {
private:
    std::vector<std::vector<double>> matrix;
    int rows;
    int columns;
    std::string description;

public:
    bool loadFromJSON(const std::string& filename) {
        try {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Cannot open file " << filename << std::endl;
                return false;
            }

            json j;
            file >> j;
            file.close();

            // Parse metadata
            description = j["description"].get<std::string>();
            rows = j["rows"].get<int>();
            columns = j["columns"].get<int>();

            // Parse matrix data
            matrix.clear();
            matrix.reserve(rows);

            for (const auto& row : j["data"]) {
                std::vector<double> matrixRow;
                matrixRow.reserve(columns);
                
                for (const auto& element : row) {
                    matrixRow.push_back(element.get<double>());
                }
                
                matrix.push_back(matrixRow);
            }

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading JSON: " << e.what() << std::endl;
            return false;
        }
    }

    const std::vector<std::vector<double>>& getMatrix() const {
        return matrix;
    }

    int getRows() const { return rows; }
    int getColumns() const { return columns; }
    const std::string& getDescription() const { return description; }

    void printMatrixInfo() const {
        std::cout << "Description: " << description << std::endl;
        std::cout << "Matrix size: " << rows << " x " << columns << std::endl;
        std::cout << "Total elements: " << (rows * columns) << std::endl;
    }

    void printMatrixSample(int sampleRows = 5) const {
        std::cout << "\nFirst " << sampleRows << " rows:\n";
        for (int i = 0; i < std::min(sampleRows, rows); ++i) {
            for (int j = 0; j < columns; ++j) {
                printf("%8.4f ", matrix[i][j]);
            }
            std::cout << "\n";
        }
    }

    double getElement(int row, int col) const {
        if (row >= 0 && row < rows && col >= 0 && col < columns) {
            return matrix[row][col];
        }
        throw std::out_of_range("Matrix index out of range");
    }
};

// int main() {
//     LagrangeMatrixLoader loader;

//     // Load matrix from JSON file
//     if (!loader.loadFromJSON("lagrange_matrix.json")) {
//         return 1;
//     }

//     // Display info
//     loader.printMatrixInfo();
//     loader.printMatrixSample(8);

//     // Access specific element example
//     try {
//         std::cout << "\nElement [0][1] = " << loader.getElement(0, 1) << std::endl;
//         std::cout << "Element [8][0] = " << loader.getElement(8, 0) << std::endl;
//     }
//     catch (const std::exception& e) {
//         std::cerr << e.what() << std::endl;
//     }

//     return 0;
// }
