#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstdlib>
#include <ctime>
#include <sstream> //  std::istringstream
#include <cmath>
#include <mutex>
#include <thread>


#define MAX_THREADS 16

extern void updateGrid(std::vector<std::vector<bool>>& grid, int gridSize,
                       int numThreads);

void generateGrid(std::vector<std::vector<bool>>& grid, int gridSize) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < gridSize; ++i) {
        std::vector<bool> row;
        for (int j = 0; j < gridSize; ++j) {
            row.push_back(dis(gen));
        }
        grid.push_back(row);
    }
}

void saveGridToFile(const std::vector<std::vector<bool>>& grid, int gridSize, const std::string& filename) {
    std::ofstream outfile(filename);

    if (outfile.is_open()) {
        for (int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                outfile << (grid[i][j] ? "1 " : "0 ");
            }
            outfile << "\n";
        }

        outfile.close();
        std::cout << "State saved to file: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void readInitialStateFromFile(std::vector<std::vector<bool>>& grid, int gridSize, const std::string& filename) {
    std::ifstream infile(filename);
    
    if (infile.is_open()) {
        grid.clear();
        std::string line;
        while (std::getline(infile, line)) {
            std::vector<bool> row;
            std::istringstream iss(line);
            int value;
            while (iss >> value) {
                row.push_back(value == 1);
            }
            grid.push_back(row);
        }
        infile.close();
        std::cout << "Initial state read from file: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void printGrid(const std::vector<std::vector<bool>>& grid) {
    for (const auto& row : grid) {
        for (bool cell : row) {
            std::cout << (cell ? " ■ " : "   ");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <gridSize> <numThreads>" << std::endl;
        return 1;
    }

    int gridSize = std::stoi(argv[1]);
    int numThreads = std::stoi(argv[2]);

    if (gridSize % 4 != 0 || (numThreads != 1 && numThreads != 4 && numThreads != 16)) {
        std::cerr << "Grid size must be a multiple of 4 and numThreads must be 1, 4, or 16." << std::endl;
        return 1;
    }

    std::vector<std::vector<bool>> grid;
    // generateGrid(grid, gridSize);
    // saveGridToFile(grid, gridSize, "./artifacts/conway_initial_state_" + std::to_string(gridSize) + ".txt");

    // Read initial state from file
    readInitialStateFromFile(grid, gridSize, "./artifacts/conway_initial_state_" + std::to_string(gridSize) + ".txt");

    // Call updateGrid function with the initial state
    updateGrid(grid, gridSize, numThreads);

    saveGridToFile(grid, gridSize, "./artifacts/conway_final_state_" + std::to_string(gridSize) + ".txt");

    return 0;
}

