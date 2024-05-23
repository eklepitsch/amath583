#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <sstream> //  std::istringstream

#define GRID_SIZE_X 20
#define GRID_SIZE_Y 20

// Function to generate the initial grid randomly
void generateGrid(std::vector<std::vector<bool>>& grid) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < GRID_SIZE_X; ++i) {
        std::vector<bool> row;
        for (int j = 0; j < GRID_SIZE_Y; ++j) {
            row.push_back(dis(gen));
        }
        grid.push_back(row);
    }
}

// Function to print the grid
void printGrid(const std::vector<std::vector<bool>>& grid) {
    for (int i = 0; i < GRID_SIZE_X; ++i) {
        for (int j = 0; j < GRID_SIZE_Y; ++j) {
            std::cout << (grid[i][j] ? " ■ " : "   ");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to update the grid for each generation
void updateGrid(std::vector<std::vector<bool>>& grid) {
    std::vector<std::vector<bool>> newGrid(GRID_SIZE_X, std::vector<bool>(GRID_SIZE_Y, false));

    for (int i = 0; i < GRID_SIZE_X; ++i) {
        for (int j = 0; j < GRID_SIZE_Y; ++j) {
            int aliveNeighbors = 0;

            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    if (x == 0 && y == 0) continue;

                    int neighborX = i + x;
                    int neighborY = j + y;

                    if (neighborX >= 0 && neighborX < GRID_SIZE_X && neighborY >= 0 && neighborY < GRID_SIZE_Y) {
                        if (grid[neighborX][neighborY]) {
                            aliveNeighbors++;
                        }
                    }
                }
            }

            if (grid[i][j]) {
                if (aliveNeighbors < 2 || aliveNeighbors > 3) {
                    newGrid[i][j] = false;
                } else {
                    newGrid[i][j] = true;
                }
            } else {
                if (aliveNeighbors == 3) {
                    newGrid[i][j] = true;
                }
            }
        }
    }

    grid = newGrid;
}

// Function to save the grid to a file
void saveGridToFile(const std::vector<std::vector<bool>>& grid) {
    std::ofstream outfile("./artifacts/conway_grid.txt");

    if (outfile.is_open()) {
        for (int i = 0; i < GRID_SIZE_X; ++i) {
            for (int j = 0; j < GRID_SIZE_Y; ++j) {
                outfile << (grid[i][j] ? "1 " : "0 ");
            }
            outfile << "\n";
        }

        outfile.close();
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

int main() {
    std::vector<std::vector<bool>> grid;
    // generateGrid(grid);
    auto gridSize = GRID_SIZE_X;
    readInitialStateFromFile(grid, gridSize, "./artifacts/conway_initial_state_" + std::to_string(gridSize) + ".txt");
    
    int generations = 1; // Number of generations to simulate

    for (int gen = 0; gen < generations; ++gen) {
        std::system("python3 ./src/conway_display.py"); // Call the Python script to display the grid
        
        printGrid(grid);
        updateGrid(grid);
        saveGridToFile(grid);

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    return 0;
}

