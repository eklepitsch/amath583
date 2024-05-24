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

std::mutex cout_mtx;

void kernel(std::vector<std::vector<bool>>& grid,
            std::vector<std::vector<bool>>& newGrid,
            int startingRow, int endingRow,
            int startingColumn, int endingColumn,
            int rank)
{
    if(startingRow < 0 || startingRow > grid.size() ||
       endingRow < 0 || endingRow > grid.size() ||
       startingColumn < 0 || startingColumn > grid[0].size() ||
       endingColumn < 0 || endingColumn > grid[0].size() ||
       startingRow > endingRow || startingColumn > endingColumn)
    {
        cout_mtx.lock();
        std::cout << "Invalid bounds for thread " << rank << std::endl;
        cout_mtx.unlock();
        return;
    }

    for(int i=startingRow; i<endingRow; ++i)
    {
        for(int j=startingColumn; j<endingColumn; ++j)
        {
            int numAliveNeighbors = 0;
            for(int x=-1; x<=1; ++x)
            {
                for(int y=-1; y<=1; ++y)
                {
                    if(x == 0 && y == 0)
                    {
                        // This is the current cell
                        continue;
                    }

                    int neighborX = i + x;
                    int neighborY = j + y;

                    if(neighborX < 0 || neighborY < 0 ||
                       neighborX >= grid.size() || neighborY >= grid[0].size())
                    {
                        // This neighbor is out of bounds
                        continue;
                    }
                    else if(grid[neighborX][neighborY])
                    {
                        // Valid neighbor
                        numAliveNeighbors++;
                    }
                }
            }
            if(grid[i][j])
            {
                newGrid[i][j] = (numAliveNeighbors == 2 || numAliveNeighbors == 3);
            }
            else
            {
                newGrid[i][j] = (numAliveNeighbors == 3);
            }
        }
    }
}

void updateGrid(std::vector<std::vector<bool>>& grid, int gridSize, int numThreads) {
    // Implementation of updateGrid function
    // Add your existing implementation here
    std::vector<std::vector<bool>> newGrid(
        gridSize, std::vector<bool>(gridSize, false));

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for(int i=0; i<numThreads; ++i)
    {
        // numThreads is 1, 4, or 16, so the following division is valid
        int subgridSize = gridSize / std::sqrt(numThreads);

        int startingColumn = (i * subgridSize) % gridSize;
        int startingRow = ((i * subgridSize) / gridSize) * subgridSize;
        int endingColumn = startingColumn + subgridSize;
        int endingRow = startingRow + subgridSize;
        threads.emplace_back(kernel, std::ref(grid), std::ref(newGrid),
                             startingRow, endingRow, startingColumn, endingColumn,
                             i);
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    grid = newGrid;
}
