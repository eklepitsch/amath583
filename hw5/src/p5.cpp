#include "transpose.hpp"
#include "matrix-utils.hpp"
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <n> <m>" << endl;
        return 1;
    }

    int n = atoi(argv[1]); // Rows
    int m = atoi(argv[2]); // Columns
}