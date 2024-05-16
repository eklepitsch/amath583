// @uw.edu
// AMATH 483-583
// strassen.cpp : starter code for Strassen implementation

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

template <typename T>
vector<vector<T>> addMatrix(const vector<vector<T>> &A, const vector<vector<T>> &B)
{
    int n = A.size();
    int m = A[0].size();
    vector<vector<T>> C(n, vector<T>(m));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

template <typename T>
vector<vector<T>> subtractMatrix(const vector<vector<T>> &A, const vector<vector<T>> &B)
{
    int n = A.size();
    int m = A[0].size();
    vector<vector<T>> C(n, vector<T>(m));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

template <typename T>
vector<vector<T>> strassenMultiply(const vector<vector<T>> &A, const vector<vector<T>> &B)
{
    // Bounds check
    int rowsA = A.size();
    int rowsB = B.size();
    if(rowsA != rowsB) {throw std::invalid_argument("A and B must have matching rows");}
    if(!rowsA || !rowsB) {throw std::invalid_argument("Matrix dimension is zero");}
    int colsA = A[0].size();
    int colsB = B[0].size();
    if(colsA != colsB) {throw std::invalid_argument("A and B must have matching cols");}
    if(!colsA || !colsB) {throw std::invalid_argument("Matrix dimension is zero");}
    if(std::ceil(std::log2(rowsA)) != std::floor(std::log2(rowsA))){
        throw std::invalid_argument("Dim of A is not a power of 2");}
    if(std::ceil(std::log2(rowsB)) != std::floor(std::log2(rowsB))){
        throw std::invalid_argument("Dim of B is not a power of 2");}
    if(std::ceil(std::log2(colsA)) != std::floor(std::log2(colsA))){
        throw std::invalid_argument("Dim of A is not a power of 2");}
    if(std::ceil(std::log2(colsB)) != std::floor(std::log2(colsB))){
        throw std::invalid_argument("Dim of B is not a power of 2");}

    // If we make it here, we know that A and B are square matrices with
    // dimensions of powers of 2 (requirement for Strassen).

    // Base case
    if(1 == rowsA)
    {
        vector<vector<T>> base;
        base.push_back(vector<T>(1, A[0][0] * B[0][0]));
        return base;
    }

    // Recursive case
    auto split = rowsA / 2;
    vector<vector<T>> Atl(split, vector<T>(split));  // top-left
    vector<vector<T>> Atr(split, vector<T>(split));  // top-right
    vector<vector<T>> Abl(split, vector<T>(split));  // bottom-left
    vector<vector<T>> Abr(split, vector<T>(split));  // bottom-right
    vector<vector<T>> Btl(split, vector<T>(split));
    vector<vector<T>> Btr(split, vector<T>(split));
    vector<vector<T>> Bbl(split, vector<T>(split));
    vector<vector<T>> Bbr(split, vector<T>(split));

    for(int i=0; i<split; ++i)
    {
        for(int j=0; j<split; ++j)
        {
            Atl[i][j] = A[i][j];
            Atr[i][j] = A[i][j + split];
            Abl[i][j] = A[i + split][j];
            Abr[i][j] = A[i + split][j + split];

            Btl[i][j] = B[i][j];
            Btr[i][j] = B[i][j + split];
            Bbl[i][j] = B[i + split][j];
            Bbr[i][j] = B[i + split][j + split];
        }
    }

    // Submatrices M1 - M7 are recursively defined by Strassen algorithm
    std::vector<std::vector<T>> M1 = strassenMultiply(addMatrix(Atl, Abl),
                                                      addMatrix(Btl, Btr));
    std::vector<std::vector<T>> M2 = strassenMultiply(addMatrix(Atr, Abr),
                                                      addMatrix(Bbl, Bbr));
    std::vector<std::vector<T>> M3 = strassenMultiply(subtractMatrix(Atl, Abr),
                                                      addMatrix(Btl, Bbr));
    std::vector<std::vector<T>> M4 = strassenMultiply(Atl,
                                                      subtractMatrix(Btr, Bbr));
    std::vector<std::vector<T>> M5 = strassenMultiply(addMatrix(Abl, Abr),
                                                      Btl);
    std::vector<std::vector<T>> M6 = strassenMultiply(addMatrix(Atl, Atr),
                                                      Bbr);
    std::vector<std::vector<T>> M7 = strassenMultiply(Abr,
                                                      subtractMatrix(Bbl, Btl));

    std::vector<std::vector<T>> I = addMatrix(M2, subtractMatrix(M3, addMatrix(M6, M7)));
    std::vector<std::vector<T>> J = addMatrix(M4, M6);
    std::vector<std::vector<T>> K = addMatrix(M5, M7);
    std::vector<std::vector<T>> L = subtractMatrix(M1, addMatrix(M3, addMatrix(M4, M5)));

    // Reassenble submatrices
    std::vector<std::vector<T>> C(rowsA, vector<T>(colsA));  // C has same dimensions as A,B
    for(int i=0; i<split; ++i)
    {
        for(int j=0; j<split; ++j)
        {
            C[i][j] = I[i][j];
            C[i][j + split] = J[i][j];
            C[i + split][j] = K[i][j];
            C[i + split][j + split] = L[i][j];
        }
    }
    return C;
}

template <typename T>
void printMatrix(const vector<vector<T>> &matrix)
{
    int n = matrix.size();
    int m = matrix[0].size();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// int
template vector<vector<int>> addMatrix<int>(const vector<vector<int>> &A, const vector<vector<int>> &B);
template vector<vector<int>> subtractMatrix<int>(const vector<vector<int>> &A, const vector<vector<int>> &B);
template vector<vector<int>> strassenMultiply<int>(const vector<vector<int>> &A, const vector<vector<int>> &B);
template void printMatrix<int>(const vector<vector<int>> &matrix);
// double
template vector<vector<double>> addMatrix<double>(const vector<vector<double>> &A, const vector<vector<double>> &B);
template vector<vector<double>> subtractMatrix<double>(const vector<vector<double>> &A, const vector<vector<double>> &B);
template vector<vector<double>> strassenMultiply<double>(const vector<vector<double>> &A, const vector<vector<double>> &B);
template void printMatrix<double>(const vector<vector<double>> &matrix);