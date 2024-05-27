#include<iostream>
#include<complex>
#include<cstdlib>
#include<string>
#include<cmath>
#include<vector>
#include<chrono>
#include<limits>
#include<blas.h>
#include<lapacke.h>

int main() {
//...
    a=(std::complex<double>*)malloc(sizeof(std::complex<double>) * ma * na);
    b=(std::complex<double>*)malloc(sizeof(std::complex<double>) * ma);
    z=(std::complex<double>*)malloc(sizeof(std::complex<double>) * na);
//...
    srand(0);
    int k=0;
    for(int j=0; j<na; j++) {
        for(int i=0; i<ma; i++) {
            a[k]=0.5-(double)rand()/(double)RAND_MAX +std::complex<double>(0,1) *(0.5-(double)rand()/(double)RAND_MAX);
            if(i==j) a[k]*=static_cast<double>(ma);
            k++;
        }
    }
    srand(1);
    for(int i=0; i<ma; i++) {
        b[i]=0.5-(double)rand()/(double)RAND_MAX +std::complex<double>(0,1) *(0.5-(double)rand()/(double)RAND_MAX);
    }
//...
}
