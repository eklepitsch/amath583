#!/bin/bash

#SBATCH --job-name=hw5-p4

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=5G
#SBATCH --gpus=1
#SBATCH --time=00-00:05:00

#SBATCH --chdir=/gscratch/amath/evanklep/hw5
#SBATCH --export=all
#SBATCH --output=hw5_p4-out.txt
#SBATCH --error=hw5_p4-err.txt

# Modules to use (optional).
module load gcc cuda

# Make certain the loader can access the fftw library
export PATH=/gscratch/amath/roche-libs/bin:$PATH
export LD_LIBRARY_PATH=/gscratch/amath/roche-libs/lib:$LD_LIBRARY_PATH

# Clean up any old artifacts
rm p4.o
rm xhw5_p4

# Do the build
nvcc -c -O3 -I. -I/gscratch/amath/roche-libs/include -arch sm_75 p4.cpp
g++ -g -o xhw5_p4 p4.o -L/sw/cuda/12.0.0/lib64 \
   -L/gscratch/amath/roche-libs/lib -lcudart -lopenblas -lcublas -lm

# Run the code
./xhw5_p4 8192
