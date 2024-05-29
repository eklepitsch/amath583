#!/bin/bash

#SBATCH --job-name=hw6-p2

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=5G
#SBATCH --gpus=1
#SBATCH --time=00-00:03:00

#SBATCH --chdir=/gscratch/amath/evanklep/hw6
#SBATCH --export=all
#SBATCH --output=hw6_p2-out.txt
#SBATCH --error=hw6_p2-err.txt

# Modules to use (optional).
module load gcc cuda

# Make certain the loader can access the fftw library
export PATH=/gscratch/amath/roche-libs/bin:$PATH
export LD_LIBRARY_PATH=/gscratch/amath/roche-libs/lib:$LD_LIBRARY_PATH

# Clean up any old artifacts
rm p2.o
rm xhw6_p2

# Do the build
nvcc -c -O3 -I. -I/gscratch/amath/roche-libs/include -arch sm_75 p2.cpp
g++ -g -o xhw6_p2 p2.o -L/sw/cuda/12.0.0/lib64 \
   -L/gscratch/amath/roche-libs/lib -lcudart -lm

