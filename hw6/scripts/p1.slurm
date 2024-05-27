#!/bin/bash

#SBATCH --job-name=hw6-p1

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=5G
#SBATCH --gpus=0
#SBATCH --time=00-00:02:00

#SBATCH --chdir=/gscratch/amath/evanklep/hw6
#SBATCH --export=all
#SBATCH --output=hw6_p1-out.txt
#SBATCH --error=hw6_p1-err.txt

# Modules to use (optional).
module load gcc

# Make certain the loader can access the fftw library
export PATH=/gscratch/amath/roche-libs/bin:$PATH
export LD_LIBRARY_PATH=/gscratch/amath/roche-libs/lib:$LD_LIBRARY_PATH

# Clean up any old artifacts
rm xhw6_p1

# Do the build
g++ -g -O3 -o xhw6_p1 -I. -I/gscratch/amath/roche-libs/include \
   p1.cpp -L/gscratch/amath/roche-libs/lib -lopenblas -lm

# Run the code
./xhw6_p1 8192
