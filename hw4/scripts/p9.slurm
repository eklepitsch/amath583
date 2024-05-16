#!/bin/bash

#SBATCH --job-name=hw4-p9

#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=5G
#SBATCH --gpus=0
#SBATCH --time=00-00:03:00

#SBATCH --chdir=/gscratch/amath/evanklep/hw4
#SBATCH --export=all
#SBATCH --output=hw4_p9-out.txt
#SBATCH --error=hw4_p9-err.txt

# Modules to use (optional).
module load gcc ompi

# Your programs to run.
mpic++ -o hw4_p9 -I. p9.cpp

# Create a file to store the results
rm p9-results.csv
touch p9-results.csv

# Part a - homebrew broadcast
for ((i = 8 ; i <= 262144 ; i*=2)); do time mpirun -np 4 ./hw4_p9 a $i; done
for ((i = 8 ; i <= 262144 ; i*=2)); do time mpirun -np 32 ./hw4_p9 a $i; done

# Part b - MPI_Bcast
for ((i = 8 ; i <= 262144 ; i*=2)); do time mpirun -np 4 ./hw4_p9 b $i; done
for ((i = 8 ; i <= 262144 ; i*=2)); do time mpirun -np 32 ./hw4_p9 b $i; done