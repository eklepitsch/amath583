#!/bin/bash

#SBATCH --job-name=hw4-p7

#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=5G
#SBATCH --gpus=0
#SBATCH --time=00-00:02:00

#SBATCH --chdir=/gscratch/amath/evanklep/hw4
#SBATCH --export=all
#SBATCH --output=hw4_p7-out.txt
#SBATCH --error=hw4_p7-err.txt

# Modules to use (optional).
module load gcc ompi

# Your programs to run.
time mpirun -np 1 ./hw4_p7 a
time mpirun -np 2 ./hw4_p7 a
time mpirun -np 4 ./hw4_p7 a
time mpirun -np 8 ./hw4_p7 a
time mpirun -np 16 ./hw4_p7 a
time mpirun -np 32 ./hw4_p7 a
mpirun -np 8 ./hw4_p7 b 10
mpirun -np 8 ./hw4_p7 b 100
mpirun -np 8 ./hw4_p7 b 1000
mpirun -np 8 ./hw4_p7 b 10000
mpirun -np 8 ./hw4_p7 b 100000
mpirun -np 8 ./hw4_p7 b 1000000
