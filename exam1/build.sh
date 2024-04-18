#!/bin/bash

set -x

g++ -c mm-ijk.cpp
g++ -c mm-ikj.cpp
g++ -c mm-jik.cpp
g++ -c mm-jki.cpp
g++ -c mm-kij.cpp
g++ -c mm-kji.cpp

# Build problem 1
g++ -std=c++14 -o xtst-ijk-perms tst-ijk-permutations.cpp mm-ijk.o \
mm-jki.o mm-kij.o mm-jik.o mm-ikj.o mm-kji.o

# Build problem 2
g++ -std=c++14 -o xtst-matrix-swaps tst-matrix-swaps.cpp