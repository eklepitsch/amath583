#!/bin/bash

set -x
g++ hw3.cpp mm-jki.cpp mm-kij.cpp ref_daxpy.cpp ref_dgemv.cpp ref_dgemm.cpp ref_axpyt.cpp -o hw3
