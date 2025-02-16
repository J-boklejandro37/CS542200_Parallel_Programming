#! /bin/bash

mkdir -p nsys_reports

# Output to ./nsys_reports/hw2a.nsys-rep
nsys profile \
    -o "./nsys_reports/hw2b_$PMI_RANK.nsys-rep" \
    --mpi-impl openmpi \
    --trace mpi,ucx,nvtx,osrt \
    $@