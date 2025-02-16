#! /bin/bash

mkdir -p nsys_reports

# Output to ./nsys_reports/hw2a.nsys-rep
nsys profile \
    -o "./nsys_reports/hw2a.nsys-rep" \
    --trace nvtx,osrt \
    $@